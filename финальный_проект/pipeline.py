"""
Двухэтапный пайплайн: классификация → RAG → «что делать».

Запуск:
    python pipeline.py "Не работает СДО, пропали задачи"
    python pipeline.py --no-judge "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from critic import critique_action
from judge import judge_ticket
from llm_client import get_model, make_client
from rag import check_quotes_in_corpus, format_context, hybrid_retrieve
from schema import (
    META_CATEGORIES,
    MetaCategory,
    PipelineResult,
    TicketActionResponse,
    TicketClassification,
)

ROOT = Path(__file__).resolve().parent
LABEL_MAP_PATH = ROOT / "input" / "label_map.json"
TRACE_PATH = ROOT / "output" / "trace.jsonl"

client = make_client()
MODEL = get_model()


def load_label_map() -> dict:
    return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))


def gold_meta_from_label(label_id: int, label_map: dict) -> MetaCategory:
    return label_map["label_to_meta"].get(str(label_id), "other")  # type: ignore[return-value]


def _normalize_category(text: str, cls: TicketClassification) -> TicketClassification:
    """
    Правила поверх LLM-классификации.
    Нужны для частых промахов request/account на операционных тикетах.
    """
    t = (text + " " + cls.summary).lower()
    cat = cls.meta_category

    # Явные сигналы предметных категорий.
    if any(k in t for k in ["почт", "outlook", "mail", "str.mos.ru"]):
        cat = "email"
    if any(k in t for k in ["принтер", "мфу", "картридж", "телефон не работает", "не включается компьютер"]):
        cat = "hardware"
    if any(k in t for k in ["арм", "рабочее место", "подключение компьютера", "перемещ", "настройка рабочего места"]):
        cat = "workplace"
    if any(k in t for k in ["сдо", "мосэдо", "vdi", "oracle", "доступ", "разблок", "парол"]):
        # В нашем gold-маппинге password/dostup сценарии чаще относятся к access.
        cat = "access"
    if any(k in t for k in ["установ", "криптопро", "сапр", "ас договор", "модул"]) and "рабочее место" not in t:
        cat = "software"
    if any(k in t for k in ["инцидент", "массов", "авар", "простой"]):
        cat = "incident"
    if any(k in t for k in ["нового сотрудника", "создать учетн", "создать учётн"]):
        cat = "account"

    if cat == cls.meta_category:
        return cls
    return cls.model_copy(update={"meta_category": cat, "confidence": min(cls.confidence, 0.85)})


CLASSIFY_SYSTEM = f"""Ты — диспетчер IT ServiceDesk. Классифицируй обращение пользователя.

Допустимые meta_category (выбери одну):
{', '.join(META_CATEGORIES)}

priority:
- critical — массовый сбой, простой ключевой системы
- high — пользователь не может работать
- medium — частичная деградация
- low — консультация, некритично

Извлеки affected_systems из текста (СДО, Oracle, VDI, почта, принтер…).
confidence: 0.9+ если категория очевидна, <0.7 если текст неоднозначен."""


ACTION_SYSTEM = """Ты — L1 инженер helpdesk. По тикету и фрагментам runbook составь план действий.

Правила:
1. Опирайся ТОЛЬКО на контекст KB ниже. Не выдумывай системы и процедуры.
2. suggested_steps — 2–5 конкретных шагов для оператора L1.
3. quotes — 1–3 короткие дословные цитаты из контекста (не пересказ).
4. sources — id блоков [source__N] из контекста.
5. escalation_team — из runbook или «ServiceDesk L2».
6. meta_category должна совпадать с переданной классификацией.
7. confidence < 0.5 если контекст не покрывает проблему."""


def classify_ticket(text: str) -> TicketClassification:
    cls: TicketClassification = client.chat.completions.create(
        model=MODEL,
        response_model=TicketClassification,
        max_retries=3,
        temperature=0.1,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Тикет:\n{text[:3000]}"},
        ],
    )
    return _normalize_category(text, cls)


def generate_action(
    text: str,
    classification: TicketClassification,
    hits: dict,
    *,
    feedback: str | None = None,
) -> TicketActionResponse:
    ctx = format_context(hits)
    prompt = (
        f"Тикет:\n{text[:2000]}\n\n"
        f"Классификация: category={classification.meta_category}, "
        f"priority={classification.priority}, systems={classification.affected_systems}\n\n"
        f"Контекст KB:\n{ctx}\n\n"
        "Составь TicketActionResponse."
    )
    if feedback:
        prompt += f"\n\nРевизор отклонил предыдущий ответ: {feedback}\nИсправь план."
    action: TicketActionResponse = client.chat.completions.create(
        model=MODEL,
        response_model=TicketActionResponse,
        max_retries=3,
        temperature=0.2,
        messages=[
            {"role": "system", "content": ACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    if action.meta_category != classification.meta_category:
        action = action.model_copy(update={"meta_category": classification.meta_category})
    return action


def generate_action_with_critic(
    text: str,
    classification: TicketClassification,
    hits: dict,
    *,
    max_rework: int = 2,
) -> tuple[TicketActionResponse, int, list[str], bool]:
    """Сгенерировать план + цикл rework по критику (семинар 5/6)."""
    ctx = format_context(hits)
    issues: list[str] = []
    action = generate_action(text, classification, hits)
    reworks = 0

    while True:
        verdict = critique_action(text, classification, action, ctx)
        if verdict.ok:
            return action, reworks, issues, True
        issues.append(verdict.issue)
        if reworks >= max_rework:
            return action, reworks, issues, False
        action = generate_action(text, classification, hits, feedback=verdict.issue)
        reworks += 1


def _log_trace(record: dict) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    ticket_text: str,
    *,
    gold_label_id: int | None = None,
    use_judge: bool = True,
    use_critic: bool = True,
    run_id: str | None = None,
) -> PipelineResult:
    label_map = load_label_map()
    gold_meta = (
        gold_meta_from_label(gold_label_id, label_map) if gold_label_id is not None else None
    )
    rid = run_id or str(uuid.uuid4())
    tools: list[str] = []

    # Этап 1
    classification = classify_ticket(ticket_text)
    tools.append("classify")

    # Этап 2 — RAG
    query = f"{classification.summary} {classification.meta_category} {' '.join(classification.affected_systems)}"
    hits = hybrid_retrieve(query, k=5)
    tools.append("search_kb")

    critic_reworks = 0
    critic_issues: list[str] = []
    critic_ok: bool | None = None
    if use_critic:
        action, critic_reworks, critic_issues, critic_ok = generate_action_with_critic(
            ticket_text, classification, hits
        )
        tools.append("critic")
        if critic_reworks:
            tools.append(f"rework_x{critic_reworks}")
    else:
        action = generate_action(ticket_text, classification, hits)
    tools.append("generate_action")

    ctx = format_context(hits)
    all_quotes = list(action.quotes) + [
        s.quote for s in action.suggested_steps if s.quote
    ]
    ghosts = check_quotes_in_corpus(all_quotes, ctx)

    verdict = None
    if use_judge:
        verdict = judge_ticket(ticket_text, classification, action, ctx, gold_meta)
        tools.append("judge")

    result = PipelineResult(
        ticket_text=ticket_text,
        gold_label_id=gold_label_id,
        gold_meta=gold_meta,
        classification=classification,
        action=action,
        retrieved_sources=hits["ids"],
        ghost_quotes=ghosts,
        steps=3 if use_judge else 2,
        tools_used=tools,
        judge=verdict,
        critic_reworks=critic_reworks,
        critic_issues=critic_issues,
        critic_ok=critic_ok,
    )

    _log_trace(
        {
            "run_id": rid,
            "gold_label_id": gold_label_id,
            "gold_meta": gold_meta,
            "predicted_meta": classification.meta_category,
            "priority": classification.priority,
            "tools_used": tools,
            "retrieved": hits["ids"],
            "ghost_quotes": len(ghosts),
            "critic_reworks": critic_reworks,
            "critic_issues": critic_issues,
            "critic_ok": critic_ok,
            "judge_score": verdict.overall_score if verdict else None,
            "judge_steps_supported": verdict.steps_supported if verdict else None,
            "category_ok": (
                classification.meta_category == gold_meta if gold_meta else None
            ),
        }
    )
    return result


def _print_result(r: PipelineResult) -> None:
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ")
    print(f"  meta_category: {r.classification.meta_category}")
    print(f"  priority:      {r.classification.priority}")
    print(f"  systems:       {r.classification.affected_systems}")
    print(f"  summary:       {r.classification.summary}")
    print(f"  confidence:    {r.classification.confidence:.2f}")
    if r.gold_meta:
        ok = r.classification.meta_category == r.gold_meta
        print(f"  gold_meta:     {r.gold_meta} {'✓' if ok else '✗'}")

    print("\nЧТО ДЕЛАТЬ (RAG)")
    print(f"  escalation:    {r.action.escalation_team}")
    for s in r.action.suggested_steps:
        print(f"  {s.step}. {s.action}")
    print(f"  sources:       {', '.join(r.retrieved_sources)}")
    print(f"  ghost_quotes:  {len(r.ghost_quotes)}")
    if r.ghost_quotes:
        for g in r.ghost_quotes[:3]:
            print(f"    ! {g[:60]}…")

    if r.judge:
        print("\nJUDGE")
        print(f"  score: {r.judge.overall_score:.2f} — {r.judge.comment}")

    if r.critic_reworks or r.critic_issues:
        print("\nCRITIC")
        print(f"  ok:      {r.critic_ok}")
        print(f"  reworks: {r.critic_reworks}")
        for issue in r.critic_issues:
            print(f"  - {issue}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Helpdesk pipeline")
    parser.add_argument("ticket", nargs="?", help="Текст тикета")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--json", action="store_true", help="Вывод JSON в stdout")
    args = parser.parse_args()

    if not args.ticket:
        parser.print_help()
        sys.exit(1)

    result = run_pipeline(
        args.ticket,
        use_judge=not args.no_judge,
        use_critic=not args.no_critic,
    )
    if args.json:
        print(result.model_dump_json(indent=2, ensure_ascii=False))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
