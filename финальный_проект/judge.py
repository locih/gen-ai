"""LLM-as-judge для финального ответа."""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import JudgeVerdict, MetaCategory, TicketActionResponse, TicketClassification

client = make_client()
MODEL = get_model()

JUDGE_SYSTEM = """Ты — ревизор L1 helpdesk. Оцени ответ системы на тикет.

Правила:
- category_correct: совпадает ли meta_category с gold (если gold задан).
- steps_supported: шаги логичны для категории и не противоречат контексту KB.
- overall_score: 1.0 если оба true и шаги конкретны; 0.5 если категория верна, но шаги общие;
  0.0 если категория явно неверна или шаги выдуманы.
- comment: одна фраза на русском."""


def judge_ticket(
    ticket_text: str,
    classification: TicketClassification,
    action: TicketActionResponse,
    kb_context: str,
    gold_meta: MetaCategory | None = None,
) -> JudgeVerdict:
    gold_line = f"Gold meta_category: {gold_meta}" if gold_meta else "Gold: не задан"
    user = f"""Тикет:
{ticket_text[:2000]}

{gold_line}

Классификация: {classification.model_dump_json()}
Ответ L1: {action.model_dump_json()}

Фрагменты KB:
{kb_context[:4000]}
"""
    return client.chat.completions.create(
        model=MODEL,
        response_model=JudgeVerdict,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
