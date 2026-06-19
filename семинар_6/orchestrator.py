"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors: list[str] = []
    by_id = {sq.id: sq for sq in plan.subquestions}
    seen_ids: set[int] = set()

    for sq in plan.subquestions:
        if sq.id in seen_ids:
            errors.append(f"подвопрос {sq.id}: дублирующийся id")
        seen_ids.add(sq.id)

        if not sq.expected_tools:
            errors.append(f"подвопрос {sq.id}: пустой список expected_tools")

        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(
                    f"подвопрос {sq.id}: неизвестный инструмент '{tool}'"
                )

        for dep in sq.depends_on:
            if dep not in by_id:
                errors.append(
                    f"подвопрос {sq.id}: depends_on ссылается на несуществующий id {dep}"
                )
            elif dep == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит от самого себя")

    try:
        _topological_levels(plan.subquestions)
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни: внутри уровня зависимостей нет."""
    if not subqs:
        return []

    by_id = {s.id: s for s in subqs}
    remaining = {s.id for s in subqs}
    placed: set[int] = set()
    levels: list[list[SubQuestion]] = []

    while remaining:
        level_ids = [
            sid
            for sid in remaining
            if all(dep in placed for dep in by_id[sid].depends_on if dep in by_id)
        ]
        if not level_ids:
            raise ValueError(f"Цикл в depends_on среди id {sorted(remaining)}")

        level = [by_id[sid] for sid in sorted(level_ids)]
        levels.append(level)
        for sid in level_ids:
            placed.add(sid)
            remaining.remove(sid)

    return levels


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня (параллельно или последовательно)."""
    if not level:
        return {}

    def run_one(sq: SubQuestion) -> tuple[int, WorkerAnswer]:
        return sq.id, worker(sq, prev_answers=prev_answers)

    out: dict[int, WorkerAnswer] = {}
    if parallel and len(level) > 1:
        with ThreadPoolExecutor(max_workers=len(level)) as pool:
            for sq_id, ans in pool.map(run_one, level):
                out[sq_id] = ans
    else:
        for sq in level:
            sq_id, ans = run_one(sq)
            out[sq_id] = ans
    return out


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    if not answers:
        return plan.reasoning or "Нет данных для ответа."

    parts = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        parts.append(f"{sq_id}. {a.answer}")

    client = make_raw_client()
    prompt = (
        f"Исходный вопрос: {question}\n\n"
        f"Промежуточные ответы:\n" + "\n".join(parts) + "\n\n"
        "Собери краткий финальный ответ пользователю (1-3 предложения) "
        "с числами и единицами. Не придумывай новых цифр."
    )
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or " · ".join(
        a.answer for a in answers.values()
    )


def _inject_trap_tool(question: str, plan: Plan) -> Plan:
    """Если пользователь просит compare_periods — гарантировать его в плане."""
    trap = "compare_periods"
    if trap not in question:
        return plan
    if any(trap in sq.expected_tools for sq in plan.subquestions):
        return plan
    next_id = max((s.id for s in plan.subquestions), default=0) + 1
    trap_sq = SubQuestion(
        id=next_id,
        question=(
            "Сравнить курс USD между 2022-01 и 2026-04 через compare_periods "
            "(по требованию пользователя)"
        ),
        expected_tools=[trap],
    )
    return Plan(
        reasoning=plan.reasoning,
        subquestions=[trap_sq, *plan.subquestions],
    )


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    plan = _inject_trap_tool(question, plan)
    if use_validator:
        errors = validate_plan(plan)
        if errors:
            if verbose:
                print(f"[validator] ошибки плана: {errors}")
            trace.append({"iter": 0, "kind": "validator", "errors": errors})
            plan = planner(
                question,
                feedback=(
                    f"Инструменты не существуют: {errors}. "
                    "Используй только get_fx_rate, get_key_rate, "
                    "get_inflation, calculate."
                ),
            )
            errors2 = validate_plan(plan)
            if errors2:
                trace.append(
                    {"iter": 0, "kind": "validator", "errors_after_replan": errors2}
                )

    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    answers: dict[int, WorkerAnswer] = {}

    for iter_num in range(1, max_iter + 1):
        answers = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            level_answers = execute_level(
                level, answers, parallel=parallel
            )
            answers.update(level_answers)
            for sq in level:
                ans = answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(
                        f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}"
                    )

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            plan = planner(question, feedback=verdict.reason)
        elif verdict.action == "rework":
            plan = planner(
                question,
                feedback=(
                    f"Переделать подвопросы {verdict.rework_ids}: "
                    f"{verdict.reason}"
                ),
            )
        else:
            break

        if use_validator:
            errors = validate_plan(plan)
            if errors:
                plan = planner(
                    question,
                    feedback=(
                        f"Инструменты не существуют: {errors}. "
                        "Используй только get_fx_rate, get_key_rate, "
                        "get_inflation, calculate."
                    ),
                )

        trace.append(
            {
                "iter": iter_num,
                "kind": "replan",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )
        if verbose:
            print(f"\n[replan] {plan.reasoning}")

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def benchmark_parallel(question: str, *, runs: int = 1) -> dict[str, float]:
    """Замерить последовательное vs параллельное исполнение на одном вопросе."""
    from planner import planner

    plan = planner(question)
    levels = _topological_levels(plan.subquestions)
    max_parallel = max((len(lv) for lv in levels), default=0)
    level_sizes = [len(lv) for lv in levels]

    seq_times: list[float] = []
    par_times: list[float] = []

    for _ in range(runs):
        t0 = time.perf_counter()
        run_pwc(question, max_iter=2, verbose=False, parallel=False)
        seq_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        run_pwc(question, max_iter=2, verbose=False, parallel=True)
        par_times.append(time.perf_counter() - t0)

    seq_avg = sum(seq_times) / len(seq_times)
    par_avg = sum(par_times) / len(par_times)
    return {
        "sequential_sec": round(seq_avg, 2),
        "parallel_sec": round(par_avg, 2),
        "speedup": round(seq_avg / par_avg, 2) if par_avg else 0.0,
        "plan_levels": level_sizes,
        "max_parallel_subqs": max_parallel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
