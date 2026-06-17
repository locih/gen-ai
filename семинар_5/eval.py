"""
Eval макро-агента: 10 вопросов, hit-rate по инструментам и must_have.

Запуск: python eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый — один инструмент.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова get_fx_rate с разными датами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Многостадийный кейс из стартера.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Правило 72.",
    },
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "ДЗ: compare_periods fx_USD.",
    },
    {
        "id": 6,
        "query": "На сколько процентных пунктов изменилась ключевая ставка с февраля 2022 по март 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "ДЗ: compare_periods key_rate, delta.",
    },
    {
        "id": 7,
        "query": "Какая была инфляция в марте?",
        "expected_tools": ["get_inflation"],
        "must_have": [],
        "comment": "Трудный: год не указан — агент может угадать неверно.",
    },
    {
        "id": 8,
        "query": "Сколько юаней за один доллар в апреле 2026? (кросс-курс через рубль)",
        "expected_tools": ["get_fx_rate", "calculate"],
        "must_have": [],
        "comment": "Трудный: два курса + деление, легко перепутать порядок.",
    },
    {
        "id": 9,
        "query": "Имеет ли смысл держать рублёвый вклад сейчас, если сравнить ключевую ставку и инфляцию?",
        "expected_tools": ["get_key_rate", "get_inflation"],
        "must_have": [],
        "comment": "Реальный вопрос про реальную доходность вклада.",
    },
    {
        "id": 10,
        "query": "Как изменился курс юаня с января 2023 по декабрь 2025?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Реальный вопрос про динамику CNY; compare_periods fx_CNY.",
    },
]


def run_case(case: dict, *, use_cache: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(case["query"], max_iter=8, verbose=True, use_cache=use_cache)
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match and "error" not in res

    print(f"\n  tools used : {used_tools}")
    print(f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}")
    print(f"  answer      : {answer[:220]}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
        "run_id": res.get("run_id"),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [run_case(c, use_cache=a.cache) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено")
    print(f"{'id':>3} | {'ok':^4} | {'steps':^5} | query")
    print("-" * 70)
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"{r['id']:>3} | {mark:^4} | {r['steps']:^5} | {r['query'][:50]}")

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")
    print("Трассы: trace.jsonl")


if __name__ == "__main__":
    main()
