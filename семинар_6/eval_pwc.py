"""
Eval мульти-агента: 6 вопросов × 3 конфигурации.

Конфигурации:
  1) одиночный агент (agent_s5)
  2) PWC без валидатора
  3) PWC + валидатор схемы

Запуск:
    python eval_pwc.py
    python eval_pwc.py --single
    python eval_pwc.py -n 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import benchmark_parallel, run_pwc

CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс C: нужен calculate. Естественная параллельность: два get_fx_rate."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD"],
        "forbid_hallucinated_tools": True,
        "parallel_benchmark": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": "Класс B: последний доступный месяц ИПЦ.",
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс D: Планировщик часто выдумывает get_cumulative_inflation; "
            "валидатор должен перепланировать."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q4",
        "query": (
            "Во сколько раз вырос курс USD с января 2022 по апрель 2026? "
            "Обязательно вызови compare_periods(metric='fx_USD', "
            "period_a='2022-01', period_b='2026-04')."
        ),
        "comment": (
            "ДЗ (валидатор): пользователь просит compare_periods — Планировщик "
            "кладёт его в план (правило 5). Без валидатора план бракованный; "
            "с валидатором — get_fx_rate×2 + calculate. Одиночный агент зовёт "
            "compare_periods → unknown tool."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD"],
        "forbid_hallucinated_tools": True,
        "validator_fix": True,
        "require_plan_hallucination_without_validator": True,
    },
    {
        "id": "Q5",
        "query": (
            "Какие сегодня официальные курсы USD, EUR и CNY к рублю? "
            "Перечисли все три числа."
        ),
        "comment": (
            "ДЗ: 3+ независимых подвопроса (параллельность). "
            "Замер ускорения на Q1 и Q5."
        ),
        "expected_tools_pwc": {"get_fx_rate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "forbid_hallucinated_tools": True,
        "parallel_benchmark": True,
    },
    {
        "id": "Q6",
        "query": (
            "Имеет ли смысл сейчас брать ипотеку, если сравнить ключевую ставку "
            "и инфляцию за последний доступный месяц?"
        ),
        "comment": "Реальный макро-вопрос про ставку vs инфляцию.",
        "expected_tools_pwc": {"get_key_rate", "get_inflation"},
        "must_have_keywords": [],
        "forbid_hallucinated_tools": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = (
        case["id"] in {"Q1", "Q2", "Q3", "Q4"}
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    if case.get("validator_fix"):
        # Ловушка: одиночный не может корректно вызвать compare_periods
        ok = False
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _first_plan_tools(result: dict) -> set[str]:
    for t in result.get("trace", []):
        if t.get("kind") == "plan":
            tools: set[str] = set()
            for sq in t.get("subquestions", []):
                tools.update(sq.get("expected_tools", []))
            return tools
    return set()


def _check_pwc(case: dict, result: dict, *, with_validator: bool = False) -> dict:
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])

    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES

    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
    )
    if case.get("validator_fix"):
        if not with_validator and "compare_periods" in _first_plan_tools(result):
            ok = False
        if with_validator and plan_hallucinated:
            ok = False
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc = {"runs": [], "pass": 0}
    pwc_val = {"runs": [], "pass": 0}

    for _ in range(n):
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        c1 = _check_single(case, r1)
        single["runs"].append(c1)
        single["pass"] += int(c1["ok"])

        try:
            r2 = run_pwc(
                case["query"], max_iter=4, verbose=False, use_validator=False
            )
        except Exception as e:
            r2 = {
                "answer": None,
                "error": f"{type(e).__name__}: {e}",
                "trace": [],
                "plan": None,
            }
        c2 = _check_pwc(case, r2, with_validator=False)
        pwc["runs"].append(c2)
        pwc["pass"] += int(c2["ok"])

        try:
            r3 = run_pwc(
                case["query"], max_iter=4, verbose=False, use_validator=True
            )
        except Exception as e:
            r3 = {
                "answer": None,
                "error": f"{type(e).__name__}: {e}",
                "trace": [],
                "plan": None,
            }
        c3 = _check_pwc(case, r3, with_validator=True)
        pwc_val["runs"].append(c3)
        pwc_val["pass"] += int(c3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--benchmark", action="store_true", help="Замер параллельности")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    if args.benchmark:
        for case in CASES:
            if not case.get("parallel_benchmark"):
                continue
            print(f"Benchmark {case['id']}: {case['query'][:50]}...")
            b = benchmark_parallel(case["query"], runs=1)
            print(
                f"  уровни плана: {b.get('plan_levels')} "
                f"(max параллельных={b.get('max_parallel_subqs')})"
            )
            print(f"  seq={b['sequential_sec']}s  par={b['parallel_sec']}s  "
                  f"speedup={b['speedup']}x\n")
        return

    print(f"Eval С6: {len(CASES)} кейсов × {n} прогонов × 3 конфигурации\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        print(
            f"   single: {r['single']['pass']}/{n}   "
            f"pwc: {r['pwc']['pass']}/{n}   "
            f"pwc+val: {r['pwc_validator']['pass']}/{n}"
        )
        for run in r["pwc"]["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(
                    f"   ⚠ pwc план: выдуманные tools {run['hallucinated_in_plan']}"
                )
        print()

    print("=" * 60)
    print("ИТОГО:")
    for r in results:
        print(
            f"  {r['id']}: single {r['single']['pass']}/{n}  "
            f"pwc {r['pwc']['pass']}/{n}  "
            f"pwc+val {r['pwc_validator']['pass']}/{n}"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
