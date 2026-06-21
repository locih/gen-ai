"""
Eval на test.csv из SIA86.

Запуск:
    python eval.py --limit 20          # быстрый прогон (с judge — оценка ответа)
    python eval.py                     # все 231 кейс
    python eval.py --no-judge --limit 15   # только category + critic, без LLM-judge
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pipeline import gold_meta_from_label, load_label_map, run_pipeline

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "input" / "tickets" / "test.csv"
OUT_PATH = ROOT / "output" / "eval_results.json"


def _rate(num: int, denom: int) -> float | None:
    return round(num / denom, 4) if denom else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = все")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if not TEST_PATH.is_file():
        raise SystemExit("Нет test.csv. Запустите: python download_data.py")

    label_map = load_label_map()
    rows = list(csv.DictReader(TEST_PATH.open(encoding="utf-8")))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    results = []
    t0 = time.time()
    cat_ok = 0
    ghosts_total = 0
    judge_scores: list[float] = []
    judge_steps_ok = 0
    judge_pass = 0  # overall_score >= 0.5
    critic_ok_count = 0
    critic_ran = 0
    critic_reworks_total = 0
    grounded_ok = 0  # critic_ok + 0 ghost quotes

    for i, row in enumerate(rows, 1):
        lid = int(row["label"])
        gold = gold_meta_from_label(lid, label_map)
        text = row["text"]
        print(f"[{i}/{len(rows)}] label={lid} gold={gold} …", flush=True)

        try:
            r = run_pipeline(
                text,
                gold_label_id=lid,
                use_judge=not args.no_judge,
            )
            ok = r.classification.meta_category == gold
            cat_ok += int(ok)
            n_ghosts = len(r.ghost_quotes)
            ghosts_total += n_ghosts
            jscore = r.judge.overall_score if r.judge else None
            if jscore is not None:
                judge_scores.append(jscore)
                if r.judge and r.judge.steps_supported:
                    judge_steps_ok += 1
                if jscore >= 0.5:
                    judge_pass += 1

            if r.critic_ok is not None:
                critic_ran += 1
                critic_reworks_total += r.critic_reworks
                if r.critic_ok:
                    critic_ok_count += 1
                if r.critic_ok and n_ghosts == 0:
                    grounded_ok += 1

            row_out = {
                "id": i,
                "gold_label_id": lid,
                "gold_meta": gold,
                "predicted_meta": r.classification.meta_category,
                "priority": r.classification.priority,
                "category_ok": ok,
                "confidence": r.classification.confidence,
                "ghost_quotes": n_ghosts,
                "critic_ok": r.critic_ok,
                "critic_reworks": r.critic_reworks,
                "tools_used": r.tools_used,
                "steps": r.steps,
                "retrieved": r.retrieved_sources,
                "summary": r.classification.summary[:120],
                "action_steps": len(r.action.suggested_steps),
                "suggested_steps": [s.model_dump() for s in r.action.suggested_steps],
                "action_quotes": r.action.quotes,
                "action_sources": r.action.sources,
                "escalation_team": r.action.escalation_team,
            }
            if r.judge:
                row_out["judge_score"] = jscore
                row_out["judge_steps_supported"] = r.judge.steps_supported
                row_out["judge_category_correct"] = r.judge.category_correct
                row_out["judge_comment"] = r.judge.comment[:200]
            if r.critic_issues:
                row_out["critic_issues"] = r.critic_issues

            results.append(row_out)
        except Exception as e:
            results.append(
                {
                    "id": i,
                    "gold_label_id": lid,
                    "gold_meta": gold,
                    "error": str(e),
                    "category_ok": False,
                }
            )

    elapsed = time.time() - t0
    n = len(results)
    errors = sum(1 for r in results if "error" in r)
    valid = n - errors

    summary = {
        "n": n,
        "errors": errors,
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_ticket": round(elapsed / n, 1) if n else 0,
        "category": {
            "accuracy": _rate(cat_ok, valid),
            "ok": cat_ok,
            "total": valid,
        },
        "answer": {
            "judge_enabled": not args.no_judge,
            "score_mean": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
            "score_pass_rate": _rate(judge_pass, len(judge_scores)),
            "steps_supported_rate": _rate(judge_steps_ok, len(judge_scores)),
            "judge_n": len(judge_scores),
            "critic_ok_rate": _rate(critic_ok_count, critic_ran),
            "critic_ok": critic_ok_count,
            "critic_n": critic_ran,
            "critic_reworks_total": critic_reworks_total,
            "critic_reworks_mean": round(critic_reworks_total / critic_ran, 2) if critic_ran else None,
            "grounded_ok_rate": _rate(grounded_ok, critic_ran),
            "grounded_ok": grounded_ok,
            "ghost_quotes_total": ghosts_total,
        },
        # обратная совместимость со старым форматом eval_results.json
        "category_accuracy": _rate(cat_ok, valid),
        "category_ok": cat_ok,
        "ghost_quotes_total": ghosts_total,
        "judge_mean": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ans = summary["answer"]
    print("\n" + "=" * 50)
    print(f"Eval: {n} кейсов за {elapsed:.0f} с")
    print(f"Category accuracy: {cat_ok}/{valid} = {summary['category']['accuracy']:.1%}")
    print("— ответ (план действий) —")
    if ans["judge_enabled"] and judge_scores:
        print(f"  Judge score mean:     {ans['score_mean']:.2f}")
        print(f"  Judge pass (≥0.5):    {judge_pass}/{len(judge_scores)} = {ans['score_pass_rate']:.1%}")
        print(f"  Steps supported:      {judge_steps_ok}/{len(judge_scores)} = {ans['steps_supported_rate']:.1%}")
    elif ans["judge_enabled"]:
        print("  Judge: нет оценок (все кейсы с ошибкой?)")
    else:
        print("  Judge: выключен (--no-judge). Запустите без флага для оценки ответа.")
    if critic_ran:
        print(f"  Critic ok:            {critic_ok_count}/{critic_ran} = {ans['critic_ok_rate']:.1%}")
        print(f"  Grounded (critic+0 ghost): {grounded_ok}/{critic_ran} = {ans['grounded_ok_rate']:.1%}")
        print(f"  Critic reworks total: {critic_reworks_total} (mean {ans['critic_reworks_mean']:.2f})")
    print(f"  Ghost-цитат всего:    {ghosts_total}")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
