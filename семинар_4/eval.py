"""
Сравнение двух стратегий чанкинга по метрике hit-rate@5.

Метрика: доля gold_sources, попавших в топ-5 чанков.
Для multi-hop: 2 из 2 → 1.0, 1 из 2 → 0.5.

Команды:
    python eval.py              # обе стратегии
    python eval.py --strategy a
    python eval.py --strategy b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import hybrid_retrieve, ingest

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run_strategy(strategy: str, k: int = 5, verbose: bool = True) -> dict:
    ingest(strategy)
    gold = load_gold()
    label = "A fixed-size 2000" if strategy == "a" else "B recursive 400/80"
    print(f"\n{'=' * 55}")
    print(f"  Стратегия {label}")
    print(f"{'=' * 55}")

    total = 0.0
    results = []
    for item in gold:
        hits = hybrid_retrieve(item["question"], strategy=strategy, k=k)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]
        score = hit_rate(retrieved_ids, item["gold_sources"])
        total += score
        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "score": score,
                "gold": item["gold_sources"],
                "retrieved": retrieved_sources[:k],
            }
        )
        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if 0 < score < 1.0 else "✗")
            print(
                f"  [{item['id']:2d}] {item['type']:25s} "
                f"hit@{k}={score:.2f} {mark} {item['question'][:55]}"
            )

    mean = total / len(gold)
    if verbose:
        print(f"\n  hit-rate@{k} = {mean:.2f} ({total:.1f}/{len(gold)})")
    return {"strategy": strategy, "mean": mean, "k": k, "results": results}


def compare(k: int = 5) -> None:
    res_a = run_strategy("a", k=k)
    res_b = run_strategy("b", k=k)

    print(f"\n{'=' * 55}")
    print(f"  Итог сравнения (hit-rate@{k})")
    print(f"{'=' * 55}")
    print(f"  Стратегия A fixed-size 2000   : {res_a['mean']:.2f}")
    print(f"  Стратегия B recursive 400/80  : {res_b['mean']:.2f}")
    delta = res_b["mean"] - res_a["mean"]
    winner = "B" if delta > 0 else ("A" if delta < 0 else "ничья")
    print(f"  Δ (B - A)                     : {delta:+.2f} → победитель: {winner}")

    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(
        json.dumps({"strategy_a": res_a, "strategy_b": res_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Подробности сохранены: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["a", "b"], default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.strategy:
        run_strategy(args.strategy, k=args.k)
    else:
        compare(k=args.k)


if __name__ == "__main__":
    main()
