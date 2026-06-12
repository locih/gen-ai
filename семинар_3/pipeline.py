"""
pipeline.py — анализ отзывов из магазина приложений (ДЗ семинар 3, вариант A).

IE → аспекты → autodiscovery → Map-Reduce → LLM-as-judge

Запуск:
    python pipeline.py input/reviews.txt
    python pipeline.py input/reviews.txt output/
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_ROOT = Path(__file__).resolve().parent


def _load_llm_client():
    import importlib.util

    for candidate in (_ROOT / "starter", _ROOT.parent / "семинар_2" / "starter"):
        path = candidate / "llm_client.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("llm_client", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    sys.exit("Нужен llm_client.py (папка starter/ из репозитория курса).")


_llm = _load_llm_client()
get_model = _llm.get_model
make_client = _llm.make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    DISCOVER_SYSTEM,
    DYNAMIC_ASPECTS_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    ALL_ASPECTS,
    ChunkSummary,
    DiscoveredAspects,
    DynamicReview,
    JudgeReport,
    Review,
    ReviewSentiment,
    ReviewsSummary,
)

client = make_client()
MODEL = get_model()

BATCH_SIZE = 10
_PRICE_INPUT = 0.27 / 1_000_000
_PRICE_OUTPUT = 1.10 / 1_000_000


class _UsageTracker:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, resp) -> None:
        if resp and hasattr(resp, "usage") and resp.usage:
            self.input_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
            self.calls += 1

    def cost_usd(self) -> float:
        return self.input_tokens * _PRICE_INPUT + self.output_tokens * _PRICE_OUTPUT


_tracker = _UsageTracker()


def _create(**kwargs):
    result, resp = client.chat.completions.create(with_completion=True, **kwargs)
    _tracker.add(resp)
    return result


def _split_into_batches(text: str, batch_size: int = BATCH_SIZE) -> list[tuple[str, str]]:
    blocks = re.split(r"(?=^=== ОТЗЫВ #R)", text, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip().startswith("=== ОТЗЫВ")]
    batches = []
    for i in range(0, len(blocks), batch_size):
        chunk = "\n\n".join(blocks[i : i + batch_size])
        batches.append((f"Batch_{i // batch_size + 1}", chunk))
    return batches


def extract_reviews(text: str) -> tuple[list[Review], int]:
    """Вернуть (валидные отзывы, число ValidationError)."""
    raw_batches: list[Review] = []
    errors = 0
    for _bid, chunk in _split_into_batches(text):
        try:
            batch = _create(
                model=MODEL,
                response_model=list[Review],
                max_retries=3,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": IE_SYSTEM},
                    {"role": "user", "content": chunk},
                ],
            )
            raw_batches.extend(batch)
        except Exception as e:
            print(f"  ⚠ IE batch error: {e}")
            errors += 1
    return raw_batches, errors


def extract_aspects(text: str) -> list[ReviewSentiment]:
    results: list[ReviewSentiment] = []
    for _bid, chunk in _split_into_batches(text):
        batch = _create(
            model=MODEL,
            response_model=list[ReviewSentiment],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": ASPECTS_SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        results.extend(batch)
    return results


def check_quotes(
    aspects: list[ReviewSentiment],
    source_text: str,
) -> list[tuple[str, str]]:
    t = source_text.lower()
    ghosts: list[tuple[str, str]] = []
    for r in aspects:
        for a in r.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((r.review_id, a.quote))
    return ghosts


def build_heatmap(aspects: list[ReviewSentiment], out_path: str) -> None:
    ids = [r.review_id for r in aspects]
    sent_map = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(ids), len(ALL_ASPECTS)), np.nan)
    for i, r in enumerate(aspects):
        for a in r.aspects:
            if a.aspect in ALL_ASPECTS:
                j = ALL_ASPECTS.index(a.aspect)
                matrix[i, j] = sent_map[a.sentiment]

    fig_h = max(5, len(ids) * 0.28)
    plt.figure(figsize=(10, fig_h))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=ALL_ASPECTS,
        yticklabels=ids,
        center=0,
        cmap="RdYlGn",
        linewidths=0.3,
    )
    plt.title("Тональность по аспектам (отзывы)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def discover_aspects(text: str) -> DiscoveredAspects:
    return _create(
        model=MODEL,
        response_model=DiscoveredAspects,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": DISCOVER_SYSTEM},
            {"role": "user", "content": text},
        ],
    )


def extract_dynamic_aspects(text: str, discovered: DiscoveredAspects) -> list[DynamicReview]:
    aspects_list = "\n".join(f"- {a.name}: {a.description}" for a in discovered.aspects)
    system = DYNAMIC_ASPECTS_SYSTEM.format(aspects_list=aspects_list)
    results: list[DynamicReview] = []
    for _bid, chunk in _split_into_batches(text):
        batch = _create(
            model=MODEL,
            response_model=list[DynamicReview],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
        )
        results.extend(batch)
    return results


def _summarize_batch(batch_id: str, chunk: str) -> ChunkSummary:
    result = _create(
        model=MODEL,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": f"Пакет отзывов {batch_id}:\n\n{chunk}"},
        ],
    )
    result.batch_id = batch_id
    return result


def _reduce_summaries(
    summaries: list[ChunkSummary],
    reduce_prompt: str = REDUCE_SYSTEM,
) -> ReviewsSummary:
    joined = "\n\n".join(
        f"## {s.batch_id} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    return _create(
        model=MODEL,
        response_model=ReviewsSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": joined},
        ],
    )


def summarize_reviews(text: str, workers: int = 4) -> ReviewsSummary:
    batches = _split_into_batches(text)
    n = len(batches)
    print(f"  [MR] MAP: {n} пакетов...")
    t0 = time.time()
    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_summarize_batch, bid, chunk): i
            for i, (bid, chunk) in enumerate(batches)
        }
        for fut in as_completed(futures):
            summaries[futures[fut]] = fut.result()
    result = _reduce_summaries([s for s in summaries if s is not None])
    print(f"  [MR] {time.time() - t0:.1f}с")
    return result


def _build_evidence_packet(reviews: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Issues из отзывов")
    for r in reviews:
        for issue in r.get("issues", []):
            parts.append(
                f"  - [{r['review_id']}/{issue['category']}, sev={issue['severity']}, "
                f"★{r['rating']}] «{issue['quote']}»"
            )
    return "\n".join(parts)


def run_judge(reviews: list[dict], summary: dict) -> JudgeReport:
    return _create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _build_evidence_packet(reviews, summary)},
        ],
    )


def analyze(input_path: str, out_dir: str = "output") -> dict:
    global _tracker
    _tracker = _UsageTracker()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(input_path).read_text(encoding="utf-8")
    t0 = time.time()
    stats: dict = {}

    print("→ Акт 1: IE (Review + issues)...")
    reviews, ie_errors = extract_reviews(text)
    stats["reviews_valid"] = len(reviews)
    stats["ie_errors"] = ie_errors
    stats["issues_total"] = sum(len(r.issues) for r in reviews)
    print(f"  {stats['reviews_valid']} отзывов, {stats['issues_total']} issues, ошибок IE: {ie_errors}")

    reviews_data = [r.model_dump(mode="json") for r in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("→ Акт 2: аспекты...")
    aspects = extract_aspects(text)
    ghosts = check_quotes(aspects, text)
    stats["ghost_count"] = len(ghosts)
    stats["quotes_total"] = sum(len(r.aspects) for r in aspects)
    stats["ghost_pct"] = (
        stats["ghost_count"] / stats["quotes_total"] * 100 if stats["quotes_total"] else 0
    )
    print(f"  ghost: {stats['ghost_count']}/{stats['quotes_total']} ({stats['ghost_pct']:.1f}%)")
    if ghosts:
        for rid, q in ghosts[:3]:
            print(f"    {rid}: «{q[:70]}…»")

    build_heatmap(aspects, str(out / "heatmap.png"))
    (out / "aspects.json").write_text(
        json.dumps([a.model_dump() for a in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("→ Акт 2.5: autodiscovery...")
    discovered = discover_aspects(text)
    extract_dynamic_aspects(text, discovered)

    print("→ Акт 3: Map-Reduce...")
    summary = summarize_reviews(text)
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    print("→ Акт 5: judge...")
    summary_dict = json.loads(summary.model_dump_json())
    report = run_judge(reviews_data, summary_dict)

    if report.overall_score < 0.7:
        print(f"  ⚠ score={report.overall_score:.2f} — повтор REDUCE...")
        improved = REDUCE_SYSTEM + (
            "\n\nКаждый action_item — только из реальных жалоб в отзывах."
        )
        batches = _split_into_batches(text)
        batch_summaries = [_summarize_batch(bid, chunk) for bid, chunk in batches]
        summary = _reduce_summaries(batch_summaries, reduce_prompt=improved)
        (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        report = run_judge(reviews_data, json.loads(summary.model_dump_json()))

    (out / "judge_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    stats["overall_score"] = report.overall_score
    stats["elapsed_sec"] = time.time() - t0
    stats["input_tokens"] = _tracker.input_tokens
    stats["output_tokens"] = _tracker.output_tokens
    stats["api_calls"] = _tracker.calls
    stats["cost_usd"] = _tracker.cost_usd()

    print(f"\n══ {summary.headline}")
    print(f"Judge: {report.overall_score:.2f} | ${stats['cost_usd']:.4f} | {stats['elapsed_sec']:.0f}с")
    return stats


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py input/reviews.txt [output/]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
