"""
RAG-пайплайн: ChromaDB + sentence-transformers + BM25 гибрид.
Поддерживает две стратегии чанкинга:
  --strategy a   fixed-size 2000 символов без перекрытия
  --strategy b   RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

Команды:
    python pipeline.py ingest --strategy a
    python pipeline.py ingest --strategy b
    python pipeline.py ask "вопрос" --strategy a
    python pipeline.py ask "вопрос" --strategy b
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent / "data"
CHROMA_A = Path(__file__).parent / "chroma_a"
CHROMA_B = Path(__file__).parent / "chroma_b"
BM25_A = Path(__file__).parent / "bm25_a.json"
BM25_B = Path(__file__).parent / "bm25_b.json"

print("Загружаю эмбеддер...", flush=True)
_t = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t:.1f}с", flush=True)

_splitter_b = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)


def _get_chroma(strategy: str) -> chromadb.Collection:
    path = str(CHROMA_A if strategy == "a" else CHROMA_B)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=f"corpus_{strategy}",
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def _bm25_path(strategy: str) -> Path:
    return BM25_A if strategy == "a" else BM25_B


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_fixed(text: str, size: int = 2000) -> list[str]:
    """Стратегия A: фиксированный размер, без перекрытия."""
    return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size].strip()]


def chunk_recursive(text: str) -> list[str]:
    """Стратегия B: рекурсивный сплиттер по абзацам/предложениям."""
    return [c.strip() for c in _splitter_b.split_text(text) if c.strip()]


def ingest(strategy: str) -> None:
    collection = _get_chroma(strategy)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunk_fn = chunk_fixed if strategy == "a" else chunk_recursive
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_fn(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_ids.append(f"{f.stem}__{i}")
            all_meta.append({"source": f.stem, "chunk_id": i})
        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    _bm25_path(strategy).write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")

    label = "A (fixed-size 2000)" if strategy == "a" else "B (recursive 400/80)"
    print(f"\nСтратегия {label}: проиндексировано {collection.count()} чанков")


def _load_bm25(strategy: str):
    data = json.loads(_bm25_path(strategy).read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, strategy: str, k: int = 5, top: int = 15, c: int = 60) -> dict:
    collection = _get_chroma(strategy)
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25(strategy)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


def ask(query: str, strategy: str) -> None:
    """Поиск и вывод топ-5 чанков (retrieval без LLM)."""
    print(f"\nСтратегия: {'A (fixed)' if strategy == 'a' else 'B (recursive)'}")
    print(f"Запрос: {query}\n")
    hits = hybrid_retrieve(query, strategy=strategy, k=5)
    for i, (cid, doc) in enumerate(zip(hits["ids"][0], hits["documents"][0]), 1):
        preview = doc[:300].replace("\n", " ")
        print(f"[{i}] {cid}")
        print(preview + ("..." if len(doc) > 300 else ""))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python pipeline.py ingest --strategy {a|b}")
        print("  python pipeline.py ask 'вопрос' --strategy {a|b}")
        sys.exit(1)

    cmd = sys.argv[1]
    strategy = "a"
    if "--strategy" in sys.argv:
        idx = sys.argv.index("--strategy")
        strategy = sys.argv[idx + 1].lower()
    if strategy not in ("a", "b"):
        print("Стратегия должна быть 'a' или 'b'")
        sys.exit(1)

    if cmd == "ingest":
        ingest(strategy)
    elif cmd == "ask":
        query = sys.argv[2]
        ask(query, strategy)
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
