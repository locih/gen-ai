"""
RAG по runbook'ам и примерам тикетов: ChromaDB + BM25 + RRF.

Команды:
    python rag.py ingest
    python rag.py search "не работает СДО"
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

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "input" / "kb"
CHROMA_PATH = ROOT / "output" / "chroma_db"
BM25_CACHE = ROOT / "output" / "bm25_cache.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _get_collection():
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
    )
    return chroma.get_or_create_collection(
        name="helpdesk_kb",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest() -> None:
    if not KB_DIR.is_dir():
        raise SystemExit(f"Нет {KB_DIR}. Сначала: python download_data.py")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    collection = _get_collection()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for f in sorted(KB_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i})

        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")

    print(f"\nИндекс: {len(all_ids)} чанков → {CHROMA_PATH}")


def _load_bm25() -> tuple[BM25Okapi, list[str], list[str]]:
    if not BM25_CACHE.is_file():
        raise RuntimeError("BM25 кэш не найден. Запустите: python rag.py ingest")
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60) -> dict:
    """Dense + BM25 + RRF."""
    collection = _get_collection()
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25()
    scores = bm25.get_scores(tokenize_ru(query))
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[
        :top
    ]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {
        "ids": top_ids,
        "documents": [text_by_id[i] for i in top_ids],
        "scores": [rrf[i] for i in top_ids],
    }


def format_context(hits: dict) -> str:
    parts = []
    for cid, doc in zip(hits["ids"], hits["documents"]):
        parts.append(f"[{cid}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def check_quotes_in_corpus(quotes: list[str], corpus: str) -> list[str]:
    """Ghost-цитаты: фрагмент не найден в retrieved-корпусе."""
    corpus_l = corpus.lower()
    ghosts: list[str] = []
    for q in quotes:
        probe = q.strip().lower()[:40]
        if probe and probe not in corpus_l:
            ghosts.append(q)
    return ghosts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python rag.py {ingest|search} [запрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        t0 = time.time()
        ingest()
        print(f"За {time.time() - t0:.1f} с")
    elif cmd == "search":
        q = " ".join(sys.argv[2:]) or "не работает СДО"
        hits = hybrid_retrieve(q, k=5)
        print(format_context(hits))
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
