# L1 Helpdesk Copilot

```bash
cd финальный_проект && source .venv/bin/activate && python pipeline.py "текст тикета"
```

Перед первым запуском: `pip install -r requirements.txt`, `.env` с `CURSOR_API_KEY`, `python download_data.py`, `python rag.py ingest`.

тестировалось на апишке курсора, но я написал переключение на gemini, должно вроде работать

## Что где лежит

```
финальный_проект/
├── pipeline.py          # основной пайплайн: classify → RAG → план действий
├── eval.py              # оценка на test.csv → output/eval_results.json
├── schema.py            # Pydantic-схемы
├── rag.py               # hybrid RAG (Chroma + BM25)
├── critic.py            # критик + rework
├── judge.py             # LLM-as-judge
├── llm_client.py        # Cursor API
├── download_data.py     # скачать датасет и runbook'и
├── input/
│   ├── tickets/         # train.csv, test.csv
│   ├── kb/              # runbook'и
│   └── label_map.json   # 25 классов → 9 meta-категорий
└── output/
    ├── chroma_db/       # индекс RAG
    ├── eval_results.json
    └── trace.jsonl
```
