"""Проверка окружения: Cursor API или OpenAI-compatible."""

from __future__ import annotations

import sys

from llm_client import get_model, make_client, use_cursor


def main() -> None:
    print(f"provider: {'cursor' if use_cursor() else 'openai-compatible'}")
    print(f"model:    {get_model()}")

    if use_cursor():
        import os

        if not os.environ.get("CURSOR_API_KEY"):
            print("ERROR: CURSOR_API_KEY не задан в .env")
            sys.exit(1)
        print("CURSOR_API_KEY: ok (задан)")
    else:
        from llm_client import _make_openai_client

        _make_openai_client()
        print("OpenAI client: ok")

    print("\nПробный structured-вызов (TicketClassification)...")
    from schema import TicketClassification

    client = make_client()
    r = client.chat.completions.create(
        model=get_model(),
        response_model=TicketClassification,
        max_retries=1,
        messages=[
            {
                "role": "user",
                "content": "Тикет: Не работает принтер в кабинете 322, не печатает документы.",
            }
        ],
    )
    print("OK:", r.meta_category, r.priority, r.summary[:60])


if __name__ == "__main__":
    main()
