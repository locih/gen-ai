"""
Планировщик: разбирает исходный вопрос на подвопросы.
"""

from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan

SYSTEM_PROMPT = """\
Ты — планировщик макроэкономического агента. Твоя задача — разложить
сложный вопрос пользователя на 1-5 простых подвопросов, каждый из
которых решается одним конкретным инструментом.

Доступные инструменты (НЕ придумывай других):
- get_fx_rate(currency, on_date): курс валюты к рублю на дату.
  on_date можно не указывать (null) — тогда вернётся курс на сегодня.
- get_key_rate(on_date): ключевая ставка ЦБ на дату.
  on_date можно не указывать (null) — тогда вернётся текущая ставка.
- get_inflation(year, month): ИПЦ г/г на конец месяца.
- calculate(expression): безопасный калькулятор.

ПРАВИЛА:
1. Любая арифметика (разность, отношение, произведение, накопленная
   инфляция) — отдельный подвопрос с expected_tools=["calculate"].
   Не считай в уме и не выдумывай get_cumulative_inflation.
2. Если подвопрос N зависит от ответа подвопроса K — поставь K в depends_on.
3. Для вопросов про «последний доступный период» — первым шагом поставь
   подвопрос «узнать доступный период».
4. Не придумывай инструменты сами. Исключение: если пользователь явно
   просит вызвать инструмент по имени (compare_periods, get_cumulative_inflation,
   get_real_yield и т.п.) — включи это имя в expected_tools шага.
5. Если задача не решается имеющимися tools даже после п.4 — верни reasoning
   и subquestions=[].

Цель — минимальный корректный план.
"""


def planner(question: str, *, feedback: str | None = None) -> Plan:
    """Вернуть План для исходного вопроса."""
    client = make_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if feedback:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Предыдущая попытка не прошла проверку. Замечание: {feedback}"
                ),
            }
        )

    _TRAP_TOOLS = (
        "compare_periods",
        "get_cumulative_inflation",
        "get_real_yield",
        "get_real_rate",
    )
    if feedback is None:
        for name in _TRAP_TOOLS:
            if name in question:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Пользователь требует шаг с {name}. "
                            f"Подвопрос с этим вызовом должен иметь "
                            f"expected_tools: ['{name}']."
                        ),
                    }
                )
                break

    return client.chat.completions.create(
        model=get_model(),
        messages=messages,
        response_model=Plan,
        temperature=0.0,
        max_retries=2,
    )


if __name__ == "__main__":
    import sys

    q = (
        " ".join(sys.argv[1:])
        or "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
    )
    plan = planner(q)
    print(f"План (reasoning): {plan.reasoning}\n")
    for sq in plan.subquestions:
        deps = f" ← ждёт {sq.depends_on}" if sq.depends_on else ""
        print(f"  {sq.id}. [{','.join(sq.expected_tools)}]{deps}")
        print(f"     {sq.question}")
