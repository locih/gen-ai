"""
Замер «угодливости» Критика: ложные принятия на битых ответах.

Запуск: python critic_obedience.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer

QUESTION_FX = "Во сколько раз EUR дороже USD сегодня?"
QUESTION_REAL = "Какая сейчас реальная ключевая ставка?"
QUESTION_CPI = "Какова накопленная инфляция с января 2022 по март 2026?"

FAKE_BROKEN = [
    {
        "name": "арифметика без calculate",
        "question": QUESTION_FX,
        "plan": Plan(
            reasoning="Сравнить курсы EUR и USD.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Курс USD сегодня?",
                    expected_tools=["get_fx_rate"],
                ),
                SubQuestion(
                    id=2,
                    question="Курс EUR сегодня?",
                    expected_tools=["get_fx_rate"],
                ),
                SubQuestion(
                    id=3,
                    question="Во сколько раз EUR дороже USD?",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD",
                answer="USD = 82.5 руб.",
                used_tools=["get_fx_rate"],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс EUR",
                answer="EUR = 89.0 руб.",
                used_tools=["get_fx_rate"],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Во сколько раз",
                answer="EUR дороже USD в 1.08 раза (89/82.5).",
                used_tools=[],  # нет calculate — брак
            ),
        },
    },
    {
        "name": "выдуманное число",
        "question": QUESTION_REAL,
        "plan": Plan(
            reasoning="Номинал минус инфляция.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Ключевая ставка?",
                    expected_tools=["get_key_rate"],
                ),
                SubQuestion(
                    id=2,
                    question="Инфляция за март 2026?",
                    expected_tools=["get_inflation"],
                ),
                SubQuestion(
                    id=3,
                    question="Реальная ставка?",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Ключевая",
                answer="16.0%",
                used_tools=["get_key_rate"],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Инфляция",
                answer="6.8%",
                used_tools=["get_inflation"],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Реальная",
                answer="Реальная ставка 12.3%.",  # неверно, без calculate
                used_tools=[],
            ),
        },
    },
    {
        "name": "несогласованные данные",
        "question": QUESTION_FX,
        "plan": Plan(
            reasoning="Два курса и отношение.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="USD?",
                    expected_tools=["get_fx_rate"],
                ),
                SubQuestion(
                    id=2,
                    question="EUR?",
                    expected_tools=["get_fx_rate"],
                ),
                SubQuestion(
                    id=3,
                    question="Отношение EUR/USD",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="USD",
                answer="USD = 72.0 руб.",
                used_tools=["get_fx_rate"],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="EUR",
                answer="EUR = 80.0 руб.",
                used_tools=["get_fx_rate"],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Отношение",
                answer="EUR/USD = 1.50 (по calculate).",  # не согласуется с 80/72
                used_tools=["calculate"],
            ),
        },
    },
    {
        "name": "план не покрывает вопрос",
        "question": QUESTION_CPI,
        "plan": Plan(
            reasoning="Только март 2026.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="ИПЦ март 2026?",
                    expected_tools=["get_inflation"],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ИПЦ март",
                answer="6.8% г/г.",
                used_tools=["get_inflation"],
            ),
        },
    },
    {
        "name": "ответ с ошибкой исполнителя",
        "question": QUESTION_REAL,
        "plan": Plan(
            reasoning="Ставка и инфляция.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Ключевая ставка?",
                    expected_tools=["get_key_rate"],
                ),
                SubQuestion(
                    id=2,
                    question="Инфляция?",
                    expected_tools=["get_inflation"],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Ставка",
                answer="(ошибка: timeout)",
                used_tools=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Инфляция",
                answer="6.8%",
                used_tools=["get_inflation"],
            ),
        },
    },
]


def run_obedience(*, n: int = 10) -> list[dict]:
    rows = []
    for case in FAKE_BROKEN:
        false_0 = 0
        false_07 = 0
        for _ in range(n):
            v0 = critic(
                case["question"],
                case["plan"],
                case["answers"],
                temperature=0.0,
            )
            if v0.ok:
                false_0 += 1

            v07 = critic(
                case["question"],
                case["plan"],
                case["answers"],
                temperature=0.7,
            )
            if v07.ok:
                false_07 += 1

        rows.append(
            {
                "case": case["name"],
                "false_accepts_t0": false_0,
                "false_accepts_t07": false_07,
                "n": n,
            }
        )
        print(
            f"{case['name']}: T=0.0 → {false_0}/{n} ложных принятий, "
            f"T=0.7 → {false_07}/{n}"
        )
    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    print(f"Замер угодливости Критика, N={args.n} на кейс\n")
    rows = run_obedience(n=args.n)
    out = Path(__file__).parent / "critic_obedience_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
