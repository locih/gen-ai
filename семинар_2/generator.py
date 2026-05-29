"""
Генератор 50 синтетических заявок на курсы ДПО.

Для запуска: рядом должна быть папка starter/ с llm_client.py (из репозитория курса).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from schema import Application, CITIES, CURRENT_YEAR, DESIRED_COURSES, SPECIALITIES

ROOT = Path(__file__).parent
STARTER = ROOT / "starter"

_FEW_SHOT_EXAMPLES = [
    {
        "full_name": "Соколова Анна Викторовна",
        "age": 38,
        "address": {"city": "Москва", "district": "Северный административный округ"},
        "speciality": "учитель",
        "desired_course": "цифровая грамотность для педагогов",
        "years_of_experience": 14,
        "graduation_year": 2009,
    },
    {
        "full_name": "Петров Игорь Сергеевич",
        "age": 45,
        "address": {"city": "Новосибирск", "district": "Калининский район"},
        "speciality": "инженер",
        "desired_course": "управление проектами",
        "years_of_experience": 20,
        "graduation_year": 2002,
    },
]

_FEW_SHOT = "\n\n".join(
    f"Пример {i + 1}:\n{json.dumps(ex, ensure_ascii=False, indent=2)}"
    for i, ex in enumerate(_FEW_SHOT_EXAMPLES)
)

SYSTEM_PROMPT = f"""Ты генерируешь синтетические заявки на курсы повышения квалификации (ДПО) в России.
Создай правдоподобную заявку: **новое уникальное** русское ФИО (Фамилия Имя Отчество),
возраст 22–65, адрес, специальность, желаемый курс, стаж (0–40), год окончания вуза (1980–{CURRENT_YEAR}).

Допустимые города: {", ".join(CITIES)}.
Допустимые специальности: {", ".join(SPECIALITIES)}.
Допустимые курсы: {", ".join(DESIRED_COURSES)}.

Год окончания: не раньше (текущий_год − возраст + 22), не позже {CURRENT_YEAR}.
Стаж ≤ (текущий_год − graduation_year) + 2.

ФИО уникально в серии — не повторяй имена из списка «уже занято».

Ответ — один JSON-объект, без markdown.

Примеры формата:

{_FEW_SHOT}
"""

N_APPLICATIONS = 50
PER_CITY = 5
MAX_RETRIES = 3
NAME_ATTEMPTS = 6

client = None
MODEL = None


def _setup_client():
    global client, MODEL
    if client is not None:
        return
    if not (STARTER / "llm_client.py").is_file():
        sys.exit("Нужна папка starter/ из репозитория курса (llm_client.py).")
    sys.path.insert(1, str(STARTER))
    try:
        from dotenv import load_dotenv

        load_dotenv(STARTER / ".env")
    except ImportError:
        pass
    from llm_client import get_model, make_client  # noqa: E402

    client = make_client()
    MODEL = get_model()


def build_user_prompt(
    seed_city: str, seed_speciality: str, excluded_names: list[str]
) -> str:
    lines = [
        "Создай одну заявку.",
        f"address.city — ровно «{seed_city}».",
        f"speciality — ровно «{seed_speciality}».",
        "Придумай новое правдоподобное full_name, которого ещё нет в списке ниже.",
        "Подбери desired_course, возраст, стаж, год окончания вуза и район города.",
    ]
    if excluded_names:
        lines.append("")
        lines.append("Уже занятые ФИО (запрещено повторять целиком или ту же пару фамилия+имя):")
        for name in excluded_names:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def stratified_schedule() -> list[tuple[str, str]]:
    cities = [city for city in CITIES for _ in range(PER_CITY)]
    per_spec = N_APPLICATIONS // len(SPECIALITIES)
    specs = [spec for spec in SPECIALITIES for _ in range(per_spec)]
    slots = list(zip(cities, specs, strict=True))
    random.shuffle(slots)
    return slots


def _normalize_name(name: str) -> str:
    return " ".join(name.split())


def _name_key(name: str) -> str:
    parts = _normalize_name(name).split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".lower()
    return _normalize_name(name).lower()


def _is_duplicate(name: str, used_full: set[str], used_keys: set[str]) -> bool:
    norm = _normalize_name(name)
    return norm in used_full or _name_key(name) in used_keys


def generate_one(seed_city: str, seed_speciality: str, used_names: list[str]) -> Application:
    excluded = list(used_names)
    used_full = {_normalize_name(n) for n in used_names}
    used_keys = {_name_key(n) for n in used_names}
    last = None

    for attempt in range(NAME_ATTEMPTS):
        app = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(seed_city, seed_speciality, excluded),
                },
            ],
            response_model=Application,
            max_retries=MAX_RETRIES,
            temperature=0.95,
        )
        last = app
        if not _is_duplicate(app.full_name, used_full, used_keys):
            return app
        excluded.append(_normalize_name(app.full_name))
        print(f"    ↻ повтор ФИО ({attempt + 1}): {app.full_name!r}")

    raise ValueError(f"нет уникального ФИО, последнее: {last.full_name!r}")


def save_outputs(applications: list[Application], out_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "full_name": a.full_name,
                "age": a.age,
                "city": a.address.city,
                "district": a.address.district,
                "speciality": a.speciality,
                "desired_course": a.desired_course,
                "years_of_experience": a.years_of_experience,
                "graduation_year": a.graduation_year,
            }
            for a in applications
        ]
    )
    df.to_csv(out_dir / "applications.csv", index=False, encoding="utf-8")

    for col, title, file, color in (
        ("city", "Распределение по городам", "cities.png", "#7AB66E"),
        ("speciality", "Распределение по специальностям", "specialities.png", "#D97A4A"),
    ):
        counts = df[col].value_counts()
        plt.figure(figsize=(10, 4))
        counts.plot.bar(color=color, edgecolor="white")
        plt.title(f"{title} ({len(df)})")
        plt.ylabel("Число заявок")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / file, dpi=120)
        plt.close()

    print("\nСохранено: applications.csv, cities.png, specialities.png")
    print(f"Уникальных ФИО: {df['full_name'].nunique()}/{len(df)}")


def main():
    _setup_client()
    random.seed()
    applications: list[Application] = []
    used_names: list[str] = []
    t0 = time.time()

    print(f"Модель: {MODEL}\n")
    for i, (city, spec) in enumerate(stratified_schedule(), 1):
        print(f"  [{i:02d}/{N_APPLICATIONS}] {city}, {spec}…", end=" ", flush=True)
        for attempt in range(5):
            try:
                app = generate_one(city, spec, used_names)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"✗ {e}")
                    sys.exit(1)
                print(f"\n    ↻ слот ({attempt + 1}/5): {type(e).__name__}", flush=True)
                print(f"  [{i:02d}/{N_APPLICATIONS}] {city}, {spec}…", end=" ", flush=True)
        applications.append(app)
        used_names.append(_normalize_name(app.full_name))
        print(f"✓ {app.full_name}")
        time.sleep(0.15)

    if len({a.full_name for a in applications}) < len(applications):
        sys.exit("⚠ Повторы ФИО")
    print(f"\nГотово за {time.time() - t0:.1f} с")
    save_outputs(applications, ROOT)


if __name__ == "__main__":
    main()
