"""
Pydantic-схема заявки на курс ДПО (домашнее задание, семинар 2).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CURRENT_YEAR = date.today().year

CITIES = (
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Воронеж",
)

SPECIALITIES = (
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "юрист",
    "менеджер",
    "IT-специалист",
    "экономист",
    "психолог",
    "медсестра",
)

DESIRED_COURSES = (
    "цифровая грамотность для педагогов",
    "управление проектами",
    "налоговый учёт и отчётность",
    "психология делового общения",
    "охрана труда",
    "маркетинг и продажи",
    "медицинская реабилитация",
    "программирование на Python",
)


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=80)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[*SPECIALITIES]
    desired_course: Literal[*DESIRED_COURSES]
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=CURRENT_YEAR)

    @property
    def city(self) -> str:
        return self.address.city

    @model_validator(mode="after")
    def graduation_consistent_with_age(self) -> Application:
        """Минимум 22 года на момент окончания вуза; не позже текущего года."""
        birth_year = CURRENT_YEAR - self.age
        min_graduation = birth_year + 22
        if self.graduation_year < min_graduation:
            raise ValueError(
                f"год окончания {self.graduation_year} слишком ранний для возраста "
                f"{self.age} (ожидается ≥ {min_graduation})"
            )
        if self.graduation_year > CURRENT_YEAR:
            raise ValueError(
                f"год окончания {self.graduation_year} не может быть позже {CURRENT_YEAR}"
            )
        max_experience = CURRENT_YEAR - self.graduation_year
        if self.years_of_experience > max_experience + 2:
            raise ValueError(
                f"стаж {self.years_of_experience} лет несовместим с окончанием вуза "
                f"в {self.graduation_year}"
            )
        return self
