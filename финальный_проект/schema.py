"""Pydantic-схемы пайплайна helpdesk."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MetaCategory = Literal[
    "access",
    "hardware",
    "software",
    "email",
    "account",
    "workplace",
    "incident",
    "request",
    "other",
]

Priority = Literal["low", "medium", "high", "critical"]

META_CATEGORIES: list[str] = [
    "access",
    "hardware",
    "software",
    "email",
    "account",
    "workplace",
    "incident",
    "request",
    "other",
]


class TicketClassification(BaseModel):
    """Этап 1 — классификация тикета."""

    meta_category: MetaCategory
    priority: Priority
    affected_systems: list[str] = Field(
        default_factory=list,
        description="Системы из текста: СДО, Oracle, почта и т.п.",
    )
    summary: str = Field(description="Краткое описание проблемы одним предложением")
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def summary_min_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("summary слишком короткий (минимум 10 символов)")
        return v

    @field_validator("affected_systems")
    @classmethod
    def systems_not_empty_strings(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s.strip()]


class ActionStep(BaseModel):
    step: int = Field(ge=1, le=10)
    action: str
    quote: str | None = Field(
        default=None,
        description="Короткая цитата из, если шаг опирается на runbook",
    )


class TicketActionResponse(BaseModel):
    """Этап 2 — что делать (grounded в KB)."""

    meta_category: MetaCategory
    escalation_team: str
    suggested_steps: list[ActionStep] = Field(min_length=1, max_length=6)
    quotes: list[str] = Field(default_factory=list, max_length=5)
    sources: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("suggested_steps")
    @classmethod
    def steps_ordered(cls, v: list[ActionStep]) -> list[ActionStep]:
        nums = [s.step for s in v]
        if nums != sorted(nums):
            raise ValueError("номера шагов должны идти по возрастанию")
        return v


class JudgeVerdict(BaseModel):
    category_correct: bool
    steps_supported: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    comment: str = ""


class CriticVerdict(BaseModel):
    """Критик для цикла rework (семинар 5/6)."""

    ok: bool
    issue: str = ""


class PipelineResult(BaseModel):
    ticket_text: str
    gold_label_id: int | None = None
    gold_meta: MetaCategory | None = None
    classification: TicketClassification
    action: TicketActionResponse
    retrieved_sources: list[str]
    ghost_quotes: list[str]
    steps: int = 2
    tools_used: list[str] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    critic_reworks: int = 0
    critic_issues: list[str] = Field(default_factory=list)
    critic_ok: bool | None = Field(
        default=None,
        description="Критик одобрил финальный план (None если критик выключен)",
    )
