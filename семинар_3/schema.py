from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

CURRENT_DATE = date.today()

ALL_ASPECTS: list[str] = [
    "производительность",
    "дизайн",
    "поддержка",
    "цена",
    "реклама",
    "надежность",
]


class Issue(BaseModel):
    category: Literal["bug", "feature_request", "billing", "ux", "content", "ads"]
    severity: int = Field(ge=1, le=5)
    quote: str


class Review(BaseModel):
    review_id: str
    author_name: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    platform: Literal["ios", "android", "rustore"]
    issues: list[Issue] = Field(min_length=1)
    review_date: Optional[date] = None

    @field_validator("review_id")
    @classmethod
    def review_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("review_id не может быть пустым")
        return v.strip()

    @field_validator("review_date")
    @classmethod
    def review_date_not_in_future(cls, v: date | None) -> date | None:
        if v is not None and v > CURRENT_DATE:
            raise ValueError(f"дата отзыва {v} не может быть позже {CURRENT_DATE}")
        return v

    @model_validator(mode="after")
    def low_rating_needs_serious_issue(self) -> Review:
        if self.rating <= 2:
            if not any(i.severity >= 3 for i in self.issues):
                raise ValueError(
                    f"при рейтинге {self.rating} нужна хотя бы одна issue с severity ≥ 3"
                )
        return self


class AspectSentiment(BaseModel):
    aspect: Literal[
        "производительность",
        "дизайн",
        "поддержка",
        "цена",
        "реклама",
        "надежность",
    ]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    review_id: str
    aspects: list[AspectSentiment]


class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicReview(BaseModel):
    review_id: str
    aspects: list[DynamicAspect]


class ChunkSummary(BaseModel):
    batch_id: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
