"""Критик для цикла rework перед финальным ответом."""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import CriticVerdict, TicketActionResponse, TicketClassification

client = make_client()
MODEL = get_model()

CRITIC_SYSTEM = """Ты — придирчивый ревизор L1 helpdesk. Проверь план действий ДО выдачи пользователю.

ok=true только если:
- шаги конкретны и опираются на фрагменты KB (не общие фразы вроде «обратитесь в поддержку» без деталей);
- quotes (если есть) выглядят как фрагменты из KB, а не выдуманные;
- escalation_team логичен для meta_category;
- план соответствует тикету и классификации.

ok=false если шаги слишком общие, не из KB, или противоречат тикету.
issue — одна короткая фраза: что именно исправить."""


def critique_action(
    ticket_text: str,
    classification: TicketClassification,
    action: TicketActionResponse,
    kb_context: str,
) -> CriticVerdict:
    user = f"""Тикет:
{ticket_text[:2000]}

Классификация: {classification.model_dump_json()}
План L1: {action.model_dump_json()}

Фрагменты KB:
{kb_context[:4000]}
"""
    return client.chat.completions.create(
        model=MODEL,
        response_model=CriticVerdict,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
