from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


T = TypeVar("T")


def use_cursor() -> bool:
    """True — ходим в Cursor API через cursor-sdk (Agent.prompt)."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "cursor":
        return True
    if provider in ("openai", "deepseek", "custom"):
        return False
    if os.environ.get("CURSOR_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        return True
    return False


def _cursor_prompt(messages: list[dict], model: str) -> str:
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CURSOR_API_KEY не задан. Получите ключ: Cursor → Dashboard → Integrations → API Keys"
        )
    cwd = os.environ.get("CURSOR_CWD", os.getcwd())
    parts: list[str] = []
    for m in messages:
        parts.append(f"[{m['role']}]\n{m['content']}")
    prompt = "\n\n".join(parts)

    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=api_key,
            model=model,
            local=LocalAgentOptions(cwd=cwd),
        ),
    )
    if not result.result:
        raise RuntimeError(f"Cursor Agent вернул пустой ответ (status={result.status})")
    return result.result


def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_AUTH_TOKEN не задан. Либо экспортируй токен, "
                "либо положи LLM_AUTH_TOKEN=... в .env."
            )
        timeout = float(os.environ.get("LLM_TIMEOUT", "200"))
        http = httpx.Client(verify=False, timeout=timeout)
        return OpenAI(api_key=key, base_url=base, http_client=http)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Ни LLM_BASE_URL, ни OPENAI_API_KEY не заданы. "
            "Сконфигурируй стенд через .env (см. .env.example)."
        )
    return OpenAI(api_key=key)


def get_model() -> str:
    if use_cursor():
        return os.environ.get("LLM_MODEL", "composer-2.5")
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def _thinking_off_payload() -> dict:
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}
    return {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_effort": "none",
    }


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text).strip()
    # ```json ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str):
    t = _clean(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"В ответе не найдено валидного JSON: {text[:300]!r}")


class _CursorCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 1,
        temperature: float = 0.0,
        **kw: Any,
    ) -> T:
        del temperature, kw
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": item_schema}},
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        addendum = (
            f"\n\nОтвечай ОДНИМ валидным JSON-объектом по схеме:\n{schema_str}\n"
            "ТОЛЬКО JSON. Никакого текста до/после, никакого markdown."
        )
        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum.lstrip()})

        last_err: Exception | None = None
        raw = ""
        for _ in range(max_retries + 1):
            try:
                raw = _cursor_prompt(msgs, model)
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                return adapter.validate_python(obj)
            except Exception as e:
                last_err = e
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({
                    "role": "user",
                    "content": f"Невалидный ответ: {e}. Верни ТОЛЬКО один корректный JSON по схеме.",
                })
        assert last_err is not None
        raise last_err


class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 1,
        temperature: float = 0.0,
        **kw: Any,
    ) -> T:
        # list[Model] → оборачиваем в {items: [...]}, т.к. JSON mode требует object
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": item_schema}},
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        addendum = (
            f"\n\nОтвечай ОДНИМ валидным JSON-объектом по схеме:\n{schema_str}\n"
            "ТОЛЬКО JSON. Никакого текста до/после, никакого markdown, "
            "никаких комментариев, никаких повторных объектов."
        )
        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum.lstrip()})

        # Отключаем reasoning — иначе Qwen3 может по 30+ секунд «думать» перед ответом.
        thinking_kw = _thinking_off_payload()

        def _call(kw: dict):
            try:
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **kw,
                )
            except TypeError:
                # Старый SDK не знает reasoning_effort
                safe = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **safe,
                )

        last_err: Exception | None = None
        raw: str = ""
        for _ in range(max_retries + 1):
            try:
                try:
                    resp = _call(thinking_kw)
                except Exception as sdk_err:
                    # Сервер не переварил reasoning_effort / extra_body — сбросим их и повторим.
                    msg = str(sdk_err)
                    bad = "reasoning_effort" in msg or "chat_template_kwargs" in msg or "enable_thinking" in msg
                    if bad and thinking_kw:
                        thinking_kw = {}
                        resp = _call(thinking_kw)
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                return adapter.validate_python(obj)
            except Exception as e:
                last_err = e
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({
                    "role": "user",
                    "content": f"Невалидный ответ: {e}. Верни ТОЛЬКО один корректный JSON по схеме.",
                })
        assert last_err is not None
        raise last_err


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    """Drop-in замена instructor-клиента."""

    def __init__(self, openai_client: OpenAI | None = None, *, cursor: bool = False):
        if cursor:
            self.chat = type("_Chat", (), {"completions": _CursorCompletions()})()
        else:
            assert openai_client is not None
            self._c = openai_client
            self.chat = _Chat(openai_client)


def make_client() -> JsonClient:
    """Вернуть клиент с API `client.chat.completions.create(..., response_model=...)`."""
    if use_cursor():
        return JsonClient(cursor=True)
    return JsonClient(_make_openai_client())


# ---------------------------------------------------------------------------
# «Сырой» клиент без JSON-инструктора, но с автоотключением reasoning
# ---------------------------------------------------------------------------
# Нам нужен в семинаре 2: мы хотим увидеть грязный ответ модели как есть
# (markdown, «возраст словом», пост-амбула). Но reasoning всё равно надо
# гасить — иначе Qwen3 думает по 30 секунд перед каждым ответом, а никакого
# учебного смысла это не несёт.


class _RawCompletions:
    """Прокси над openai.chat.completions: инжектирует thinking-off kwargs."""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kw: Any):
        thinking = _thinking_off_payload()

        def _call(extra: dict):
            try:
                return self._inner.create(**kw, **extra)
            except TypeError:
                # Старый SDK не знает reasoning_effort — снимаем и повторяем.
                safe = {k: v for k, v in extra.items() if k != "reasoning_effort"}
                return self._inner.create(**kw, **safe)

        try:
            return _call(thinking)
        except Exception as e:
            msg = str(e)
            bad = (
                "reasoning_effort" in msg
                or "chat_template_kwargs" in msg
                or "enable_thinking" in msg
            )
            if bad and thinking:
                # Сервер не переварил — повторим без thinking-kwargs.
                return _call({})
            raise


class _RawChat:
    def __init__(self, inner):
        self.completions = _RawCompletions(inner.completions)


class RawClient:
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _RawChat(openai_client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def make_raw_client() -> RawClient:
    if use_cursor():
        raise RuntimeError(
            "make_raw_client() недоступен при LLM_PROVIDER=cursor."
        )
    return RawClient(_make_openai_client())
