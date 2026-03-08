#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.error
import urllib.request
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
MISTRAL_BASE = "https://api.mistral.ai/v1"

SITE_URL = "http://localhost"
SITE_NAME = "Telegram Multi-Provider AI Bot"

PAGE_SIZE = 8
MAX_HISTORY_MESSAGES = 200
MODEL_CHECK_CONCURRENCY = 6
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers concise and clear."
TELEGRAM_MESSAGE_CHUNK = 3900

BTN_PROVIDER = "Провайдер"
BTN_PICK_MODEL = "Выбрать модель"
BTN_REFRESH = "Обновить модели"
BTN_CLEAR = "Очистить диалог"
BTN_HELP = "Помощь"
BTN_PROFILE = "Профиль"

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GROQ = "groq"
PROVIDER_HF = "huggingface"
PROVIDER_MISTRAL = "mistral"

GROUP_ALL = "all"
GROUP_FAST = "fast"
GROUP_ECO = "eco"

# Approximate monthly token limits for Groq free-tier models (for profile estimation).
GROQ_MONTHLY_TOKEN_LIMITS: dict[str, int] = {
    "llama-3.1-8b-instant": 5_000_000,
    "llama3-8b-8192": 5_000_000,
    "mistral-7b-instruct-v0.2": 5_000_000,
    "gemma2-9b-it": 4_000_000,
    "mixtral-8x7b-32768": 3_000_000,
    "llama-3.3-70b-versatile": 2_000_000,
    "llama3-70b-8192": 2_000_000,
}


def model_group(provider_id: str, model_id: str) -> str:
    m = model_id.lower()
    if provider_id == PROVIDER_GROQ:
        limit = GROQ_MONTHLY_TOKEN_LIMITS.get(model_id)
        if limit is not None and limit <= 3_000_000:
            return GROUP_FAST
        if "70b" in m or "120b" in m or "405b" in m or "large" in m or "mixtral" in m:
            return GROUP_FAST
        return GROUP_ECO
    if "70b" in m or "120b" in m or "405b" in m or "large" in m:
        return GROUP_FAST
    return GROUP_ECO


def model_group_label(provider_id: str, model_id: str) -> str:
    grp = model_group(provider_id, model_id)
    if grp == GROUP_FAST:
        return f"[FAST] {model_id}"
    return f"[ECO] {model_id}"


def estimate_tokens(prompt_text: str, answer_text: str) -> int:
    return max(1, (len(prompt_text) + len(answer_text)) // 4)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_for_telegram(text: str, chunk_size: int = TELEGRAM_MESSAGE_CHUNK) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            parts.append(remaining)
            break
        cut = remaining.rfind("\n", 0, chunk_size)
        if cut < int(chunk_size * 0.5):
            cut = chunk_size
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return parts


async def reply_long(message, text: str, reply_markup=None) -> None:
    chunks = split_for_telegram(text)
    for i, part in enumerate(chunks):
        if i == len(chunks) - 1 and reply_markup is not None:
            await message.reply_text(part, reply_markup=reply_markup)
        else:
            await message.reply_text(part)


def _http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    body = None
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Codex-Telegram-Bot)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


async def call_json_with_retry(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 90,
    retries: int = 2,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(_http_json, url, method, payload, headers, timeout)
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries and "timed out" in str(e).lower():
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown network error")


def parse_http_error(e: urllib.error.HTTPError) -> tuple[int, str]:
    raw = e.read().decode("utf-8", errors="ignore")
    try:
        data = json.loads(raw)
        message = data.get("error", {}).get("message", raw)
        metadata_raw = data.get("error", {}).get("metadata", {}).get("raw", "")
        return e.code, f"{message}\n{metadata_raw}".strip()
    except Exception:
        return e.code, raw


def has_dev_instruction_error(detail: str) -> bool:
    return "Developer instruction is not enabled" in detail


def has_privacy_policy_error(detail: str) -> bool:
    return "No endpoints found matching your data policy" in detail


def has_rate_limit_error(code: int, detail: str) -> bool:
    return code == 429 or "rate-limited" in detail.lower()


def strip_system_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [m for m in messages if m.get("role") != "system"]


def initial_history() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


@dataclass
class ProviderResult:
    model_used: str | None
    answer: str | None
    warning: str | None
    usage: dict[str, int] | None = None


class BaseClient:
    provider_id: str
    title: str

    async def get_candidate_models(self) -> list[str]:
        raise NotImplementedError

    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        text, _ = await self.chat_with_usage(model, messages)
        return text

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        raise NotImplementedError

    async def check_model(self, model: str) -> tuple[bool, str]:
        probe = [{"role": "user", "content": "Reply with OK"}]
        try:
            await self.chat(model, probe)
            return True, ""
        except urllib.error.HTTPError as e:
            code, detail = parse_http_error(e)
            if has_dev_instruction_error(detail):
                try:
                    await self.chat(model, strip_system_messages(probe))
                    return True, ""
                except Exception:
                    pass
            if has_rate_limit_error(code, detail):
                return False, "rate-limited"
            return False, f"HTTP {code}"
        except Exception:
            return False, "network"


class OpenRouterClient(BaseClient):
    provider_id = PROVIDER_OPENROUTER
    title = "OpenRouter"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }

    async def get_candidate_models(self) -> list[str]:
        data = await call_json_with_retry(f"{OPENROUTER_BASE}/models", "GET", None, self.headers)
        models = data.get("data", [])
        free_ids = []
        for item in models:
            model_id = item.get("id", "")
            if ":free" in model_id:
                free_ids.append(model_id)
        return sorted(set(free_ids))

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        data = await call_json_with_retry(
            f"{OPENROUTER_BASE}/chat/completions",
            "POST",
            payload,
            self.headers,
        )
        text = data["choices"][0]["message"]["content"].strip()
        usage_raw = data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        return text, usage


class GroqClient(BaseClient):
    provider_id = PROVIDER_GROQ
    title = "Groq"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.known_free_text_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "mistral-saba-24b",
            "mistral-7b-instruct-v0.2",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ]

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    async def get_candidate_models(self) -> list[str]:
        endpoints = [f"{GROQ_BASE}/models", "https://api.groq.com/v1/models"]
        ids: list[str] = []
        has_pricing_field = False

        for url in endpoints:
            try:
                data = await call_json_with_retry(url, "GET", None, self.headers)
            except Exception:
                continue
            models = data.get("data", [])
            for item in models:
                model_id = item.get("id", "")
                if not model_id:
                    continue
                if item.get("active") is False:
                    continue
                low = model_id.lower()
                # Skip non-chat entries for chat UI.
                blocked = (
                    "embed",
                    "whisper",
                    "tts",
                    "transcribe",
                    "prompt-guard",
                    "moderation",
                    "guard",
                    "vision",
                    "image",
                    "audio",
                )
                if any(x in low for x in blocked):
                    continue
                if "pricing" in item:
                    has_pricing_field = True
                    if item.get("pricing") is None:
                        ids.append(model_id)
                else:
                    ids.append(model_id)

            if ids:
                break

        if not ids and not has_pricing_field:
            # Fallback when models endpoint is blocked or limited.
            ids = self.known_free_text_models.copy()

        return sorted(set(ids))

    async def check_model(self, model: str) -> tuple[bool, str]:
        probe = [{"role": "user", "content": "Reply with exactly OK"}]
        try:
            payload = {
                "model": model,
                "messages": probe,
                "temperature": 0,
                "max_completion_tokens": 16,
                "n": 1,
            }
            data = await call_json_with_retry(
                f"{GROQ_BASE}/chat/completions",
                "POST",
                payload,
                self.headers,
            )
            _ = data["choices"][0]["message"]["content"]
            return True, ""
        except urllib.error.HTTPError as e:
            code, detail = parse_http_error(e)
            if has_rate_limit_error(code, detail):
                return False, "rate-limited"
            return False, f"HTTP {code}"
        except Exception:
            return False, "network"

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        data = await call_json_with_retry(
            f"{GROQ_BASE}/chat/completions",
            "POST",
            payload,
            self.headers,
        )
        text = data["choices"][0]["message"]["content"].strip()
        usage_raw = data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        return text, usage


class HuggingFaceClient(BaseClient):
    provider_id = PROVIDER_HF
    title = "HuggingFace"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    async def get_candidate_models(self) -> list[str]:
        # Official OpenAI-compatible router endpoint.
        data = await call_json_with_retry("https://router.huggingface.co/v1/models", "GET", None, self.headers)
        models = data.get("data", [])
        ids = []
        for item in models:
            model_id = item.get("id", "")
            if model_id:
                ids.append(model_id)
        return sorted(set(ids))

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        data = await call_json_with_retry(
            "https://router.huggingface.co/v1/chat/completions",
            "POST",
            payload,
            self.headers,
        )
        text = data["choices"][0]["message"]["content"].strip()
        usage_raw = data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        return text, usage

    async def check_model(self, model: str) -> tuple[bool, str]:
        # HF models are often cold-started; allow warm-up during probe.
        probe = [{"role": "user", "content": "Say OK"}]
        try:
            _ = await self.chat(model, probe)
            return True, ""
        except urllib.error.HTTPError as e:
            code, detail = parse_http_error(e)
            return False, f"HTTP {code}: {detail[:120]}"
        except Exception as e:
            return False, str(e)[:120]


class MistralClient(BaseClient):
    provider_id = PROVIDER_MISTRAL
    title = "Mistral"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    async def get_candidate_models(self) -> list[str]:
        data = await call_json_with_retry(f"{MISTRAL_BASE}/models", "GET", None, self.headers)
        models = data.get("data", [])
        ids: list[str] = []
        for item in models:
            model_id = item.get("id", "")
            if not model_id:
                continue
            low = model_id.lower()
            if "embed" in low or "moderation" in low:
                continue
            ids.append(model_id)
        return sorted(set(ids))

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        data = await call_json_with_retry(
            f"{MISTRAL_BASE}/chat/completions",
            "POST",
            payload,
            self.headers,
        )
        text = data["choices"][0]["message"]["content"].strip()
        usage_raw = data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        return text, usage


def menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [BTN_PROVIDER, BTN_REFRESH],
            [BTN_PICK_MODEL, BTN_CLEAR],
            [BTN_PROFILE, BTN_HELP],
        ],
        resize_keyboard=True,
    )


def provider_keyboard(available: set[str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if PROVIDER_OPENROUTER in available:
        rows.append([InlineKeyboardButton("OpenRouter", callback_data=f"prov:{PROVIDER_OPENROUTER}")])
    if PROVIDER_GROQ in available:
        rows.append([InlineKeyboardButton("Groq", callback_data=f"prov:{PROVIDER_GROQ}")])
    if PROVIDER_HF in available:
        rows.append([InlineKeyboardButton("HuggingFace", callback_data=f"prov:{PROVIDER_HF}")])
    if PROVIDER_MISTRAL in available:
        rows.append([InlineKeyboardButton("Mistral", callback_data=f"prov:{PROVIDER_MISTRAL}")])
    rows.append([InlineKeyboardButton("Закрыть", callback_data="close")])
    return InlineKeyboardMarkup(rows)


def filtered_models_with_index(provider_id: str, models: list[str], group_filter: str) -> list[tuple[int, str]]:
    indexed = list(enumerate(models))
    if group_filter == GROUP_FAST:
        return [(i, m) for i, m in indexed if model_group(provider_id, m) == GROUP_FAST]
    if group_filter == GROUP_ECO:
        return [(i, m) for i, m in indexed if model_group(provider_id, m) == GROUP_ECO]
    return indexed


def models_keyboard(provider_id: str, models: list[str], page: int, group_filter: str) -> InlineKeyboardMarkup:
    indexed = filtered_models_with_index(provider_id, models, group_filter)
    total_pages = max(1, (len(indexed) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PAGE_SIZE
    chunk = indexed[start : start + PAGE_SIZE]

    rows: list[list[InlineKeyboardButton]] = []
    rows.append(
        [
            InlineKeyboardButton(f"{'•' if group_filter == GROUP_ALL else ''}ALL", callback_data=f"grp:{GROUP_ALL}"),
            InlineKeyboardButton(f"{'•' if group_filter == GROUP_FAST else ''}FAST", callback_data=f"grp:{GROUP_FAST}"),
            InlineKeyboardButton(f"{'•' if group_filter == GROUP_ECO else ''}ECO", callback_data=f"grp:{GROUP_ECO}"),
        ]
    )
    for idx, model in chunk:
        rows.append([InlineKeyboardButton(model_group_label(provider_id, model), callback_data=f"m:{idx}")])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Назад", callback_data=f"p:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Вперед", callback_data=f"p:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("Закрыть", callback_data="close")])
    return InlineKeyboardMarkup(rows)


def ensure_state(context: ContextTypes.DEFAULT_TYPE, default_provider: str) -> dict[str, Any]:
    return context.user_data.setdefault(
        "state",
        {
            "current_provider": default_provider,
            "selected_models": {
                PROVIDER_OPENROUTER: None,
                PROVIDER_GROQ: None,
                PROVIDER_HF: None,
                PROVIDER_MISTRAL: None,
            },
            "history": initial_history(),
            "model_group_filter": GROUP_ALL,
            "usage_stats": {},
            "usage_started_at": now_iso(),
        },
    )


def current_models_key(provider_id: str) -> str:
    return f"models:{provider_id}"


def provider_title(provider_id: str) -> str:
    if provider_id == PROVIDER_OPENROUTER:
        return "OpenRouter"
    if provider_id == PROVIDER_GROQ:
        return "Groq"
    if provider_id == PROVIDER_MISTRAL:
        return "Mistral"
    return "HuggingFace"


async def show_provider_picker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    available: set[str] = context.bot_data["available_providers"]
    await update.message.reply_text(
        "Выбери провайдера:",
        reply_markup=provider_keyboard(available),
    )


async def show_models(target_message, context: ContextTypes.DEFAULT_TYPE, page: int = 0) -> None:
    state = ensure_state(context, context.bot_data["default_provider"])
    provider_id: str = state["current_provider"]
    group_filter: str = state.get("model_group_filter", GROUP_ALL)
    models: list[str] = context.bot_data.get(current_models_key(provider_id), [])
    filtered = filtered_models_with_index(provider_id, models, group_filter)
    if not models or not filtered:
        await target_message.reply_text(
            f"Список доступных моделей для {provider_title(provider_id)} пуст. Нажми «{BTN_REFRESH}».",
            reply_markup=menu_keyboard(),
        )
        return
    await target_message.reply_text(
        f"<b>Выбор модели ({provider_title(provider_id)})</b>\n"
        "FAST = быстро тратят лимит, ECO = экономные.",
        parse_mode=ParseMode.HTML,
        reply_markup=models_keyboard(provider_id, models, page, group_filter),
    )


def update_usage_stats(
    state: dict[str, Any],
    provider_id: str,
    model_id: str,
    usage: dict[str, int] | None,
    user_text: str,
    answer_text: str,
) -> None:
    usage_stats: dict[str, dict[str, dict[str, int]]] = state.setdefault("usage_stats", {})
    provider_stats = usage_stats.setdefault(provider_id, {})
    row = provider_stats.setdefault(model_id, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    row["requests"] += 1

    if usage:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or 0)
    else:
        tt = estimate_tokens(user_text, answer_text)
        pt = max(1, len(user_text) // 4)
        ct = max(1, tt - pt)

    if tt <= 0:
        tt = pt + ct
    row["prompt_tokens"] += pt
    row["completion_tokens"] += ct
    row["total_tokens"] += tt


def format_profile(state: dict[str, Any]) -> str:
    provider_id: str = state["current_provider"]
    usage_stats: dict[str, dict[str, dict[str, int]]] = state.get("usage_stats", {})
    provider_stats = usage_stats.get(provider_id, {})
    started = state.get("usage_started_at", "-")
    title = provider_title(provider_id)
    if not provider_stats:
        return f"<b>Профиль ({title})</b>\nНет статистики запросов.\nСбор начнется после первого ответа модели."

    lines = [f"<b>Профиль ({title})</b>", f"Статистика с: <code>{started}</code>", ""]
    items = sorted(provider_stats.items(), key=lambda kv: kv[1].get("requests", 0), reverse=True)
    for model_id, stats in items[:20]:
        req = int(stats.get("requests", 0))
        total = int(stats.get("total_tokens", 0))
        avg = max(1, total // max(1, req))
        grp = "FAST" if model_group(provider_id, model_id) == GROUP_FAST else "ECO"
        line = f"{grp} <code>{model_id}</code>\nrequests: {req}, tokens: {total}, avg: {avg}"

        if provider_id == PROVIDER_GROQ:
            limit = GROQ_MONTHLY_TOKEN_LIMITS.get(model_id)
            if limit:
                left_tokens = max(0, limit - total)
                est_req_left = left_tokens // avg
                line += f"\nlimit: {limit}, left_tokens: {left_tokens}, est_req_left: {est_req_left}"
        lines.append(line)
        lines.append("")
    lines.append("Оценка остатка приблизительная и основана на токенах, собранных ботом.")
    return "\n".join(lines)


async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context, context.bot_data["default_provider"])
    text = format_profile(state)
    chunks = split_for_telegram(text)
    for i, part in enumerate(chunks):
        if i == len(chunks) - 1:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())
        else:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context, context.bot_data["default_provider"])
    current = provider_title(state["current_provider"])
    available = context.bot_data["available_providers"]
    providers_text = ", ".join(provider_title(p) for p in sorted(available))
    await update.message.reply_text(
        "<b>AI Bot</b>\n"
        f"Провайдеры: {providers_text}\n"
        f"Текущий: {current}\n"
        f"1) Нажми «{BTN_REFRESH}»\n"
        f"2) Нажми «{BTN_PICK_MODEL}»\n"
        "3) После выбора просто пиши сообщения",
        parse_mode=ParseMode.HTML,
        reply_markup=menu_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>Команды</b>\n"
        "/start - старт\n"
        "/models - выбор модели\n"
        "/profile - статистика и остаток\n"
        "/clear - очистить диалог\n"
        "Кнопка «Провайдер» переключает между OpenRouter, Groq, HuggingFace и Mistral.",
        parse_mode=ParseMode.HTML,
        reply_markup=menu_keyboard(),
    )


async def refresh_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context, context.bot_data["default_provider"])
    provider_id: str = state["current_provider"]
    client: BaseClient = context.bot_data["providers"][provider_id]
    state["refresh_provider"] = provider_id

    try:
        models = await client.get_candidate_models()
    except urllib.error.HTTPError as e:
        code, detail = parse_http_error(e)
        await update.message.reply_text(f"Ошибка {client.title} (HTTP {code}):\n{detail[:900]}")
        return
    except Exception as e:
        await update.message.reply_text(f"Не удалось обновить модели {client.title}: {e}")
        return

    if not models:
        context.bot_data[current_models_key(provider_id)] = []
        await update.message.reply_text(f"Не найдено моделей для {client.title}.", reply_markup=menu_keyboard())
        return

    status_msg = await update.message.reply_text(f"Проверяю доступность моделей {client.title}...")
    sem = asyncio.Semaphore(MODEL_CHECK_CONCURRENCY)

    async def run_check(model_id: str) -> tuple[str, bool, str]:
        async with sem:
            ok, reason = await client.check_model(model_id)
            return model_id, ok, reason

    results = await asyncio.gather(*(run_check(m) for m in models))
    good = [m for m, ok, _ in results if ok]
    bad = [(m, r) for m, ok, r in results if not ok]

    # Ignore stale refresh result if user switched provider while checks were running.
    if state.get("current_provider") != provider_id:
        await status_msg.edit_text(f"Проверка {client.title} завершена (уже переключено на другой провайдер).")
        return

    context.bot_data[current_models_key(provider_id)] = good
    context.bot_data[f"unavailable:{provider_id}"] = bad

    if not good:
        reason_stats: dict[str, int] = {}
        for _, reason in bad:
            reason_stats[reason] = reason_stats.get(reason, 0) + 1
        top_reason = ""
        if reason_stats:
            top_reason = max(reason_stats.items(), key=lambda x: x[1])[0]

        if provider_id == PROVIDER_HF:
            # HF can be flaky on health checks; keep a small fallback list so user can test manually.
            fallback = models[:20]
            context.bot_data[current_models_key(provider_id)] = fallback
            await status_msg.edit_text(
                f"Автопроверка {client.title} не нашла стабильных моделей. "
                f"Показал {len(fallback)} кандидатов для ручного теста."
            )
            await update.message.reply_text("Меню готово.", reply_markup=menu_keyboard())
            return

        if provider_id == PROVIDER_GROQ:
            hint = ""
            if "HTTP 403" in top_reason:
                fallback = models[:20]
                context.bot_data[current_models_key(provider_id)] = fallback
                hint = (
                    "\nПричина: Groq отклоняет health-check (403)."
                    "\nПоказал список кандидатов для ручного теста."
                )
                await status_msg.edit_text(
                    f"Автопроверка {client.title} не подтвердила доступные модели.{hint}"
                )
                await update.message.reply_text("Меню готово.", reply_markup=menu_keyboard())
                return
            elif "HTTP 401" in top_reason:
                hint = "\nПричина: ключ Groq недействителен (401)."
            elif "rate-limited" in top_reason or "HTTP 429" in top_reason:
                hint = "\nПричина: лимит/квота Groq (429)."
            elif top_reason:
                hint = f"\nОсновная причина: {top_reason}"
            await status_msg.edit_text(
                f"Сейчас нет доступных моделей {client.title}. Попробуй позже.{hint}"
            )
            await update.message.reply_text("Меню готово.", reply_markup=menu_keyboard())
            return

        await status_msg.edit_text(
            f"Сейчас нет доступных моделей {client.title}. Попробуй позже."
        )
        await update.message.reply_text("Меню готово.", reply_markup=menu_keyboard())
        return

    await status_msg.edit_text(
        f"Готово ({client.title}). Доступных: {len(good)} из {len(models)}. Скрыто: {len(bad)}."
    )
    await update.message.reply_text("Меню готово.", reply_markup=menu_keyboard())


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_models(update.message, context, page=0)


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context, context.bot_data["default_provider"])
    provider_id: str = state["current_provider"]
    selected = state["selected_models"].get(provider_id)
    state["history"] = initial_history()
    await update.message.reply_text(
        f"Диалог очищен. Текущая модель: {selected or 'не выбрана'}",
        reply_markup=menu_keyboard(),
    )


async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    state = ensure_state(context, context.bot_data["default_provider"])
    provider_id: str = state["current_provider"]
    group_filter: str = state.get("model_group_filter", GROUP_ALL)
    models: list[str] = context.bot_data.get(current_models_key(provider_id), [])

    if data == "noop":
        return

    if data.startswith("prov:"):
        new_provider = data.split(":", 1)[1]
        if new_provider not in context.bot_data["available_providers"]:
            await query.answer("Провайдер недоступен", show_alert=True)
            return
        state["current_provider"] = new_provider
        state["model_group_filter"] = GROUP_ALL
        state["history"] = initial_history()
        await query.edit_message_text(f"Провайдер переключен на: {provider_title(new_provider)}")
        await query.message.reply_text(
            "Теперь нажми «Обновить модели», затем «Выбрать модель».",
            reply_markup=menu_keyboard(),
        )
        return

    if data.startswith("p:"):
        page = int(data.split(":", 1)[1])
        await query.edit_message_reply_markup(reply_markup=models_keyboard(provider_id, models, page, group_filter))
        return

    if data.startswith("grp:"):
        state["model_group_filter"] = data.split(":", 1)[1]
        await query.edit_message_reply_markup(
            reply_markup=models_keyboard(provider_id, models, 0, state.get("model_group_filter", GROUP_ALL))
        )
        return

    if data.startswith("m:"):
        idx = int(data.split(":", 1)[1])
        if idx < 0 or idx >= len(models):
            await query.answer("Модель не найдена", show_alert=True)
            return
        selected = models[idx]
        state["selected_models"][provider_id] = selected
        state["history"] = initial_history()
        await query.edit_message_text(
            f"<b>Модель выбрана ({provider_title(provider_id)}):</b>\n<code>{selected}</code>",
            parse_mode=ParseMode.HTML,
        )
        await query.message.reply_text(
            "Теперь пиши сообщение, я отвечу через выбранную модель.",
            reply_markup=menu_keyboard(),
        )
        return

    if data == "close":
        await query.edit_message_text("Окно закрыто.")
        return


async def try_with_fallbacks(
    client: BaseClient,
    provider_id: str,
    current_model: str,
    history: list[dict[str, str]],
    models: list[str],
) -> ProviderResult:
    try:
        answer, usage = await client.chat_with_usage(current_model, history)
        return ProviderResult(current_model, answer, None, usage)
    except urllib.error.HTTPError as e:
        code, detail = parse_http_error(e)

        if has_dev_instruction_error(detail):
            try:
                answer, usage = await client.chat_with_usage(current_model, strip_system_messages(history))
                return ProviderResult(
                    current_model,
                    answer,
                    "Модель не принимает system-инструкции, запрос отправлен без них.",
                    usage,
                )
            except urllib.error.HTTPError:
                pass

        if provider_id == PROVIDER_OPENROUTER and has_privacy_policy_error(detail):
            return ProviderResult(
                None,
                None,
                (
                    "OpenRouter блокирует free endpoints вашей privacy-политикой. "
                    "Открой: https://openrouter.ai/settings/privacy и разреши Free model publication."
                ),
            )

        if not has_rate_limit_error(code, detail):
            return ProviderResult(None, None, f"Ошибка {client.title} (HTTP {code}):\n{detail[:900]}", None)
    except Exception as e:
        return ProviderResult(None, None, f"Ошибка запроса {client.title}: {e}", None)

    candidates = [m for m in models if m != current_model]
    for alt in candidates:
        try:
            answer, usage = await client.chat_with_usage(alt, history)
            return ProviderResult(
                alt,
                answer,
                f"Текущая модель недоступна, автоматически переключил на:\n<code>{alt}</code>",
                usage,
            )
        except urllib.error.HTTPError as e:
            _, detail = parse_http_error(e)
            if has_dev_instruction_error(detail):
                try:
                    answer, usage = await client.chat_with_usage(alt, strip_system_messages(history))
                    return ProviderResult(
                        alt,
                        answer,
                        (
                            "Автопереключение сработало. "
                            f"Выбрана модель:\n<code>{alt}</code>\n(запрос отправлен без system-инструкции)"
                        ),
                        usage,
                    )
                except Exception:
                    continue
            continue
        except Exception:
            continue

    return ProviderResult(None, None, "Сейчас нет доступных моделей. Нажми «Обновить модели» и попробуй позже.", None)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    low = text.lower()
    if low == BTN_PROVIDER.lower():
        await show_provider_picker(update, context)
        return
    if low == BTN_PICK_MODEL.lower():
        await show_models(update.message, context, page=0)
        return
    if low == BTN_REFRESH.lower():
        await refresh_models(update, context)
        return
    if low == BTN_CLEAR.lower():
        await clear_cmd(update, context)
        return
    if low == BTN_PROFILE.lower():
        await profile_cmd(update, context)
        return
    if low == BTN_HELP.lower():
        await help_cmd(update, context)
        return

    state = ensure_state(context, context.bot_data["default_provider"])
    provider_id: str = state["current_provider"]
    selected_model = state["selected_models"].get(provider_id)
    if not selected_model:
        await update.message.reply_text(
            f"Сначала выбери модель кнопкой «{BTN_PICK_MODEL}».",
            reply_markup=menu_keyboard(),
        )
        return

    history: list[dict[str, str]] = state["history"]
    history.append({"role": "user", "content": text})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1) :]
        state["history"] = history

    await update.message.chat.send_action("typing")
    client: BaseClient = context.bot_data["providers"][provider_id]
    models: list[str] = context.bot_data.get(current_models_key(provider_id), [])
    result = await try_with_fallbacks(client, provider_id, selected_model, history, models)

    if result.warning and result.answer is None:
        await update.message.reply_text(result.warning, reply_markup=menu_keyboard(), parse_mode=ParseMode.HTML)
        return
    if result.warning:
        await update.message.reply_text(result.warning, parse_mode=ParseMode.HTML)
    if result.model_used and result.model_used != selected_model:
        state["selected_models"][provider_id] = result.model_used
    if not result.answer:
        await update.message.reply_text("Не удалось получить ответ от модели.", reply_markup=menu_keyboard())
        return

    used_model = result.model_used or selected_model
    update_usage_stats(state, provider_id, used_model, result.usage, text, result.answer)

    history.append({"role": "assistant", "content": result.answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1) :]
        state["history"] = history

    await reply_long(update.message, result.answer, reply_markup=menu_keyboard())


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if update and isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"Внутренняя ошибка бота: {err}")


def validate_env() -> tuple[str, str | None, str | None, str | None, str | None]:
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or None
    hf_key = os.getenv("HF_API_KEY", "").strip() or None
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip() or None

    if not tg_token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN")
    if not or_key and not groq_key and not hf_key and not mistral_key:
        raise RuntimeError("Set at least one key: OPENROUTER_API_KEY or GROQ_API_KEY or HF_API_KEY or MISTRAL_API_KEY")

    if or_key:
        try:
            or_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("OPENROUTER_API_KEY must be ASCII key (sk-or-...)") from e
        if not or_key.startswith("sk-or-"):
            raise RuntimeError("OPENROUTER_API_KEY must start with sk-or-")

    if groq_key:
        try:
            groq_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("GROQ_API_KEY must be ASCII key (gsk_...)") from e

    if hf_key:
        try:
            hf_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("HF_API_KEY must be ASCII key (hf_...)") from e

    if mistral_key:
        try:
            mistral_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("MISTRAL_API_KEY must be ASCII") from e

    return tg_token, or_key, groq_key, hf_key, mistral_key


def main() -> None:
    tg_token, or_key, groq_key, hf_key, mistral_key = validate_env()
    providers: dict[str, BaseClient] = {}

    if or_key:
        providers[PROVIDER_OPENROUTER] = OpenRouterClient(or_key)
    if groq_key:
        providers[PROVIDER_GROQ] = GroqClient(groq_key)
    if hf_key:
        providers[PROVIDER_HF] = HuggingFaceClient(hf_key)
    if mistral_key:
        providers[PROVIDER_MISTRAL] = MistralClient(mistral_key)

    if PROVIDER_OPENROUTER in providers:
        default_provider = PROVIDER_OPENROUTER
    elif PROVIDER_GROQ in providers:
        default_provider = PROVIDER_GROQ
    elif PROVIDER_MISTRAL in providers:
        default_provider = PROVIDER_MISTRAL
    else:
        default_provider = PROVIDER_HF

    app = Application.builder().token(tg_token).build()
    app.bot_data["providers"] = providers
    app.bot_data["available_providers"] = set(providers.keys())
    app.bot_data["default_provider"] = default_provider
    app.bot_data[current_models_key(PROVIDER_OPENROUTER)] = []
    app.bot_data[current_models_key(PROVIDER_GROQ)] = []
    app.bot_data[current_models_key(PROVIDER_HF)] = []
    app.bot_data[current_models_key(PROVIDER_MISTRAL)] = []

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CallbackQueryHandler(callback_router))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    port = int(os.getenv("PORT", "0") or "0")
    if port > 0:
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format, *args):
                return

        def run_health_server() -> None:
            httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
            httpd.serve_forever()

        threading.Thread(target=run_health_server, daemon=True).start()

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

