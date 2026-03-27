#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import urllib.parse
import urllib.error
import urllib.request
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
    WebAppInfo,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
MISTRAL_BASE = "https://api.mistral.ai/v1"
SILICONFLOW_BASE = "https://api.siliconflow.com/v1"
LEGNEXT_BASE = os.getenv("LEGNEXT_BASE_URL", "https://api.legnext.ai/api/v1")
POLLINATIONS_TEXT_BASES = (
    "https://gen.pollinations.ai/v1",
    "https://text.pollinations.ai/openai/v1",
)
POLLINATIONS_GEN_BASE = os.getenv("POLLINATIONS_GEN_BASE", "https://gen.pollinations.ai")

SITE_URL = os.getenv("SITE_URL", "http://localhost")
SITE_NAME = "Telegram Multi-Provider AI Bot"

PAGE_SIZE = 8
AGENTS_PAGE_SIZE = 8
ROLES_PAGE_SIZE = 8
MAX_HISTORY_MESSAGES = 200
MODEL_CHECK_CONCURRENCY = 6
MAX_CONCURRENT_REQUESTS = 2
MAX_GLOBAL_AGENTS = 50
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers concise and clear."
TELEGRAM_MESSAGE_CHUNK = 3900
TELEGRAM_SEND_RETRIES = 4
TELEGRAM_SEND_DELAY_SEC = 0.08

BTN_PICK_MODEL = "\U0001f9e0 \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c"
BTN_CLEAR = "\U0001f9f9 \u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0434\u0438\u0430\u043b\u043e\u0433"
BTN_HELP = "\u2139\ufe0f \u041f\u043e\u043c\u043e\u0449\u044c"
BTN_PROFILE = "\U0001f4ca \u041f\u0440\u043e\u0444\u0438\u043b\u044c"
BTN_LIMITS = "\u23f1\ufe0f \u041b\u0438\u043c\u0438\u0442\u044b"
BTN_ROLE = "\U0001f3ad \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0440\u043e\u043b\u044c"
BTN_WEB = "\U0001f310 Web UI"

WEB_RUNTIME: dict[str, Any] = {}

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GROQ = "groq"
PROVIDER_HF = "huggingface"
PROVIDER_MISTRAL = "mistral"
PROVIDER_SILICONFLOW = "siliconflow"
PROVIDER_LEGNEXT = "legnext"
PROVIDER_POLLINATIONS = "pollinations"

GROUP_ALL = "all"
GROUP_FAST = "fast"
GROUP_ECO = "eco"

# Model type filters.
MODEL_TYPE_CHAT = "chat"
MODEL_TYPE_IMAGE = "image"
MODEL_TYPE_VIDEO = "video"

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

DEFAULT_POLLINATIONS_TEXT_MODELS = ["gpt-5", "claude", "gemini", "deepseek", "qwen3-coder"]
DEFAULT_POLLINATIONS_IMAGE_MODELS = ["flux", "gptimage-large", "seedream", "kontext"]
DEFAULT_POLLINATIONS_VIDEO_MODELS = ["seedance", "veo"]
DEFAULT_LEGNEXT_IMAGE_MODELS = ["midjourney"]
DEFAULT_LEGNEXT_VIDEO_MODELS = ["midjourney-video"]
STATIC_CHAT_MODELS: dict[str, list[str]] = {
    PROVIDER_GROQ: [
        "allam-2-7b",
        "groq/compound",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    PROVIDER_OPENROUTER: [
        "arcee-ai/trinity-large-preview:free",
    ],
    PROVIDER_HF: [
        "CohereLabs/aya-expanse-32b",
        "CohereLabs/c4ai-command-a-03-2025",
        "CohereLabs/c4ai-command-r-08-2024",
        "CohereLabs/c4ai-command-r7b-12-2024",
        "CohereLabs/c4ai-command-r7b-arabic-02-2025",
    ],
    PROVIDER_POLLINATIONS: [
        "gpt-5",
        "claude",
        "gemini",
        "deepseek",
        "qwen3-coder",
    ],
}

PREFERRED_CHAT_MODELS = [
    "llama-3.3-70b-versatile",
    "arcee-ai/trinity-large-preview:free",
    "gpt-5",
    "claude",
    "gemini",
    "deepseek",
    "qwen3-coder",
    "CohereLabs/aya-expanse-32b",
    "CohereLabs/c4ai-command-a-03-2025",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/compound",
    "groq/compound-mini",
]

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


def model_key(provider_id: str, model_id: str, model_type: str) -> str:
    if model_type == MODEL_TYPE_CHAT:
        return model_id
    return f"{provider_id}:{model_id}"


def split_model_key(key: str) -> tuple[str, str]:
    parts = key.split(":", 1)
    if len(parts) != 2:
        return "", key
    return parts[0], parts[1]


def estimate_tokens(prompt_text: str, answer_text: str) -> int:
    return max(1, (len(prompt_text) + len(answer_text)) // 4)


def has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


def has_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


def answer_looks_broken(user_text: str, answer_text: str) -> bool:
    if not answer_text or len(answer_text.strip()) < 8:
        return True
    if has_cyrillic(user_text) and has_arabic(answer_text):
        return True
    upper_words = answer_text.split()
    if upper_words:
        repeated = 0
        prev = None
        for word in upper_words:
            token = word.strip(".,!?:;()[]{}\"'").lower()
            if token and token == prev:
                repeated += 1
            prev = token
        if repeated >= 8:
            return True
    if answer_text.count("ВЫБОР") >= 6 or answer_text.count("ТРАНСТРОЙТЕЛЬ") >= 4:
        return True
    return False


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


async def telegram_api_call(bot_data: dict[str, Any] | None, func, *args, **kwargs):
    semaphore: asyncio.Semaphore | None = None
    if bot_data is not None:
        semaphore = bot_data.get("telegram_send_semaphore")

    async def _run():
        last_exc: Exception | None = None
        for attempt in range(TELEGRAM_SEND_RETRIES):
            try:
                if TELEGRAM_SEND_DELAY_SEC > 0:
                    await asyncio.sleep(TELEGRAM_SEND_DELAY_SEC)
                return await func(*args, **kwargs)
            except RetryAfter as e:
                last_exc = e
                await asyncio.sleep(float(getattr(e, "retry_after", 1.0)) + 0.2)
            except (TimedOut, NetworkError) as e:
                last_exc = e
                await asyncio.sleep(0.6 * (attempt + 1))
        if last_exc:
            raise last_exc
        return await func(*args, **kwargs)

    if semaphore is None:
        return await _run()
    async with semaphore:
        return await _run()


def provider_badge_html(provider_id: str | None, model_id: str | None) -> str:
    if not provider_id and not model_id:
        return ""
    title = provider_title(provider_id or "") if provider_id else "AI"
    parts = [f"⚡ <b>{html.escape(title)}</b>"]
    if model_id:
        parts.append(f"<code>{html.escape(model_id)}</code>")
    return " · ".join(parts)


def _apply_inline_markup(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<b>{m.group(1)}</b>", text)
    text = re.sub(r"`([^`\n]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    return text


def markdownish_to_telegram_html(text: str) -> str:
    if not text:
        return ""

    parts: list[str] = []
    last = 0
    code_pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)
    for match in code_pattern.finditer(text):
        plain = text[last:match.start()]
        if plain:
            escaped = html.escape(plain)
            escaped = _apply_inline_markup(escaped)
            parts.append(escaped.replace("\n", "\n"))
        lang = match.group(1).strip()
        code = html.escape(match.group(2).strip("\n"))
        if lang:
            parts.append(f"<b>{html.escape(lang)}</b>\n<pre><code>{code}</code></pre>")
        else:
            parts.append(f"<pre><code>{code}</code></pre>")
        last = match.end()

    tail = text[last:]
    if tail:
        escaped = html.escape(tail)
        escaped = _apply_inline_markup(escaped)
        parts.append(escaped)
    return "".join(parts)


def build_telegram_answer_html(answer: str, provider_id: str | None, model_id: str | None) -> str:
    badge = provider_badge_html(provider_id, model_id)
    body = markdownish_to_telegram_html(answer)
    if badge and body:
        return f"{badge}\n\n{body}"
    return badge or body


def estimate_cost_usd(provider_id: str | None, usage: dict[str, int] | None) -> float:
    if not provider_id or not usage:
        return 0.0
    # Current catalog is built around free/test models. Keep visible but honest.
    return 0.0


def format_metadata_lines(
    provider_id: str | None,
    model_id: str | None,
    usage: dict[str, int] | None,
    elapsed_sec: float | None,
) -> list[str]:
    total_tokens = int((usage or {}).get("total_tokens", 0) or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
    completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
    cost = estimate_cost_usd(provider_id, usage)
    return [
        f"🤖 Модель: {model_id or '-'}",
        f"🌐 API: {provider_title(provider_id or '') if provider_id else '-'}",
        f"⏳ Время: {elapsed_sec:.1f} сек" if elapsed_sec is not None else "⏳ Время: -",
        f"💳 Токены: {total_tokens} (in {prompt_tokens} / out {completion_tokens})",
        f"💰 Цена: ${cost:.4f}",
    ]


def format_metadata_html(
    provider_id: str | None,
    model_id: str | None,
    usage: dict[str, int] | None,
    elapsed_sec: float | None,
) -> str:
    return "<pre>" + html.escape("\n".join(format_metadata_lines(provider_id, model_id, usage, elapsed_sec))) + "</pre>"


async def reply_answer(
    message,
    answer: str,
    provider_id: str | None,
    model_id: str | None,
    usage: dict[str, int] | None = None,
    elapsed_sec: float | None = None,
    bot_data: dict[str, Any] | None = None,
    reply_markup=None,
) -> None:
    formatted = build_telegram_answer_html(answer, provider_id, model_id)
    metadata = format_metadata_html(provider_id, model_id, usage, elapsed_sec)
    combined = f"{formatted}\n\n{metadata}" if metadata else formatted
    if len(combined) <= TELEGRAM_MESSAGE_CHUNK:
        await telegram_api_call(bot_data, message.reply_text, combined, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        return

    badge = provider_badge_html(provider_id, model_id)
    if badge:
        await telegram_api_call(bot_data, message.reply_text, badge, parse_mode=ParseMode.HTML)

    chunks = split_for_telegram(answer)
    for i, part in enumerate(chunks):
        chunk_html = markdownish_to_telegram_html(part)
        if i == len(chunks) - 1 and reply_markup is not None:
            await telegram_api_call(bot_data, message.reply_text, chunk_html, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        else:
            await telegram_api_call(bot_data, message.reply_text, chunk_html, parse_mode=ParseMode.HTML)
    await telegram_api_call(bot_data, message.reply_text, metadata, parse_mode=ParseMode.HTML, reply_markup=reply_markup)


async def progress_bar_updater(status_message, stop_event: asyncio.Event, bot_data: dict[str, Any] | None = None) -> None:
    frames = [10, 24, 39, 53, 67, 79, 90, 94]
    idx = 0
    while not stop_event.is_set():
        percent = frames[idx % len(frames)]
        filled = max(0, min(10, round(percent / 10)))
        bar = "█" * filled + "░" * (10 - filled)
        try:
            await telegram_api_call(bot_data, status_message.edit_text, f"Генерирую... [{bar}] {percent}%")
        except Exception:
            return
        idx += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue


async def start_progress_message(message, bot_data: dict[str, Any] | None = None):
    status = await telegram_api_call(bot_data, message.reply_text, "Генерирую... [░░░░░░░░░░] 0%")
    stop_event = asyncio.Event()
    task = asyncio.create_task(progress_bar_updater(status, stop_event, bot_data))
    return status, stop_event, task


async def finish_progress_message(status_message, stop_event: asyncio.Event, task: asyncio.Task, bot_data: dict[str, Any] | None = None, ok: bool = True) -> None:
    stop_event.set()
    try:
        await task
    except Exception:
        pass
    try:
        await telegram_api_call(bot_data, status_message.edit_text, "Готово. [██████████] 100%" if ok else "Ошибка генерации.")
    except Exception:
        pass


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


def parse_csv_env_list(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def parse_limit_map(value: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, raw = chunk.split("=", 1)
        key = key.strip()
        raw = raw.strip().replace("_", "")
        if not key or not raw.isdigit():
            continue
        result[key] = int(raw)
    return result


def strip_system_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [m for m in messages if m.get("role") != "system"]


def role_prompt(role_id: str | None) -> str:
    strict_prefix = (
        "You must strictly follow the assigned role. "
        "Stay in role for the entire reply. "
        "Answer only from the perspective of that specialist. "
        "Do not switch to a generic assistant voice. "
        "If information is missing, ask clarifying questions that fit the role. "
        "Answer in Russian unless the user explicitly asks for another language. "
    )
    if not role_id:
        return strict_prefix + SYSTEM_PROMPT
    spec = ROLE_MAP.get(role_id)
    if not spec:
        return strict_prefix + SYSTEM_PROMPT
    contract = role_output_contract(role_id)
    if contract:
        return strict_prefix + spec.system_prompt + " " + contract
    return strict_prefix + spec.system_prompt


def role_title(role_id: str | None) -> str:
    if not role_id:
        return "Универсал"
    spec = ROLE_MAP.get(role_id)
    if not spec:
        return "Универсал"
    return spec.title


def role_output_contract(role_id: str | None) -> str:
    contracts = {
        "avitolog": (
            "Return the final answer in Russian with these exact sections and nothing extra: "
            "1. Разбор 2. Готовый заголовок 3. Готовое описание 4. Скрипт ответа клиенту 5. CTA. "
            "If data is missing, first give a short practical draft based on reasonable assumptions, then list up to 3 уточняющих вопроса."
        ),
        "mechanic": (
            "Return the final answer in Russian with these exact sections and nothing extra: "
            "1. Причины 2. Проверка 3. Ремонт 4. Риски. "
            "Use step-by-step numbered checks where appropriate."
        ),
        "repair_master": (
            "Return the final answer in Russian with these exact sections and nothing extra: "
            "1. Возможные причины 2. Проверка 3. Что делать 4. Когда в сервис."
        ),
        "programmer": (
            "Return the final answer in Russian with these exact sections when relevant: "
            "1. Что сделать 2. Код или пример 3. Проверка 4. Подводные камни."
        ),
    }
    return contracts.get(role_id or "", "")


def initial_history(role_id: str | None = None) -> list[dict[str, str]]:
    return [{"role": "system", "content": role_prompt(role_id)}]


@dataclass
class ProviderResult:
    model_used: str | None
    answer: str | None
    warning: str | None
    usage: dict[str, int] | None = None


@dataclass
class ModelEntry:
    key: str
    provider_id: str
    model_id: str
    model_type: str
    providers: list[str] | None = None


@dataclass
class AgentSpec:
    agent_id: str
    provider_id: str
    model_id: str


@dataclass
class RoleSpec:
    role_id: str
    title: str
    system_prompt: str


ROLE_SPECS: list[RoleSpec] = [
    RoleSpec("general", "Универсал", "You are a practical, concise general assistant. Answer in Russian unless asked otherwise."),
    RoleSpec("teacher", "Учитель", "You are a patient teacher. Explain step-by-step, with simple examples and short checks for understanding."),
    RoleSpec("lang_teacher", "Учитель языков", "You are a foreign language teacher. Help with grammar, pronunciation, vocabulary, dialogues, exercises, corrections, and structured learning plans. Correct mistakes carefully and explain why the correction is right."),
    RoleSpec("coder", "Кодер", "You are a senior software engineer. Provide robust code, edge cases, and practical implementation details."),
    RoleSpec("programmer", "Программист", "You are a professional programmer. Solve coding tasks with correct architecture, working code, debugging steps, practical tradeoffs, and concise technical explanations."),
    RoleSpec("debugger", "Отладчик", "You are a debugging specialist. Isolate root cause, propose reproducible checks, and minimal safe fixes."),
    RoleSpec("architect", "Архитектор", "You are a software architect. Design scalable, maintainable systems with clear tradeoffs."),
    RoleSpec("product", "Продакт", "You are a product manager. Define goals, user value, metrics, and prioritized roadmap."),
    RoleSpec("marketing", "Маркетолог", "You are a marketing strategist. Build offers, positioning, channels, and measurable campaigns."),
    RoleSpec("copywriter", "Копирайтер", "You are a conversion copywriter. Write clear, persuasive texts with strong structure and CTA."),
    RoleSpec("sales", "Продажник", "You are a sales advisor. Qualify needs, handle objections, and propose next best sales actions."),
    RoleSpec("avitolog", "Авитолог", "You are an Avito lead-generation and conversion expert. Answer only in Russian. Treat the user as a seller or intermediary on Avito. Give practical recommendations only for Avito: title, category, geo, USP, trust, offer, funnel, scripts, objections, lead qualification, and promotion. Default reply format: 1) Короткий вывод 2) Что исправить 3) Готовый текст для Avito 4) Следующий шаг. Never generate nonsense examples or placeholder spam."),
    RoleSpec("repair_master", "Ремонт техники", "You are a universal repair technician. Answer only in Russian. Diagnose phones, laptops, PCs, home appliances, TVs, routers, and electronics. Default reply format: 1) Возможные причины 2) Что проверить по шагам 3) Что нужно для ремонта 4) Когда лучше не лезть и отдать в сервис. Prioritize safety and clear diagnostics."),
    RoleSpec("gardener", "Садовод", "You are an experienced gardener. Answer only in Russian. Help with soil, watering, fertilizers, pests, pruning, planting schedules, seedlings, fruit trees, greenhouse work, and seasonal care. Default reply format: 1) Диагноз ситуации 2) Что делать сейчас 3) Что делать дальше 4) Чего избегать."),
    RoleSpec("mechanic", "Механик", "You are a practical mechanic. Answer only in Russian. Diagnose cars, motorcycles, scooters, and equipment. Default reply format: 1) Вероятные причины 2) Проверка по шагам 3) Ремонт/замена 4) Риски и безопасность. Avoid vague generic advice."),
    RoleSpec("recruiter", "Рекрутер", "You are a recruiter and career advisor. Optimize resumes, vacancies, and interview strategy."),
    RoleSpec("law_basic", "Юрист-черновик", "You are a legal drafting assistant (not a lawyer). Create safe draft texts and highlight legal risks."),
    RoleSpec("finance_basic", "Финансы", "You are a personal finance analyst. Build budgets, scenarios, and risk-aware plans."),
    RoleSpec("analyst", "Аналитик", "You are a data analyst. Structure problems, compute assumptions, and present concise conclusions."),
    RoleSpec("scientist", "Исследователь", "You are a research assistant. Evaluate hypotheses, methods, and evidence quality."),
    RoleSpec("biologist", "Биолог", "You are a biology tutor. Explain concepts accurately in accessible terms."),
    RoleSpec("chemist", "Химик", "You are a chemistry tutor. Explain mechanisms and equations clearly and safely."),
    RoleSpec("physicist", "Физик", "You are a physics tutor. Use first principles, formulas, and intuitive interpretations."),
    RoleSpec("doctor_info", "Мед-справка", "You are a medical information assistant, not a doctor. Provide cautious educational info and advise professional care."),
    RoleSpec("psychology", "Психолог (база)", "You are a supportive psychology educator. Offer non-clinical guidance and coping techniques."),
    RoleSpec("translator", "Переводчик", "You are a professional translator. Preserve meaning, tone, and context."),
    RoleSpec("editor", "Редактор", "You are a strict editor. Improve clarity, structure, and correctness without changing intent."),
    RoleSpec("summarizer", "Краткий обзор", "You are a summarization assistant. Extract key facts, decisions, and action items."),
    RoleSpec("prompt_engineer", "Промпт-инженер", "You are a prompt engineer. Create precise prompts, constraints, and evaluation criteria."),
    RoleSpec("smm", "SMM-менеджер", "You are an SMM specialist. Build content plans, hooks, and posting strategy."),
    RoleSpec("seo", "SEO-специалист", "You are an SEO specialist. Propose semantic core, structure, and on-page optimization."),
    RoleSpec("designer", "Дизайнер", "You are a UI/UX advisor. Propose practical layout, hierarchy, and interaction improvements."),
    RoleSpec("qa", "QA-инженер", "You are a QA engineer. Create test cases, risks, and regression checklist."),
    RoleSpec("devops", "DevOps", "You are a DevOps engineer. Improve CI/CD, reliability, and observability with pragmatic steps."),
    RoleSpec("security", "ИБ-специалист", "You are a defensive security advisor. Focus on hardening, detection, and secure configuration."),
    RoleSpec("historian", "Историк", "You are a history explainer. Provide context, chronology, and critical nuance."),
    RoleSpec("travel", "Тревел-консультант", "You are a travel planner. Build practical itineraries, budgets, and logistics."),
    RoleSpec("chef", "Шеф-повар", "You are a culinary advisor. Give precise recipes and substitutions by available ingredients."),
    RoleSpec("fitness", "Фитнес-тренер", "You are a fitness coach. Build safe, progressive training plans."),
    RoleSpec("storyteller", "Сценарист", "You are a creative storyteller. Write vivid scenes, characters, and plot structure."),
    RoleSpec("crime_fiction", "Преступник (fiction)", "You are a crime-fiction character consultant for stories and analysis only. Never provide real-world wrongdoing instructions."),
]

ROLE_MAP: dict[str, RoleSpec] = {r.role_id: r for r in ROLE_SPECS}

WEB_ROLE_TITLES: dict[str, str] = {
    "general": "Универсал",
    "teacher": "Учитель",
    "lang_teacher": "Учитель языков",
    "coder": "Кодер",
    "programmer": "Программист",
    "debugger": "Отладчик",
    "architect": "Архитектор",
    "product": "Продакт",
    "marketing": "Маркетолог",
    "copywriter": "Копирайтер",
    "sales": "Продажник",
    "avitolog": "Авитолог",
    "repair_master": "Ремонт техники",
    "gardener": "Садовод",
    "mechanic": "Механик",
    "recruiter": "Рекрутер",
    "law_basic": "Юрист-черновик",
    "finance_basic": "Финансы",
    "analyst": "Аналитик",
    "scientist": "Исследователь",
    "biologist": "Биолог",
    "chemist": "Химик",
    "physicist": "Физик",
    "doctor_info": "Мед-справка",
    "psychology": "Психолог",
    "translator": "Переводчик",
    "editor": "Редактор",
    "summarizer": "Краткий обзор",
    "prompt_engineer": "Промпт-инженер",
    "smm": "SMM",
    "seo": "SEO",
    "designer": "Дизайнер",
    "qa": "QA",
    "devops": "DevOps",
    "security": "Безопасность",
    "historian": "Историк",
    "travel": "Тревел",
    "chef": "Шеф-повар",
    "fitness": "Фитнес",
    "storyteller": "Сценарист",
    "crime_fiction": "Crime Fiction",
}


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


class SiliconFlowClient(BaseClient):
    provider_id = PROVIDER_SILICONFLOW
    title = "SiliconFlow"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    async def get_candidate_models(self) -> list[str]:
        data = await call_json_with_retry(f"{SILICONFLOW_BASE}/models", "GET", None, self.headers)
        models = data.get("data", [])
        ids: list[str] = []
        for item in models:
            model_id = item.get("id", "")
            if not model_id:
                continue
            low = model_id.lower()
            if any(x in low for x in ("embed", "rerank", "moderation", "image", "audio", "vision")):
                continue
            ids.append(model_id)
        return sorted(set(ids))

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        data = await call_json_with_retry(
            f"{SILICONFLOW_BASE}/chat/completions",
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


class PollinationsTextClient(BaseClient):
    provider_id = PROVIDER_POLLINATIONS
    title = "Pollinations"

    def __init__(self, api_key: str | None, fallback_models: list[str]):
        self.api_key = api_key or ""
        self.fallback_models = fallback_models

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def get_candidate_models(self) -> list[str]:
        ids: list[str] = []
        for base in POLLINATIONS_TEXT_BASES:
            try:
                data = await call_json_with_retry(f"{base}/models", "GET", None, self.headers)
            except Exception:
                continue
            if isinstance(data, dict):
                models = data.get("data", [])
            else:
                models = data or []
            for item in models:
                model_id = item.get("id", "")
                if model_id:
                    ids.append(model_id)
            if ids:
                break
        if not ids:
            ids = self.fallback_models.copy()
        return sorted(set(ids))

    async def chat_with_usage(self, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
        payload = {"model": model, "messages": messages, "temperature": 0.6}
        last_err: Exception | None = None
        for base in POLLINATIONS_TEXT_BASES:
            try:
                data = await call_json_with_retry(
            f"{base}/chat/completions",
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
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Pollinations chat failed")


def menu_keyboard() -> ReplyKeyboardMarkup:
    rows: list[list[str | KeyboardButton]] = [
        [BTN_PICK_MODEL, BTN_CLEAR],
        [BTN_ROLE, BTN_PROFILE],
        [BTN_LIMITS, BTN_HELP],
    ]
    if SITE_URL.startswith("https://"):
        rows.append([KeyboardButton(BTN_WEB, web_app=WebAppInfo(url=SITE_URL))])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def start_shortcuts_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("🎭 Роль", callback_data="open_roles"),
            InlineKeyboardButton("🧠 Модель", callback_data="open_models"),
        ],
        [
            InlineKeyboardButton("📊 Профиль", callback_data="quick_profile"),
            InlineKeyboardButton("⏱️ Лимиты", callback_data="quick_limits"),
        ],
    ]
    if SITE_URL.startswith("https://"):
        rows.append([InlineKeyboardButton("🌐 Открыть Web UI", url=SITE_URL)])
    return InlineKeyboardMarkup(rows)


def model_entry_label(entry: ModelEntry) -> str:
    if entry.model_type == MODEL_TYPE_CHAT and entry.providers:
        providers = entry.providers
        prefix = provider_title(providers[0])
        if len(providers) > 1:
            prefix = f"{prefix}+{len(providers)-1}"
    else:
        prefix = provider_title(entry.provider_id)
    if entry.model_type == MODEL_TYPE_CHAT:
        label = model_group_label(entry.provider_id, entry.model_id)
    elif entry.model_type == MODEL_TYPE_IMAGE:
        label = f"[IMG] {entry.model_id}"
    else:
        label = f"[VID] {entry.model_id}"
    return f"{prefix} | {label}"


def filter_model_entries(
    catalog: list[ModelEntry],
    model_type: str,
    group_filter: str,
) -> list[ModelEntry]:
    filtered = [m for m in catalog if m.model_type == model_type]
    if model_type == MODEL_TYPE_CHAT and group_filter != GROUP_ALL:
        filtered = [m for m in filtered if model_group(m.provider_id, m.model_id) == group_filter]
    return filtered


def models_keyboard(
    entries: list[ModelEntry],
    page: int,
    model_type: str,
    group_filter: str,
) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(entries) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PAGE_SIZE
    chunk = entries[start : start + PAGE_SIZE]

    rows: list[list[InlineKeyboardButton]] = []
    rows.append(
        [
            InlineKeyboardButton(
        f"{'•' if model_type == MODEL_TYPE_CHAT else ''}CHAT",
                callback_data=f"t:{MODEL_TYPE_CHAT}",
            ),
            InlineKeyboardButton(
        f"{'•' if model_type == MODEL_TYPE_IMAGE else ''}IMG",
                callback_data=f"t:{MODEL_TYPE_IMAGE}",
            ),
            InlineKeyboardButton(
        f"{'•' if model_type == MODEL_TYPE_VIDEO else ''}VIDEO",
                callback_data=f"t:{MODEL_TYPE_VIDEO}",
            ),
        ]
    )

    if model_type == MODEL_TYPE_CHAT:
        rows.append(
            [
                InlineKeyboardButton(
            f"{'•' if group_filter == GROUP_ALL else ''}ALL",
                    callback_data=f"grp:{GROUP_ALL}",
                ),
                InlineKeyboardButton(
            f"{'•' if group_filter == GROUP_FAST else ''}FAST",
                    callback_data=f"grp:{GROUP_FAST}",
                ),
                InlineKeyboardButton(
            f"{'•' if group_filter == GROUP_ECO else ''}ECO",
                    callback_data=f"grp:{GROUP_ECO}",
                ),
            ]
        )

    for idx, entry in enumerate(chunk, start=start):
        rows.append([InlineKeyboardButton(model_entry_label(entry), callback_data=f"m:{idx}")])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Назад", callback_data=f"p:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Вперед", callback_data=f"p:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("Закрыть", callback_data="close")])
    return InlineKeyboardMarkup(rows)


def ensure_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    return context.user_data.setdefault(
        "state",
        {
            "selected_model_key": None,
            "model_type_filter": MODEL_TYPE_CHAT,
            "model_group_filter": GROUP_ALL,
            "selected_role_id": "general",
            "history": initial_history("general"),
            "usage_stats": {},
            "usage_started_at": now_iso(),
            "selected_agent_id": None,
            "agent_mode": "manual",
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
    if provider_id == PROVIDER_SILICONFLOW:
        return "SiliconFlow"
    if provider_id == PROVIDER_LEGNEXT:
        return "LegNext"
    if provider_id == PROVIDER_POLLINATIONS:
        return "Pollinations"
    return "HuggingFace"


def build_global_agents(context: ContextTypes.DEFAULT_TYPE) -> list[AgentSpec]:
    providers: set[str] = context.bot_data.get("chat_providers", set())
    all_agents: list[AgentSpec] = []
    idx = 1
    for provider_id in sorted(providers):
        models: list[str] = context.bot_data.get(current_models_key(provider_id), [])
        for model_id in models:
            if len(all_agents) >= MAX_GLOBAL_AGENTS:
                return all_agents
            all_agents.append(
                AgentSpec(
                    agent_id=f"a{idx}",
                    provider_id=provider_id,
                    model_id=model_id,
                )
            )
            idx += 1
    return all_agents


def find_agent_by_id(context: ContextTypes.DEFAULT_TYPE, agent_id: str) -> AgentSpec | None:
    agents: list[AgentSpec] = context.bot_data.get("global_agents", [])
    for agent in agents:
        if agent.agent_id == agent_id:
            return agent
    return None


def format_agents_list(agents: list[AgentSpec], limit: int = 50) -> str:
    if not agents:
        return f"Список агентов пуст."
    lines = ["<b>Агенты</b>", "Формат: /agent ID", ""]
    for agent in agents[:limit]:
        model_safe = html.escape(agent.model_id)
        lines.append(
            f"<code>{agent.agent_id}</code> | {provider_title(agent.provider_id)} | <code>{model_safe}</code>"
        )
    if len(agents) > limit:
        lines.append("")
        lines.append(f"Показано {limit} из {len(agents)}.")
    return "\n".join(lines)


def compact_model_label(model_id: str, max_len: int = 36) -> str:
    if len(model_id) <= max_len:
        return model_id
    return model_id[: max_len - 1] + "…"


def agents_keyboard(agents: list[AgentSpec], page: int = 0) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(agents) + AGENTS_PAGE_SIZE - 1) // AGENTS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * AGENTS_PAGE_SIZE
    chunk = agents[start : start + AGENTS_PAGE_SIZE]

    rows: list[list[InlineKeyboardButton]] = []
    for agent in chunk:
        label = f"{agent.agent_id} | {provider_title(agent.provider_id)} | {compact_model_label(agent.model_id)}"
        rows.append([InlineKeyboardButton(label, callback_data=f"ag:{agent.agent_id}")])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Назад", callback_data=f"agp:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Вперед", callback_data=f"agp:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("Выключить агента", callback_data="ag:off")])
    rows.append([InlineKeyboardButton("Закрыть", callback_data="close")])
    return InlineKeyboardMarkup(rows)


def roles_keyboard(selected_role_id: str | None, page: int = 0) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(ROLE_SPECS) + ROLES_PAGE_SIZE - 1) // ROLES_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * ROLES_PAGE_SIZE
    chunk = ROLE_SPECS[start : start + ROLES_PAGE_SIZE]

    rows: list[list[InlineKeyboardButton]] = []
    for role in chunk:
        marker = "• " if role.role_id == selected_role_id else ""
        rows.append([InlineKeyboardButton(f"{marker}{role.title}", callback_data=f"rr:{role.role_id}")])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Назад", callback_data=f"rp:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Вперед", callback_data=f"rp:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("Закрыть", callback_data="close")])
    return InlineKeyboardMarkup(rows)


async def show_models(target_message, context: ContextTypes.DEFAULT_TYPE, page: int = 0) -> None:
    state = ensure_state(context)
    catalog: list[ModelEntry] = context.bot_data.get("models_catalog", [])
    if not catalog:
        await refresh_all_cmd(target_message, context, notify_only=True)
        catalog = context.bot_data.get("models_catalog", [])
    model_type: str = state.get("model_type_filter", MODEL_TYPE_CHAT)
    group_filter: str = state.get("model_group_filter", GROUP_ALL)
    filtered = filter_model_entries(catalog, model_type, group_filter)
    if not filtered:
        await target_message.reply_text(
            f"\u0421\u043f\u0438\u0441\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439 ({model_type}) \u043f\u0443\u0441\u0442.",
            reply_markup=menu_keyboard(),
        )
        return
    await target_message.reply_text(
        "<b>\u0412\u044b\u0431\u043e\u0440 \u043c\u043e\u0434\u0435\u043b\u0438</b>\n"
        "CHAT: FAST = \u0431\u044b\u0441\u0442\u0440\u043e \u0442\u0440\u0430\u0442\u044f\u0442 \u043b\u0438\u043c\u0438\u0442, ECO = \u044d\u043a\u043e\u043d\u043e\u043c\u043d\u044b\u0435.",
        parse_mode=ParseMode.HTML,
        reply_markup=models_keyboard(filtered, page, model_type, group_filter),
    )
async def show_agents_picker(target_message, context: ContextTypes.DEFAULT_TYPE, page: int = 0) -> None:
    agents: list[AgentSpec] = context.bot_data.get("global_agents", [])
    if not agents:
        agents = build_global_agents(context)
        context.bot_data["global_agents"] = agents
    if not agents:
        await target_message.reply_text(
            f"Агенты пока не собраны.",
            reply_markup=menu_keyboard(),
        )
        return
    await target_message.reply_text(
        "<b>Выбор агента</b>\nОдин агент = одна модель одного провайдера.",
        parse_mode=ParseMode.HTML,
        reply_markup=agents_keyboard(agents, page),
    )


async def show_roles_picker(target_message, context: ContextTypes.DEFAULT_TYPE, page: int = 0) -> None:
    state = ensure_state(context)
    selected_role_id = state.get("selected_role_id", "general")
    await target_message.reply_text(
        "<b>Выбор роли</b>\nРоль задает стиль и специализацию ответа.",
        parse_mode=ParseMode.HTML,
        reply_markup=roles_keyboard(selected_role_id, page),
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


def format_profile(state: dict[str, Any], catalog_by_key: dict[str, ModelEntry]) -> str:
    usage_stats: dict[str, dict[str, dict[str, int]]] = state.get("usage_stats", {})
    started = state.get("usage_started_at", "-")
    if not usage_stats:
        return "<b>Профиль</b>\nНет статистики запросов.\nСбор начнется после первого ответа модели."

    lines = ["<b>Профиль</b>", f"Статистика с: <code>{started}</code>", ""]
    for provider_id in sorted(usage_stats.keys()):
        provider_stats = usage_stats.get(provider_id, {})
        if not provider_stats:
            continue
        lines.append(f"<b>{provider_title(provider_id)}</b>")
        items = sorted(provider_stats.items(), key=lambda kv: kv[1].get("requests", 0), reverse=True)
        for model_id, stats in items[:20]:
            req = int(stats.get("requests", 0))
            total = int(stats.get("total_tokens", 0))
            avg = max(1, total // max(1, req))
            entry = catalog_by_key.get(model_key(provider_id, model_id, MODEL_TYPE_CHAT))
            model_type = entry.model_type if entry else MODEL_TYPE_CHAT
            if model_type == MODEL_TYPE_CHAT:
                grp = "FAST" if model_group(provider_id, model_id) == GROUP_FAST else "ECO"
                line = f"{grp} <code>{model_id}</code>\nrequests: {req}, tokens: {total}, avg: {avg}"
                if provider_id == PROVIDER_GROQ:
                    limit = GROQ_MONTHLY_TOKEN_LIMITS.get(model_id)
                    if limit:
                        left_tokens = max(0, limit - total)
                        est_req_left = left_tokens // avg
                        line += f"\nlimit: {limit}, left_tokens: {left_tokens}, est_req_left: {est_req_left}"
            else:
                line = f"{model_type.upper()} <code>{model_id}</code>\nrequests: {req}"
            lines.append(line)
            lines.append("")
        lines.append("")
    lines.append("Оценка остатка приблизительная и основана на токенах, собранных ботом.")
    return "\n".join(lines)


async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context)
    catalog_by_key: dict[str, ModelEntry] = context.bot_data.get("catalog_by_key", {})
    text = format_profile(state, catalog_by_key)
    chunks = split_for_telegram(text)
    for i, part in enumerate(chunks):
        if i == len(chunks) - 1:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())
        else:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML)


def get_token_limit(context: ContextTypes.DEFAULT_TYPE, provider_id: str, model_id: str) -> int | None:
    overrides: dict[str, int] = context.bot_data.get("token_limits", {})
    key = f"{provider_id}:{model_id}"
    if key in overrides:
        return overrides[key]
    if provider_id == PROVIDER_GROQ:
        return GROQ_MONTHLY_TOKEN_LIMITS.get(model_id)
    return None


def get_request_limit(context: ContextTypes.DEFAULT_TYPE, provider_id: str, model_id: str) -> int | None:
    overrides: dict[str, int] = context.bot_data.get("request_limits", {})
    key = f"{provider_id}:{model_id}"
    return overrides.get(key)


def format_limits(state: dict[str, Any], context: ContextTypes.DEFAULT_TYPE) -> str:
    usage_stats: dict[str, dict[str, dict[str, int]]] = state.get("usage_stats", {})
    catalog_by_key: dict[str, ModelEntry] = context.bot_data.get("catalog_by_key", {})
    if not usage_stats:
        return (
            "<b>\u041b\u0438\u043c\u0438\u0442\u044b</b>\n"
            "\u041d\u0435\u0442 \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0438 \u0437\u0430\u043f\u0440\u043e\u0441\u043e\u0432. "
            "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u043e\u0442\u043f\u0440\u0430\u0432\u044c \u0445\u043e\u0442\u044f \u0431\u044b 1 \u0437\u0430\u043f\u0440\u043e\u0441."
        )

    lines = ["<b>\u041b\u0438\u043c\u0438\u0442\u044b</b>", ""]
    for provider_id in sorted(usage_stats.keys()):
        provider_stats = usage_stats.get(provider_id, {})
        if not provider_stats:
            continue
        lines.append(f"<b>{provider_title(provider_id)}</b>")
        for model_id, stats in sorted(provider_stats.items(), key=lambda kv: kv[1].get("requests", 0), reverse=True):
            req = int(stats.get("requests", 0))
            total_tokens = int(stats.get("total_tokens", 0))
            token_limit = get_token_limit(context, provider_id, model_id)
            req_limit = get_request_limit(context, provider_id, model_id)
            entry = catalog_by_key.get(model_key(provider_id, model_id, MODEL_TYPE_CHAT))
            model_type = entry.model_type if entry else MODEL_TYPE_CHAT

            line = f"<code>{model_id}</code>"
            details: list[str] = []
            if model_type == MODEL_TYPE_CHAT and token_limit:
                left_tokens = max(0, token_limit - total_tokens)
                avg = max(1, total_tokens // max(1, req))
                est_req_left = left_tokens // avg
                details.append(f"tokens: {total_tokens}/{token_limit} (left {left_tokens})")
                details.append(f"est req left: {est_req_left}")
            if req_limit:
                left_req = max(0, req_limit - req)
                details.append(f"requests: {req}/{req_limit} (left {left_req})")
            if not details:
                details.append(f"requests: {req}")
            lines.append(line)
            lines.extend(details)
            lines.append("")
        lines.append("")
    lines.append(
        "\u041b\u0438\u043c\u0438\u0442\u044b \u043e\u0440\u0438\u0435\u043d\u0442\u0438\u0440\u043e\u0432\u043e\u0447\u043d\u044b\u0435. "
        "\u0422\u043e\u0447\u043d\u044b\u0435 \u043b\u0438\u043c\u0438\u0442\u044b \u043c\u043e\u0436\u043d\u043e \u0437\u0430\u0434\u0430\u0442\u044c \u0447\u0435\u0440\u0435\u0437 MODEL_TOKEN_LIMITS \u0438 MODEL_REQUEST_LIMITS."
    )
    return "\n".join(lines)
async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context)
    text = format_limits(state, context)
    chunks = split_for_telegram(text)
    for i, part in enumerate(chunks):
        if i == len(chunks) - 1:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())
        else:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context)
    current_role = role_title(state.get("selected_role_id"))
    available = context.bot_data.get("available_providers", set())
    providers_text = ", ".join(provider_title(p) for p in sorted(available))
    await refresh_all_cmd(update, context, notify_only=True)
    await update.message.reply_text(
        "<b>AI Bot</b>\n"
        "Темная multi-provider среда для chat, image и video.\n\n"
        f"<b>Провайдеры:</b> {html.escape(providers_text)}\n"
        "<b>Типы:</b> CHAT / IMG / VIDEO\n"
        f"<b>Роль:</b> {html.escape(current_role)}\n\n"
        "<b>Быстрый старт</b>\n"
        "1. Нажми <b>«Выбрать роль»</b>\n"
        "2. Нажми <b>«Выбрать модель»</b>\n"
        "3. Пиши сообщение или открой <b>Web UI</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=start_shortcuts_keyboard(),
    )
    await update.message.reply_text(
        "Подсказки: <code>сделай продающий текст</code>, <code>найди причину поломки</code>, <code>напиши код</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=menu_keyboard(),
    )
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>Управление</b>\n"
        "• <b>🎭 Выбрать роль</b> — задаёт специализацию ответа\n"
        "• <b>🧠 Выбрать модель</b> — выбирает конкретную модель\n"
        "• <b>📊 Профиль</b> — показывает статистику использования\n"
        "• <b>⏱️ Лимиты</b> — показывает оценку остатка по лимитам\n"
        "• <b>🧹 Очистить диалог</b> — сбрасывает историю\n"
        "• <b>🌐 Web UI</b> — тёмный интерфейс с пузырями и подсветкой кода\n\n"
        "<b>Формат ответов</b>\n"
        "Код отображается в блоках, важные мысли выделяются, под каждым ответом виден провайдер и модель.",
        parse_mode=ParseMode.HTML,
        reply_markup=start_shortcuts_keyboard(),
    )
async def refresh_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await refresh_all_cmd(update, context, notify_only=False)


async def refresh_provider_models(
    context: ContextTypes.DEFAULT_TYPE,
    provider_id: str,
    verify: bool = False,
) -> tuple[int, int, str]:
    models = list(STATIC_CHAT_MODELS.get(provider_id, []))
    if not models:
        context.bot_data[current_models_key(provider_id)] = []
        context.bot_data[f"unavailable:{provider_id}"] = []
        return 0, 0, f"Не найдено моделей для {provider_title(provider_id)}."
    context.bot_data[current_models_key(provider_id)] = models
    context.bot_data[f"unavailable:{provider_id}"] = []
    return len(models), len(models), f"Готово ({provider_title(provider_id)})."


def build_model_catalog(context: ContextTypes.DEFAULT_TYPE) -> list[ModelEntry]:
    catalog: list[ModelEntry] = []
    chat_providers: set[str] = context.bot_data.get("chat_providers", set())
    chat_map: dict[str, set[str]] = {}
    for provider_id in sorted(chat_providers):
        models: list[str] = context.bot_data.get(current_models_key(provider_id), [])
        for model_id in models:
            chat_map.setdefault(model_id, set()).add(provider_id)
    for model_id, providers in sorted(chat_map.items(), key=lambda kv: kv[0].lower()):
        prov_list = sorted(providers)
        catalog.append(
            ModelEntry(
                key=model_key(prov_list[0], model_id, MODEL_TYPE_CHAT),
                provider_id=prov_list[0],
                model_id=model_id,
                model_type=MODEL_TYPE_CHAT,
                providers=prov_list,
            )
        )

    legnext_image_models = context.bot_data.get("legnext_image_models", [])
    legnext_video_models = context.bot_data.get("legnext_video_models", [])
    pollinations_image_models = context.bot_data.get("pollinations_image_models", [])
    pollinations_video_models = context.bot_data.get("pollinations_video_models", [])

    for model_id in legnext_image_models:
        catalog.append(
            ModelEntry(
                key=model_key(PROVIDER_LEGNEXT, model_id, MODEL_TYPE_IMAGE),
                provider_id=PROVIDER_LEGNEXT,
                model_id=model_id,
                model_type=MODEL_TYPE_IMAGE,
            )
        )
    for model_id in legnext_video_models:
        catalog.append(
            ModelEntry(
                key=model_key(PROVIDER_LEGNEXT, model_id, MODEL_TYPE_VIDEO),
                provider_id=PROVIDER_LEGNEXT,
                model_id=model_id,
                model_type=MODEL_TYPE_VIDEO,
            )
        )
    for model_id in pollinations_image_models:
        catalog.append(
            ModelEntry(
                key=model_key(PROVIDER_POLLINATIONS, model_id, MODEL_TYPE_IMAGE),
                provider_id=PROVIDER_POLLINATIONS,
                model_id=model_id,
                model_type=MODEL_TYPE_IMAGE,
            )
        )
    for model_id in pollinations_video_models:
        catalog.append(
            ModelEntry(
                key=model_key(PROVIDER_POLLINATIONS, model_id, MODEL_TYPE_VIDEO),
                provider_id=PROVIDER_POLLINATIONS,
                model_id=model_id,
                model_type=MODEL_TYPE_VIDEO,
            )
        )

    context.bot_data["models_catalog"] = catalog
    context.bot_data["catalog_by_key"] = {entry.key: entry for entry in catalog}
    return catalog


def preferred_chat_model_key(catalog: list[ModelEntry]) -> str | None:
    chat_entries = [entry for entry in catalog if entry.model_type == MODEL_TYPE_CHAT]
    if not chat_entries:
        return None
    by_id = {entry.model_id: entry.key for entry in chat_entries}
    for model_id in PREFERRED_CHAT_MODELS:
        key = by_id.get(model_id)
        if key:
            return key
    return chat_entries[0].key


def preferred_fallback_chat_entries(
    catalog: list[ModelEntry],
    current_key: str | None,
) -> list[ModelEntry]:
    chat_entries = [entry for entry in catalog if entry.model_type == MODEL_TYPE_CHAT and entry.key != current_key]
    rank = {model_id: idx for idx, model_id in enumerate(PREFERRED_CHAT_MODELS)}
    return sorted(
        chat_entries,
        key=lambda entry: (rank.get(entry.model_id, 10_000), entry.model_id.lower()),
    )


async def refresh_all_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    notify_only: bool = False,
) -> None:
    # Use static pre-verified lists to avoid slow refresh checks.
    for provider_id in sorted(context.bot_data.get("chat_providers", set())):
        models = list(STATIC_CHAT_MODELS.get(provider_id, []))
        context.bot_data[current_models_key(provider_id)] = models
        context.bot_data[f"unavailable:{provider_id}"] = []

    agents = build_global_agents(context)
    context.bot_data["global_agents"] = agents
    catalog = build_model_catalog(context)

    if update.effective_user:
        ustate = ensure_state(context)
        if catalog and not ustate.get("selected_model_key"):
            preferred_key = preferred_chat_model_key(catalog)
            ustate["selected_model_key"] = preferred_key or catalog[0].key
        if agents and not ustate.get("selected_agent_id"):
            ustate["selected_agent_id"] = agents[0].agent_id

    if notify_only:
        return

    chat_count = sum(1 for entry in catalog if entry.model_type == MODEL_TYPE_CHAT)
    media_count = sum(1 for entry in catalog if entry.model_type != MODEL_TYPE_CHAT)
    await update.message.reply_text(
        f"Меню готово. Моделей: {chat_count}. Медиа: {media_count}.",
        reply_markup=menu_keyboard(),
    )


async def agents_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agents: list[AgentSpec] = context.bot_data.get("global_agents", [])
    if not agents:
        agents = build_global_agents(context)
        context.bot_data["global_agents"] = agents
    text = format_agents_list(agents)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())


async def agent_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context)
    args = context.args or []
    if not args:
        current = state.get("selected_agent_id") or "off"
        await update.message.reply_text(
            f"Текущий агент: {current}\nИспользование: /agent a1 или /agent off",
            reply_markup=menu_keyboard(),
        )
        return

    raw = args[0].strip().lower()
    if raw in {"off", "none", "0"}:
        state["selected_agent_id"] = None
        state["agent_mode"] = "manual"
        await update.message.reply_text("Режим агента выключен.", reply_markup=menu_keyboard())
        return

    agents: list[AgentSpec] = context.bot_data.get("global_agents", [])
    if not agents:
        agents = build_global_agents(context)
        context.bot_data["global_agents"] = agents

    found = find_agent_by_id(context, raw)
    if not found:
        await update.message.reply_text("Агент не найден. Открой /agents", reply_markup=menu_keyboard())
        return

    state["selected_agent_id"] = found.agent_id
    state["agent_mode"] = "manual"
    state["history"] = initial_history(state.get("selected_role_id"))
    state["selected_model_key"] = model_key(found.provider_id, found.model_id)
    await update.message.reply_text(
        f"Агент выбран: {found.agent_id}\n{provider_title(found.provider_id)} | {found.model_id}",
        reply_markup=menu_keyboard(),
    )


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_models(update.message, context, page=0)


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_state(context)
    selected_key = state.get("selected_model_key")
    catalog_by_key: dict[str, ModelEntry] = context.bot_data.get("catalog_by_key", {})
    entry = catalog_by_key.get(selected_key) if selected_key else None
    selected = entry.model_id if entry else None
    state["history"] = initial_history(state.get("selected_role_id"))
    await update.message.reply_text(
        f"Диалог очищен. Текущая модель: {selected or 'не выбрана'}",
        reply_markup=menu_keyboard(),
    )


async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    state = ensure_state(context)
    catalog: list[ModelEntry] = context.bot_data.get("models_catalog", [])
    model_type: str = state.get("model_type_filter", MODEL_TYPE_CHAT)
    group_filter: str = state.get("model_group_filter", GROUP_ALL)
    filtered = filter_model_entries(catalog, model_type, group_filter)

    if data == "noop":
        return

    if data == "open_roles":
        await query.message.reply_text("Открыл выбор ролей.", reply_markup=menu_keyboard())
        await show_roles_picker(query.message, context, page=0)
        return

    if data == "open_models":
        await query.message.reply_text("Открыл выбор моделей.", reply_markup=menu_keyboard())
        await show_models(query.message, context, page=0)
        return

    if data == "quick_profile":
        state = ensure_state(context)
        catalog_by_key: dict[str, ModelEntry] = context.bot_data.get("catalog_by_key", {})
        text = format_profile(state, catalog_by_key)
        chunks = split_for_telegram(text)
        for i, part in enumerate(chunks):
            if i == len(chunks) - 1:
                await query.message.reply_text(part, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())
            else:
                await query.message.reply_text(part, parse_mode=ParseMode.HTML)
        return

    if data == "quick_limits":
        state = ensure_state(context)
        text = format_limits(state, context)
        chunks = split_for_telegram(text)
        for i, part in enumerate(chunks):
            if i == len(chunks) - 1:
                await query.message.reply_text(part, parse_mode=ParseMode.HTML, reply_markup=menu_keyboard())
            else:
                await query.message.reply_text(part, parse_mode=ParseMode.HTML)
        return

    if data.startswith("p:"):
        page = int(data.split(":", 1)[1])
        await query.edit_message_reply_markup(reply_markup=models_keyboard(filtered, page, model_type, group_filter))
        return

    if data.startswith("agp:"):
        page = int(data.split(":", 1)[1])
        agents: list[AgentSpec] = context.bot_data.get("global_agents", [])
        await query.edit_message_reply_markup(reply_markup=agents_keyboard(agents, page))
        return

    if data.startswith("ag:"):
        agent_id = data.split(":", 1)[1]
        if agent_id == "off":
            state["selected_agent_id"] = None
            state["history"] = initial_history(state.get("selected_role_id"))
            await query.edit_message_text("Режим агента выключен.")
            await query.message.reply_text("Выбери модель вручную или нового агента.", reply_markup=menu_keyboard())
            return
        found = find_agent_by_id(context, agent_id)
        if not found:
            await query.answer("Агент не найден", show_alert=True)
            return
        state["selected_agent_id"] = found.agent_id
        state["history"] = initial_history(state.get("selected_role_id"))
        state["selected_model_key"] = model_key(found.provider_id, found.model_id, MODEL_TYPE_CHAT)
        await query.edit_message_text(
            f"Выбран агент: {found.agent_id}\n{provider_title(found.provider_id)} | {found.model_id}"
        )
        await query.message.reply_text("Теперь просто пиши сообщение.", reply_markup=menu_keyboard())
        return

    if data.startswith("rp:"):
        page = int(data.split(":", 1)[1])
        await query.edit_message_reply_markup(
            reply_markup=roles_keyboard(state.get("selected_role_id", "general"), page)
        )
        return

    if data.startswith("rr:"):
        role_id = data.split(":", 1)[1]
        if role_id not in ROLE_MAP:
            await query.answer("Роль не найдена", show_alert=True)
            return
        state["selected_role_id"] = role_id
        state["history"] = initial_history(role_id)
        preferred_key = preferred_chat_model_key(context.bot_data.get("models_catalog", []))
        if preferred_key:
            state["selected_model_key"] = preferred_key
        await query.edit_message_text(f"Роль выбрана: {role_title(role_id)}")
        await query.message.reply_text("Роль применена. Пиши сообщение.", reply_markup=menu_keyboard())
        return

    if data.startswith("grp:"):
        state["model_group_filter"] = data.split(":", 1)[1]
        model_type = state.get("model_type_filter", MODEL_TYPE_CHAT)
        group_filter = state.get("model_group_filter", GROUP_ALL)
        filtered = filter_model_entries(catalog, model_type, group_filter)
        await query.edit_message_reply_markup(
            reply_markup=models_keyboard(filtered, 0, model_type, group_filter)
        )
        return

    if data.startswith("t:"):
        state["model_type_filter"] = data.split(":", 1)[1]
        model_type = state.get("model_type_filter", MODEL_TYPE_CHAT)
        group_filter = state.get("model_group_filter", GROUP_ALL)
        filtered = filter_model_entries(catalog, model_type, group_filter)
        await query.edit_message_reply_markup(
            reply_markup=models_keyboard(filtered, 0, model_type, group_filter)
        )
        return

    if data.startswith("m:"):
        idx = int(data.split(":", 1)[1])
        if idx < 0 or idx >= len(filtered):
            await query.answer("Модель не найдена", show_alert=True)
            return
        entry = filtered[idx]
        state["selected_agent_id"] = None
        state["selected_model_key"] = entry.key
        if entry.model_type == MODEL_TYPE_CHAT:
            state["history"] = initial_history(state.get("selected_role_id"))
        await query.edit_message_text(
            f"<b>Модель выбрана ({provider_title(entry.provider_id)}):</b>\n<code>{entry.model_id}</code>",
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
    allow_model_fallback: bool = True,
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
        if not allow_model_fallback:
            return ProviderResult(None, None, f"Ошибка {client.title} (HTTP {code}):\n{detail[:220]}", None)
    except Exception as e:
        if not allow_model_fallback:
            return ProviderResult(None, None, f"Ошибка запроса {client.title}: {e}", None)
        return ProviderResult(None, None, f"Ошибка запроса {client.title}: {e}", None)

    if not allow_model_fallback:
        return ProviderResult(None, None, "Модель недоступна.", None)

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

    return ProviderResult(None, None, "Сейчас нет доступных моделей. Попробуй позже.", None)


def pollinations_media_url(prompt: str, model_id: str, media_type: str, api_key: str | None) -> str:
    safe_prompt = urllib.parse.quote(prompt)
    base = f"{POLLINATIONS_GEN_BASE}/{media_type}/{safe_prompt}"
    params = {"model": model_id}
    if api_key:
        params["key"] = api_key
    return f"{base}?{urllib.parse.urlencode(params)}"


async def legnext_create_task(api_key: str, endpoint: str, payload: dict[str, Any]) -> str:
    headers = {"x-api-key": api_key}
    data = await call_json_with_retry(f"{LEGNEXT_BASE}/{endpoint}", "POST", payload, headers)
    return str(
        data.get("task_id")
        or data.get("id")
        or data.get("task", {}).get("id")
        or data.get("task", {}).get("task_id")
        or ""
    )


async def legnext_get_task(api_key: str, task_id: str) -> dict[str, Any]:
    headers = {"x-api-key": api_key}
    return await call_json_with_retry(f"{LEGNEXT_BASE}/task/{task_id}", "GET", None, headers, timeout=60)


async def legnext_wait_result(api_key: str, task_id: str, timeout_sec: int = 180) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while True:
        data = await legnext_get_task(api_key, task_id)
        status = str(data.get("status") or data.get("state") or "").lower()
        if status in {"succeeded", "success", "completed", "done"}:
            return data
        if status in {"failed", "error", "canceled"}:
            return data
        if asyncio.get_event_loop().time() >= deadline:
            return data
        await asyncio.sleep(2.5)


async def execute_chat_turn(
    bot_data: dict[str, Any],
    state: dict[str, Any],
    user_text: str,
) -> tuple[ProviderResult | None, str | None, str | None, ModelEntry | None]:
    catalog: list[ModelEntry] = bot_data.get("models_catalog", [])
    catalog_by_key: dict[str, ModelEntry] = bot_data.get("catalog_by_key", {})
    selected_key = state.get("selected_model_key")
    entry = catalog_by_key.get(selected_key) if selected_key else None
    if not entry or entry.model_type != MODEL_TYPE_CHAT:
        return None, None, "Сначала выбери чат-модель.", entry

    history = state["history"]
    providers = entry.providers or [entry.provider_id]
    last_warning_local: str | None = None

    for provider_id in providers:
        client = bot_data["providers"].get(provider_id)
        if not client:
            continue
        models = bot_data.get(current_models_key(provider_id), [])
        if entry.model_id not in models:
            continue
        attempt = await try_with_fallbacks(
            client,
            provider_id,
            entry.model_id,
            history,
            models,
            allow_model_fallback=False,
        )
        if attempt.answer:
            if answer_looks_broken(user_text, attempt.answer):
                for alt_entry in preferred_fallback_chat_entries(catalog, state.get("selected_model_key")):
                    alt_providers = alt_entry.providers or [alt_entry.provider_id]
                    for alt_provider_id in alt_providers:
                        alt_client = bot_data["providers"].get(alt_provider_id)
                        if not alt_client:
                            continue
                        alt_models = bot_data.get(current_models_key(alt_provider_id), [])
                        if alt_entry.model_id not in alt_models:
                            continue
                        try:
                            alt_answer, alt_usage = await alt_client.chat_with_usage(alt_entry.model_id, history)
                            if answer_looks_broken(user_text, alt_answer):
                                continue
                            state["selected_model_key"] = alt_entry.key
                            return (
                                ProviderResult(
                                    alt_entry.model_id,
                                    alt_answer,
                                    "Бот отбросил некорректный ответ и автоматически переключил запрос на другую модель.",
                                    alt_usage,
                                ),
                                alt_provider_id,
                                last_warning_local,
                                alt_entry,
                            )
                        except Exception:
                            continue
                last_warning_local = "Модель вернула некорректный ответ, попробуй другую модель."
                continue
            return attempt, provider_id, last_warning_local, entry
        if attempt.warning:
            last_warning_local = attempt.warning
    return None, None, last_warning_local, entry


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    await refresh_all_cmd(update, context, notify_only=True)
    if text == BTN_ROLE:
        await show_roles_picker(update.message, context, page=0)
        return
    if text == BTN_PICK_MODEL:
        await show_models(update.message, context, page=0)
        return
    if text == BTN_CLEAR:
        await clear_cmd(update, context)
        return
    if text == BTN_PROFILE:
        await profile_cmd(update, context)
        return
    if text == BTN_LIMITS:
        await limits_cmd(update, context)
        return
    if text == BTN_HELP:
        await help_cmd(update, context)
        return

    state = ensure_state(context)
    catalog_by_key: dict[str, ModelEntry] = context.bot_data.get("catalog_by_key", {})

    selected_key = state.get("selected_model_key")
    if not selected_key:
        await update.message.reply_text(
            "Сначала выбери модель.",
            reply_markup=menu_keyboard(),
        )
        return
    entry = catalog_by_key.get(selected_key)
    if not entry:
        await update.message.reply_text(
            "Текущая модель не найдена. Выбери другую.",
            reply_markup=menu_keyboard(),
        )
        return

    if entry.model_type == MODEL_TYPE_CHAT:
        history = state["history"]
        history.append({"role": "user", "content": text})
        if len(history) > MAX_HISTORY_MESSAGES:
            history = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1) :]
            state["history"] = history

        await update.message.chat.send_action("typing")
        status_msg, stop_event, progress_task = await start_progress_message(update.message, context.bot_data)
        started_at = time.perf_counter()
        semaphore: asyncio.Semaphore = context.bot_data["request_semaphore"]
        try:
            async with semaphore:
                result, used_provider, warning, actual_entry = await execute_chat_turn(context.bot_data, state, text)
        finally:
            await finish_progress_message(status_msg, stop_event, progress_task, context.bot_data, ok=True)

        if not result or not result.answer:
            await update.message.reply_text(
                "Ответ не получен. Попробуй еще раз.",
                reply_markup=menu_keyboard(),
            )
            return

        answer = result.answer
        elapsed_sec = time.perf_counter() - started_at
        usage_provider = used_provider or entry.provider_id
        usage_model = actual_entry.model_id if actual_entry else (result.model_used or entry.model_id)
        update_usage_stats(state, usage_provider, usage_model, result.usage, text, answer)

        if warning:
            await update.message.reply_text(warning, reply_markup=menu_keyboard())
        if result.warning:
            await update.message.reply_text(result.warning, reply_markup=menu_keyboard())

        await reply_answer(
            update.message,
            answer,
            usage_provider,
            usage_model,
            usage=result.usage,
            elapsed_sec=elapsed_sec,
            bot_data=context.bot_data,
            reply_markup=menu_keyboard(),
        )
        return

    if entry.model_type == MODEL_TYPE_IMAGE:
        await handle_image_request(update, context, entry, text)
        return
    await handle_video_request(update, context, entry, text)

async def handle_image_request(update: Update, context: ContextTypes.DEFAULT_TYPE, entry: ModelEntry, prompt: str) -> None:
    provider_id = entry.provider_id
    if provider_id == PROVIDER_POLLINATIONS:
        api_key = context.bot_data.get("pollinations_api_key")
        url = pollinations_media_url(prompt, entry.model_id, "image", api_key)
        await update.message.reply_photo(photo=url, caption=f"{entry.model_id}")
        return

    if provider_id == PROVIDER_LEGNEXT:
        api_key = context.bot_data.get("legnext_api_key")
        if not api_key:
            await update.message.reply_text("LegNext API ???? ?? ?????.", reply_markup=menu_keyboard())
            return
        task_id = await legnext_create_task(api_key, "image", entry.model_id, prompt)
        result = await legnext_wait_result(api_key, task_id)
        url = result.get("output_url") or result.get("url") or result.get("result", {}).get("url")
        if not url:
            await update.message.reply_text("?? ??????? ???????? ???????????.", reply_markup=menu_keyboard())
            return
        await update.message.reply_photo(photo=url, caption=f"{entry.model_id}")
        return

    await update.message.reply_text("????????? ?? ???????????? ????????? ???????????.", reply_markup=menu_keyboard())


async def handle_video_request(update: Update, context: ContextTypes.DEFAULT_TYPE, entry: ModelEntry, prompt: str) -> None:
    provider_id = entry.provider_id
    if provider_id == PROVIDER_POLLINATIONS:
        api_key = context.bot_data.get("pollinations_api_key")
        url = pollinations_media_url(prompt, entry.model_id, "video", api_key)
        await update.message.reply_text(url, reply_markup=menu_keyboard())
        return

    if provider_id == PROVIDER_LEGNEXT:
        api_key = context.bot_data.get("legnext_api_key")
        if not api_key:
            await update.message.reply_text("LegNext API ???? ?? ?????.", reply_markup=menu_keyboard())
            return
        task_id = await legnext_create_task(api_key, "video", entry.model_id, prompt)
        result = await legnext_wait_result(api_key, task_id)
        url = result.get("output_url") or result.get("url") or result.get("result", {}).get("url")
        if not url:
            await update.message.reply_text("?? ??????? ???????? ?????.", reply_markup=menu_keyboard())
            return
        await update.message.reply_text(url, reply_markup=menu_keyboard())
        return

    await update.message.reply_text("????????? ?? ???????????? ????????? ?????.", reply_markup=menu_keyboard())
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if update and isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"?????????? ?????? ????: {err}")


def build_web_catalog(bot_data: dict[str, Any]) -> list[dict[str, str]]:
    catalog: list[ModelEntry] = bot_data.get("models_catalog", [])
    if not catalog:
        chat_providers: set[str] = bot_data.get("chat_providers", set())
        fallback_catalog: list[ModelEntry] = []
        for provider_id in sorted(chat_providers):
            for model_id in STATIC_CHAT_MODELS.get(provider_id, []):
                fallback_catalog.append(
                    ModelEntry(
                        key=model_key(provider_id, model_id, MODEL_TYPE_CHAT),
                        provider_id=provider_id,
                        model_id=model_id,
                        model_type=MODEL_TYPE_CHAT,
                        providers=[provider_id],
                    )
                )
        for model_id in bot_data.get("pollinations_image_models", []):
            fallback_catalog.append(
                ModelEntry(
                    key=model_key(PROVIDER_POLLINATIONS, model_id, MODEL_TYPE_IMAGE),
                    provider_id=PROVIDER_POLLINATIONS,
                    model_id=model_id,
                    model_type=MODEL_TYPE_IMAGE,
                )
            )
        for model_id in bot_data.get("pollinations_video_models", []):
            fallback_catalog.append(
                ModelEntry(
                    key=model_key(PROVIDER_POLLINATIONS, model_id, MODEL_TYPE_VIDEO),
                    provider_id=PROVIDER_POLLINATIONS,
                    model_id=model_id,
                    model_type=MODEL_TYPE_VIDEO,
                )
            )
        catalog = fallback_catalog
    items: list[dict[str, str]] = []
    for entry in catalog:
        items.append(
            {
                "key": entry.key,
                "providerId": entry.provider_id,
                "providerTitle": provider_title(entry.provider_id),
                "modelId": entry.model_id,
                "type": entry.model_type,
                "label": model_entry_label(entry),
            }
        )
    return items


def build_web_config(bot_data: dict[str, Any]) -> dict[str, Any]:
    catalog: list[ModelEntry] = bot_data.get("models_catalog", [])
    if not catalog:
        catalog = []
        for provider_id in sorted(bot_data.get("chat_providers", set())):
            for model_id in STATIC_CHAT_MODELS.get(provider_id, []):
                catalog.append(
                    ModelEntry(
                        key=model_key(provider_id, model_id, MODEL_TYPE_CHAT),
                        provider_id=provider_id,
                        model_id=model_id,
                        model_type=MODEL_TYPE_CHAT,
                        providers=[provider_id],
                    )
                )
    preferred_key = preferred_chat_model_key(catalog)
    key_slot_active = max(1, int(os.getenv("KEY_SLOT_ACTIVE", "1") or "1"))
    key_slots_total = max(key_slot_active, int(os.getenv("KEY_SLOTS_TOTAL", "20") or "20"))
    return {
        "siteName": SITE_NAME,
        "siteUrl": SITE_URL,
        "defaultRoleId": "general",
        "defaultModelKey": preferred_key,
        "keySlotActive": key_slot_active,
        "keySlotsTotal": key_slots_total,
        "roles": [{"id": role.role_id, "title": WEB_ROLE_TITLES.get(role.role_id, role.role_id)} for role in ROLE_SPECS],
        "models": build_web_catalog(bot_data),
        "quickReplies": [
            "Напиши продающее описание услуги",
            "Разбери ошибку в коде",
            "Составь план ремонта",
            "Сделай оффер для Avito",
        ],
    }


def json_for_html(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


async def web_chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    bot_data = WEB_RUNTIME["bot_data"]
    user_text = str(payload.get("message") or "").strip()
    if not user_text:
        return {"ok": False, "error": "Пустое сообщение."}

    catalog: list[ModelEntry] = bot_data.get("models_catalog", [])
    preferred_key = preferred_chat_model_key(catalog)
    role_id = str(payload.get("roleId") or "general")
    if role_id not in ROLE_MAP:
        role_id = "general"

    history = initial_history(role_id)
    incoming_history = payload.get("history") or []
    if isinstance(incoming_history, list):
        for item in incoming_history[-20:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                history.append({"role": role, "content": content})

    state = {
        "selected_model_key": str(payload.get("selectedModelKey") or preferred_key or ""),
        "selected_role_id": role_id,
        "history": history,
    }
    state["history"].append({"role": "user", "content": user_text})
    if len(state["history"]) > MAX_HISTORY_MESSAGES:
        state["history"] = [state["history"][0]] + state["history"][-(MAX_HISTORY_MESSAGES - 1) :]

    started_at = time.perf_counter()
    result, used_provider, warning, actual_entry = await execute_chat_turn(bot_data, state, user_text)
    if not result or not result.answer:
        return {"ok": False, "error": warning or "Ответ не получен."}

    answer = result.answer
    elapsed_sec = time.perf_counter() - started_at
    state["history"].append({"role": "assistant", "content": answer})
    resolved_entry = actual_entry or bot_data.get("catalog_by_key", {}).get(state.get("selected_model_key"))
    provider_id = used_provider or (resolved_entry.provider_id if resolved_entry else "")
    model_id = result.model_used or (resolved_entry.model_id if resolved_entry else "")
    return {
        "ok": True,
        "answer": answer,
        "warning": warning,
        "providerId": provider_id,
        "providerTitle": provider_title(provider_id) if provider_id else "AI",
        "modelId": model_id,
        "usage": result.usage or {},
        "elapsedSec": elapsed_sec,
        "metadataLines": format_metadata_lines(provider_id, model_id, result.usage, elapsed_sec),
        "selectedModelKey": state.get("selected_model_key"),
        "roleId": role_id,
        "roleTitle": role_title(role_id),
        "history": state["history"][1:],
    }


def web_ui_html() -> str:
    config_json = json_for_html(build_web_config(WEB_RUNTIME.get("bot_data", {})))
    html_template = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Bot Cyber UI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/base16/synth-midnight-terminal-dark.min.css">
  <style>
    :root {
      --bg: #050505;
      --bg-soft: rgba(10, 10, 14, 0.92);
      --glass: rgba(20, 20, 20, 0.8);
      --glass-2: rgba(8, 8, 8, 0.72);
      --cyan: #00f2ff;
      --pink: #ff00ff;
      --text: #d8fbff;
      --muted: #7fd9e0;
      --line: rgba(0, 242, 255, 0.22);
      --danger: #ff4fd8;
      --radius: 20px;
      --shadow-cyan: 0 0 15px rgba(0, 242, 255, 0.55);
      --shadow-pink: 0 0 15px rgba(255, 0, 255, 0.45);
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      font-family: "JetBrains Mono", "Courier New", monospace;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(0,242,255,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(255,0,255,0.10), transparent 24%),
        radial-gradient(circle at center, rgba(0,242,255,0.06), transparent 38%),
        #050505;
      overflow-x: hidden;
      overflow-y: auto;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255,255,255,0.03) 0,
        rgba(255,255,255,0.03) 1px,
        transparent 2px,
        transparent 4px
      );
      opacity: 0.08;
      mix-blend-mode: screen;
    }
    body::after {
      content: "";
      position: fixed;
      left: 0;
      right: 0;
      height: 140px;
      top: -160px;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(0,242,255,0), rgba(0,242,255,0.12), rgba(255,0,255,0));
      animation: scanline 9s linear infinite;
      filter: blur(4px);
      opacity: 0.7;
    }
    @keyframes scanline {
      from { transform: translateY(0); }
      to { transform: translateY(calc(100vh + 220px)); }
    }
    @keyframes breathe {
      0%, 100% { box-shadow: 0 0 8px rgba(0,242,255,0.35), 0 0 18px rgba(0,242,255,0.18); opacity: 0.72; }
      50% { box-shadow: 0 0 16px rgba(0,242,255,0.78), 0 0 28px rgba(255,0,255,0.22); opacity: 1; }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(18px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatGlow {
      0%, 100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.35; }
      50% { transform: translate3d(0, -8px, 0) scale(1.03); opacity: 0.6; }
    }
    @keyframes borderPulse {
      0%, 100% { box-shadow: 0 0 12px rgba(0,242,255,0.18); }
      50% { box-shadow: 0 0 22px rgba(255,0,255,0.18), 0 0 18px rgba(0,242,255,0.28); }
    }
    .app {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      width: min(1380px, calc(100vw - 24px));
      min-height: calc(100dvh - 24px);
      margin: 12px auto;
    }
    .app::before,
    .app::after {
      content: "";
      position: fixed;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      filter: blur(70px);
      pointer-events: none;
      z-index: -1;
      animation: floatGlow 7s ease-in-out infinite;
    }
    .app::before {
      top: 4vh;
      left: 2vw;
      background: rgba(0,242,255,0.12);
    }
    .app::after {
      right: 3vw;
      bottom: 10vh;
      background: rgba(255,0,255,0.1);
      animation-delay: -3.5s;
    }
    .panel {
      background: var(--glass);
      border: 1px solid var(--line);
      border-radius: 24px;
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow-cyan);
      position: relative;
      overflow: hidden;
      animation: borderPulse 5.5s ease-in-out infinite;
    }
    .panel::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(135deg, rgba(0,242,255,0.05), transparent 45%, rgba(255,0,255,0.03));
    }
    .sidebar {
      padding: 20px;
      display: grid;
      grid-template-rows: auto auto auto 1fr;
      gap: 14px;
    }
    .brand {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.08em;
      color: var(--cyan);
      text-shadow: 0 0 10px rgba(0,242,255,0.55);
    }
    .brand small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      line-height: 1.5;
    }
    .status-chip {
      border: 1px solid rgba(255,0,255,0.45);
      color: #ffd3ff;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(255,0,255,0.08);
      box-shadow: var(--shadow-pink);
      white-space: nowrap;
      font-size: 12px;
    }
    .key-strip {
      padding: 14px 16px;
      background: var(--glass-2);
      border: 1px solid rgba(0,242,255,0.18);
      border-radius: 18px;
      box-shadow: inset 0 0 0 1px rgba(255,0,255,0.08), var(--shadow-cyan);
    }
    .key-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .key-bar {
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      overflow: hidden;
      border: 1px solid rgba(0,242,255,0.22);
      box-shadow: inset 0 0 16px rgba(0,242,255,0.08);
    }
    .key-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--pink));
      box-shadow: 0 0 15px #00f2ff;
      transition: width 0.35s ease;
    }
    .field {
      display: grid;
      gap: 7px;
    }
    .field label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    select, textarea {
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(0,242,255,0.26);
      background: rgba(7, 7, 10, 0.78);
      color: var(--text);
      font: inherit;
      padding: 14px 15px;
      outline: none;
      box-shadow: inset 0 0 14px rgba(0,242,255,0.04);
    }
    select:focus, textarea:focus {
      box-shadow: 0 0 15px #00f2ff, inset 0 0 14px rgba(0,242,255,0.07);
      border-color: rgba(0,242,255,0.7);
    }
    textarea {
      min-height: 96px;
      resize: none;
    }
    .quick-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-content: flex-start;
    }
    .quick-btn, .ghost-btn, .send-btn {
      font: inherit;
      border-radius: 14px;
      padding: 10px 12px;
      cursor: pointer;
      transition: 0.22s ease;
    }
    .quick-btn, .ghost-btn {
      background: rgba(0,0,0,0.28);
      color: var(--text);
      border: 1px solid rgba(0,242,255,0.28);
    }
    .quick-btn:hover, .ghost-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 0 15px #00f2ff;
      border-color: rgba(0,242,255,0.72);
    }
    .chat-shell {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      min-height: 0;
    }
    .topbar {
      padding: 18px 22px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      border-bottom: 1px solid rgba(0,242,255,0.12);
    }
    .topbar h2 {
      margin: 0;
      color: var(--cyan);
      font-size: 22px;
      text-shadow: 0 0 10px rgba(0,242,255,0.35);
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .api-live {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(0,242,255,0.35);
      background: rgba(0,0,0,0.32);
      box-shadow: var(--shadow-cyan);
      animation: breathe 2.2s ease-in-out infinite;
      color: var(--text);
      font-size: 12px;
    }
    .api-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 15px #00f2ff;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px 22px 16px;
      border-bottom: 1px solid rgba(0,242,255,0.08);
    }
    .chat-log {
      overflow: auto;
      padding: 20px 22px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-height: 0;
    }
    .bubble {
      position: relative;
      max-width: min(860px, 88%);
      padding: 16px 18px;
      border-radius: 20px;
      border: 1px solid rgba(0,242,255,0.22);
      background: rgba(20, 20, 20, 0.8);
      backdrop-filter: blur(10px);
      box-shadow: 0 0 15px rgba(0,242,255,0.18);
      animation: rise 0.28s ease;
      line-height: 1.65;
      overflow-wrap: anywhere;
    }
    .bubble::after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      background: linear-gradient(135deg, rgba(0,242,255,0.08), transparent 42%, rgba(255,0,255,0.06));
      opacity: 0.7;
    }
    .bubble.user {
      align-self: flex-end;
      border-color: rgba(0,242,255,0.38);
      box-shadow: 0 0 15px rgba(0,242,255,0.22);
    }
    .bubble.bot {
      align-self: flex-start;
      border-color: rgba(255,0,255,0.18);
      box-shadow: 0 0 15px rgba(0,242,255,0.12);
    }
    .bubble.system {
      align-self: center;
      max-width: min(760px, 100%);
      border-color: rgba(255,0,255,0.34);
      box-shadow: var(--shadow-pink);
    }
    .bubble-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .bubble-role {
      color: var(--cyan);
      text-shadow: 0 0 6px rgba(0,242,255,0.35);
    }
    .bubble-api {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(0,242,255,0.24);
      background: rgba(0,0,0,0.28);
      box-shadow: 0 0 10px rgba(0,242,255,0.18);
    }
    .bubble pre {
      background: rgba(0, 0, 0, 0.74);
      border: 1px solid rgba(0,242,255,0.22);
      border-radius: 14px;
      padding: 14px;
      overflow: auto;
      box-shadow: inset 0 0 18px rgba(0,242,255,0.06);
    }
    .bubble code {
      font-family: "JetBrains Mono", "Courier New", monospace;
    }
    .code-wrap {
      margin: 12px 0 0;
      border: 1px solid rgba(0,242,255,0.22);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: inset 0 0 18px rgba(0,242,255,0.05);
    }
    .code-label {
      padding: 8px 12px;
      font-size: 12px;
      color: var(--muted);
      border-bottom: 1px solid rgba(0,242,255,0.14);
      background: rgba(0,242,255,0.04);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .bubble ul {
      margin: 10px 0 0;
      padding-left: 20px;
    }
    .bubble p {
      margin: 0;
    }
    .composer {
      padding: 16px 22px 22px;
      border-top: 1px solid rgba(0,242,255,0.12);
      display: grid;
      gap: 12px;
      background: rgba(5, 5, 5, 0.45);
    }
    .status-line {
      color: var(--muted);
      min-height: 18px;
      font-size: 13px;
    }
    .composer-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }
    .send-btn {
      min-width: 168px;
      height: 54px;
      background: transparent;
      color: var(--cyan);
      border: 1px solid rgba(0,242,255,0.48);
      box-shadow: var(--shadow-cyan);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .send-btn:hover {
      color: white;
      transform: translateY(-1px);
      background: rgba(0,242,255,0.08);
      box-shadow: 0 0 18px #00f2ff;
    }
    .send-btn:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    @media (max-width: 1100px) {
      .app {
        grid-template-columns: 1fr;
        min-height: calc(100vh - 24px);
      }
      .chat-shell {
        min-height: 70vh;
      }
    }
    @media (max-width: 720px) {
      body {
        padding-bottom: 0;
      }
      .app {
        width: calc(100vw - 14px);
        margin: 7px auto;
        gap: 12px;
        min-height: calc(100dvh - 14px);
      }
      .chat-shell {
        order: 1;
        min-height: calc(100dvh - 14px);
      }
      .sidebar {
        order: 2;
      }
      .topbar, .toolbar, .chat-log, .composer, .sidebar {
        padding-left: 14px;
        padding-right: 14px;
      }
      .topbar {
        position: sticky;
        top: 0;
        z-index: 4;
        background: rgba(5, 5, 5, 0.86);
        backdrop-filter: blur(10px);
      }
      .toolbar {
        overflow-x: auto;
        flex-wrap: nowrap;
        scrollbar-width: none;
      }
      .toolbar::-webkit-scrollbar {
        display: none;
      }
      .chat-log {
        min-height: 42dvh;
        max-height: 52dvh;
        padding-bottom: 120px;
      }
      .composer {
        position: sticky;
        bottom: 0;
        z-index: 5;
        background: rgba(5, 5, 5, 0.94);
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(0,242,255,0.2);
      }
      .composer-row {
        grid-template-columns: 1fr;
      }
      textarea {
        min-height: 82px;
      }
      .send-btn {
        width: 100%;
      }
      .bubble {
        max-width: 100%;
      }
      .quick-list {
        max-height: 160px;
        overflow: auto;
        padding-right: 4px;
      }
    }
    @media (max-width: 520px) {
      .brand {
        flex-direction: column;
        align-items: flex-start;
      }
      .brand h1 {
        font-size: 24px;
      }
      .status-chip,
      .api-live,
      .ghost-btn,
      .quick-btn {
        font-size: 11px;
      }
      .bubble {
        padding: 14px 14px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel sidebar">
      <div class="brand">
        <div>
          <h1>AI BOT</h1>
          <small>Cyberpunk dark neon console for multi-provider chat, code and media workflows.</small>
        </div>
        <div class="status-chip">SYNTH MODE</div>
      </div>

      <div class="key-strip">
        <div class="key-head">
          <span id="keyLabel">Key #1/20</span>
          <span id="keyPercent">5%</span>
        </div>
        <div class="key-bar"><div class="key-bar-fill" id="keyBarFill"></div></div>
      </div>

      <div class="field">
        <label for="roleSelect">ROLE</label>
        <select id="roleSelect"></select>
      </div>

      <div class="field">
        <label for="modelSelect">MODEL</label>
        <select id="modelSelect"></select>
      </div>

      <div class="field">
        <label>QUICK REPLIES</label>
        <div class="quick-list" id="quickReplies"></div>
      </div>
    </aside>

    <main class="panel chat-shell">
      <div class="topbar">
        <div>
          <h2>Cyber Console</h2>
          <div class="subtitle">Code highlighting, provider telemetry, neon response cards.</div>
        </div>
        <div class="api-live" id="sessionBadge">
          <span class="api-dot"></span>
          <span>API READY</span>
        </div>
      </div>

      <div class="toolbar">
        <button class="ghost-btn" id="clearHistoryBtn">Clear history</button>
        <button class="ghost-btn" id="scrollBottomBtn">Scroll to bottom</button>
      </div>

      <div class="chat-log" id="chatLog"></div>

      <div class="composer">
        <div class="status-line" id="statusLine">Stand by. Select role, select model, send task.</div>
        <div class="composer-row">
          <textarea id="promptInput" placeholder="Task: code, diagnostics, ad copy, analysis, plan, listing..."></textarea>
          <button class="send-btn" id="sendBtn">Send signal</button>
        </div>
      </div>
    </main>
  </div>

  <script>
    const EMBEDDED_CONFIG = __CONFIG_JSON__;
    const state = {
      config: null,
      history: JSON.parse(localStorage.getItem('webHistory') || '[]')
    };

    const roleSelect = document.getElementById('roleSelect');
    const modelSelect = document.getElementById('modelSelect');
    const promptInput = document.getElementById('promptInput');
    const sendBtn = document.getElementById('sendBtn');
    const quickReplies = document.getElementById('quickReplies');
    const chatLog = document.getElementById('chatLog');
    const statusLine = document.getElementById('statusLine');
    const sessionBadge = document.getElementById('sessionBadge');
    const keyLabel = document.getElementById('keyLabel');
    const keyPercent = document.getElementById('keyPercent');
    const keyBarFill = document.getElementById('keyBarFill');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const scrollBottomBtn = document.getElementById('scrollBottomBtn');

    function renderMarkdownLite(input) {
      const source = String(input || '');
      const codeBlocks = [];
      const protectedText = source.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push({
          lang: (lang || 'code').trim(),
          code: escapeHtml(code.trim())
        });
        return token;
      });

      const paragraphs = protectedText.split(/\n\n+/).map(chunk => chunk.trim()).filter(Boolean);
      const rendered = paragraphs.map(chunk => {
        const lines = chunk.split('\n');
        const isList = lines.every(line => /^(-|\*)\s+/.test(line.trim()));
        if (isList) {
          const items = lines.map(line => `<li>${escapeHtml(line.replace(/^(-|\*)\s+/, ''))}</li>`).join('');
          return `<ul>${items}</ul>`;
        }
        let html = escapeHtml(chunk)
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\n/g, '<br>');
        return `<p>${html}</p>`;
      }).join('');

      let finalHtml = rendered;
      codeBlocks.forEach((block, idx) => {
        const markup = `<div class="code-wrap"><div class="code-label">${block.lang}</div><pre><code>${block.code}</code></pre></div>`;
        finalHtml = finalHtml.replace(`@@CODEBLOCK_${idx}@@`, markup);
      });
      return finalHtml;
    }

    function saveHistory() {
      localStorage.setItem('webHistory', JSON.stringify(state.history.slice(-20)));
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function roleLabel(role) {
      if (role === 'user') return 'USER';
      if (role === 'system') return 'SYSTEM';
      return 'BOT';
    }

    function bubbleHtml(item) {
      const kind = item.kind || (item.role === 'user' ? 'user' : 'bot');
      const body = item.role === 'user'
        ? escapeHtml(item.content).replace(/\n/g, '<br>')
        : renderMarkdownLite(item.content || '');
      const providerMeta = item.providerTitle
        ? `<span class="bubble-api">API ${escapeHtml(item.providerTitle)} | ${escapeHtml(item.modelId || '')}</span>`
        : '';
      return `
        <article class="bubble ${kind}">
          <div class="bubble-head">
            <span class="bubble-role">${roleLabel(item.role)}</span>
            ${providerMeta}
          </div>
          <div class="bubble-body">${body}</div>
        </article>
      `;
    }

    function renderHistory() {
      chatLog.innerHTML = state.history.map(bubbleHtml).join('');
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function setKeyStatus(active, total) {
      const safeActive = Math.max(1, Number(active || 1));
      const safeTotal = Math.max(safeActive, Number(total || 20));
      const percent = Math.max(1, Math.min(100, Math.round((safeActive / safeTotal) * 100)));
      keyLabel.textContent = `Key #${safeActive}/${safeTotal}`;
      keyPercent.textContent = `${percent}%`;
      keyBarFill.style.width = `${percent}%`;
    }

    function fillConfig(config) {
      state.config = config;
      roleSelect.innerHTML = config.roles
        .map(role => `<option value="${role.id}">${escapeHtml(role.title)}</option>`)
        .join('');
      modelSelect.innerHTML = config.models
        .filter(model => model.type === 'chat')
        .map(model => `<option value="${model.key}">${escapeHtml(model.label)}</option>`)
        .join('');
      roleSelect.value = config.defaultRoleId || 'general';
      if (config.defaultModelKey) {
        modelSelect.value = config.defaultModelKey;
      }
      setKeyStatus(config.keySlotActive, config.keySlotsTotal);
      quickReplies.innerHTML = '';
      config.quickReplies.forEach(text => {
        const btn = document.createElement('button');
        btn.className = 'quick-btn';
        btn.textContent = text;
        btn.addEventListener('click', () => {
          promptInput.value = text;
          promptInput.focus();
        });
        quickReplies.appendChild(btn);
      });
    }

    function pushSystemMessage(content) {
      state.history.push({
        role: 'system',
        kind: 'system',
        content,
        providerTitle: 'SYSTEM',
        modelId: 'UI'
      });
      saveHistory();
      renderHistory();
    }

    async function boot() {
      try {
        const config = EMBEDDED_CONFIG;
        fillConfig(config);
        if (!config.models || !config.models.length) {
          pushSystemMessage('**No models loaded.** Backend returned an empty model catalog. Check provider keys and server logs.');
          statusLine.textContent = 'No models available from backend.';
          return;
        }
        if (!state.history.length) {
          pushSystemMessage('**System online.** Select role, select model, then send your task.\\n\\nSupports code, neat lists, metadata cards and provider telemetry.');
        } else {
          renderHistory();
        }
      } catch (err) {
        const message = err && err.message ? err.message : 'Config load failed';
        pushSystemMessage(`**Boot error:** ${message}`);
        statusLine.textContent = `Boot error: ${message}`;
      }
    }

    async function sendMessage() {
      const message = promptInput.value.trim();
      if (!message) return;

      const payload = {
        message,
        roleId: roleSelect.value,
        selectedModelKey: modelSelect.value,
        history: state.history
          .filter(item => item.role === 'user' || item.role === 'assistant')
          .map(item => ({ role: item.role, content: item.content }))
      };

      state.history.push({ role: 'user', kind: 'user', content: message });
      promptInput.value = '';
      saveHistory();
      renderHistory();

      statusLine.textContent = 'Generating... [----------] 0%';
      sessionBadge.innerHTML = '<span class="api-dot"></span><span>API ACTIVE</span>';
      sendBtn.disabled = true;

      let progress = 0;
      const progressTimer = setInterval(() => {
        progress = Math.min(progress + 11, 94);
        const filled = Math.max(0, Math.min(10, Math.round(progress / 10)));
        statusLine.textContent = `Generating... [${'#'.repeat(filled)}${'-'.repeat(10 - filled)}] ${progress}%`;
      }, 650);

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        clearInterval(progressTimer);
        if (!data.ok) {
          throw new Error(data.error || 'Response error');
        }

        const metadataBlock = data.metadataLines && data.metadataLines.length
          ? `\n\n\`\`\`text\n${data.metadataLines.join('\n')}\n\`\`\``
          : '';

        state.history.push({
          role: 'assistant',
          kind: 'bot',
          content: `${data.answer}${metadataBlock}`,
          providerTitle: data.providerTitle,
          modelId: data.modelId
        });
        saveHistory();
        renderHistory();

        sessionBadge.innerHTML = `<span class="api-dot"></span><span>API ${escapeHtml(data.providerTitle)} | ${escapeHtml(data.modelId)}</span>`;
        statusLine.textContent = data.warning || `Done. [##########] 100% | ${data.providerTitle} / ${data.modelId}`;
      } catch (err) {
        clearInterval(progressTimer);
        const message = err && err.message ? err.message : 'Unknown error';
        state.history.push({
          role: 'system',
          kind: 'system',
          content: `**Error:** ${message}`,
          providerTitle: 'SYSTEM',
          modelId: 'ERR'
        });
        saveHistory();
        renderHistory();
        statusLine.textContent = `Error: ${message}`;
      } finally {
        sendBtn.disabled = false;
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    clearHistoryBtn.addEventListener('click', () => {
      state.history = [];
      localStorage.removeItem('webHistory');
      pushSystemMessage('**History cleared.** Console reset complete.');
    });
    scrollBottomBtn.addEventListener('click', () => {
      chatLog.scrollTop = chatLog.scrollHeight;
    });
    promptInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        sendMessage();
      }
    });

    boot();
  </script>
</body>
</html>"""
    return html_template.replace("__CONFIG_JSON__", config_json)


def validate_env() -> tuple[str, str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or None
    hf_key = os.getenv("HF_API_KEY", "").strip() or None
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip() or None
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY", "").strip() or None
    legnext_key = os.getenv("LEGNEXT_API_KEY", "").strip() or None
    pollinations_key = os.getenv("POLLINATIONS_API_KEY", "").strip() or None
    pollinations_enabled = bool(os.getenv("POLLINATIONS_ENABLE", "").strip())

    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    if not any([or_key, groq_key, hf_key, mistral_key, siliconflow_key, legnext_key, pollinations_key, pollinations_enabled]):
        raise RuntimeError(
            "Set at least one key: OPENROUTER_API_KEY, GROQ_API_KEY, HF_API_KEY, MISTRAL_API_KEY, SILICONFLOW_API_KEY, POLLINATIONS_API_KEY, or LEGNEXT_API_KEY"
        )

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

    if siliconflow_key:
        try:
            siliconflow_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("SILICONFLOW_API_KEY must be ASCII") from e

    if legnext_key:
        try:
            legnext_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("LEGNEXT_API_KEY must be ASCII") from e

    if pollinations_key:
        try:
            pollinations_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError("POLLINATIONS_API_KEY must be ASCII") from e

    return tg_token, or_key, groq_key, hf_key, mistral_key, siliconflow_key, legnext_key, pollinations_key
def main() -> None:
    tg_token, or_key, groq_key, hf_key, mistral_key, siliconflow_key, legnext_key, pollinations_key = validate_env()
    token_limits_env = os.getenv("MODEL_TOKEN_LIMITS")
    request_limits_env = os.getenv("MODEL_REQUEST_LIMITS")
    providers: dict[str, BaseClient] = {}

    if or_key:
        providers[PROVIDER_OPENROUTER] = OpenRouterClient(or_key)
    if groq_key:
        providers[PROVIDER_GROQ] = GroqClient(groq_key)
    if hf_key:
        providers[PROVIDER_HF] = HuggingFaceClient(hf_key)
    if mistral_key:
        providers[PROVIDER_MISTRAL] = MistralClient(mistral_key)
    if siliconflow_key:
        providers[PROVIDER_SILICONFLOW] = SiliconFlowClient(siliconflow_key)

    pollinations_enabled = bool(pollinations_key or os.getenv("POLLINATIONS_ENABLE", "").strip())
    pollinations_text_models = parse_csv_env_list(
        os.getenv("POLLINATIONS_TEXT_MODELS"),
        DEFAULT_POLLINATIONS_TEXT_MODELS,
    )
    if pollinations_enabled:
        providers[PROVIDER_POLLINATIONS] = PollinationsTextClient(pollinations_key, pollinations_text_models)

    app = Application.builder().token(tg_token).concurrent_updates(False).build()
    app.bot_data["providers"] = providers
    app.bot_data["chat_providers"] = set(providers.keys())
    available_providers = set(providers.keys())
    if legnext_key:
        available_providers.add(PROVIDER_LEGNEXT)
    if pollinations_enabled:
        available_providers.add(PROVIDER_POLLINATIONS)
    app.bot_data["available_providers"] = available_providers
    app.bot_data["global_agents"] = []
    app.bot_data["models_catalog"] = []
    app.bot_data["catalog_by_key"] = {}
    app.bot_data[current_models_key(PROVIDER_OPENROUTER)] = []
    app.bot_data[current_models_key(PROVIDER_GROQ)] = []
    app.bot_data[current_models_key(PROVIDER_HF)] = []
    app.bot_data[current_models_key(PROVIDER_MISTRAL)] = []
    app.bot_data[current_models_key(PROVIDER_SILICONFLOW)] = []
    app.bot_data[current_models_key(PROVIDER_POLLINATIONS)] = []
    app.bot_data["legnext_api_key"] = legnext_key
    app.bot_data["pollinations_api_key"] = pollinations_key

    app.bot_data["legnext_image_models"] = parse_csv_env_list(
        os.getenv("LEGNEXT_IMAGE_MODELS"),
        DEFAULT_LEGNEXT_IMAGE_MODELS if legnext_key else [],
    )
    app.bot_data["legnext_video_models"] = parse_csv_env_list(
        os.getenv("LEGNEXT_VIDEO_MODELS"),
        DEFAULT_LEGNEXT_VIDEO_MODELS if legnext_key else [],
    )
    app.bot_data["pollinations_image_models"] = parse_csv_env_list(
        os.getenv("POLLINATIONS_IMAGE_MODELS"),
        DEFAULT_POLLINATIONS_IMAGE_MODELS if pollinations_enabled else [],
    )
    app.bot_data["pollinations_video_models"] = parse_csv_env_list(
        os.getenv("POLLINATIONS_VIDEO_MODELS"),
        DEFAULT_POLLINATIONS_VIDEO_MODELS if pollinations_enabled else [],
    )
    app.bot_data["token_limits"] = parse_limit_map(token_limits_env)
    app.bot_data["request_limits"] = parse_limit_map(request_limits_env)
    app.bot_data["request_semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    app.bot_data["telegram_send_semaphore"] = asyncio.Semaphore(1)

    for provider_id in sorted(app.bot_data.get("chat_providers", set())):
        app.bot_data[current_models_key(provider_id)] = list(STATIC_CHAT_MODELS.get(provider_id, []))
        app.bot_data[f"unavailable:{provider_id}"] = []

    class _BotDataContext:
        def __init__(self, bot_data: dict[str, Any]):
            self.bot_data = bot_data

    build_model_catalog(_BotDataContext(app.bot_data))
    WEB_RUNTIME["bot_data"] = app.bot_data

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_router))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    port = int(os.getenv("PORT", "0") or "0")
    if port > 0:
        web_loop = asyncio.new_event_loop()

        def run_web_loop() -> None:
            asyncio.set_event_loop(web_loop)
            web_loop.run_forever()

        threading.Thread(target=run_web_loop, daemon=True).start()

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    body = web_ui_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/config":
                    try:
                        bot_data = WEB_RUNTIME.get("bot_data", {})
                        payload_obj = build_web_config(bot_data)
                        payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
                        status = 200
                    except Exception as e:
                        payload = json.dumps({"error": str(e), "models": [], "roles": []}, ensure_ascii=False).encode("utf-8")
                        status = 500
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                if self.path != "/api/chat":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}
                try:
                    if "bot_data" not in WEB_RUNTIME:
                        raise RuntimeError("WEB_RUNTIME bot_data is not initialized")
                    result = asyncio.run_coroutine_threadsafe(web_chat_response(payload), web_loop).result(timeout=180)
                    status = 200 if result.get("ok") else 400
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
                    status = 500
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        def run_health_server() -> None:
            httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
            httpd.serve_forever()

        threading.Thread(target=run_health_server, daemon=True).start()

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

