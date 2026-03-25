#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import html
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
SILICONFLOW_BASE = "https://api.siliconflow.com/v1"
LEGNEXT_BASE = os.getenv("LEGNEXT_BASE_URL", "https://api.legnext.ai/api/v1")
POLLINATIONS_TEXT_BASES = (
    "https://gen.pollinations.ai/v1",
    "https://text.pollinations.ai/openai/v1",
)
POLLINATIONS_GEN_BASE = os.getenv("POLLINATIONS_GEN_BASE", "https://gen.pollinations.ai")

SITE_URL = "http://localhost"
SITE_NAME = "Telegram Multi-Provider AI Bot"

PAGE_SIZE = 8
AGENTS_PAGE_SIZE = 8
ROLES_PAGE_SIZE = 8
MAX_HISTORY_MESSAGES = 200
MODEL_CHECK_CONCURRENCY = 6
MAX_CONCURRENT_REQUESTS = 3
MAX_GLOBAL_AGENTS = 50
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers concise and clear."
TELEGRAM_MESSAGE_CHUNK = 3900

BTN_PICK_MODEL = "\U0001f9e0 \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c"
BTN_CLEAR = "\U0001f9f9 \u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0434\u0438\u0430\u043b\u043e\u0433"
BTN_HELP = "\u2139\ufe0f \u041f\u043e\u043c\u043e\u0449\u044c"
BTN_PROFILE = "\U0001f4ca \u041f\u0440\u043e\u0444\u0438\u043b\u044c"
BTN_LIMITS = "\u23f1\ufe0f \u041b\u0438\u043c\u0438\u0442\u044b"
BTN_ROLE = "\U0001f3ad \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0440\u043e\u043b\u044c"

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
    return ReplyKeyboardMarkup(
        [
            [BTN_PICK_MODEL, BTN_CLEAR],
            [BTN_ROLE, BTN_PROFILE],
            [BTN_LIMITS, BTN_HELP],
        ],
        resize_keyboard=True,
    )


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
        f"\u041f\u0440\u043e\u0432\u0430\u0439\u0434\u0435\u0440\u044b: {providers_text}\n"
        "\u0422\u0438\u043f\u044b: CHAT / IMG / VIDEO\n"
        f"\u0420\u043e\u043b\u044c: {current_role}\n"
        "1) \u041d\u0430\u0436\u043c\u0438 \u00ab\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0440\u043e\u043b\u044c\u00bb\n"
        "2) \u041d\u0430\u0436\u043c\u0438 \u00ab\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c\u00bb\n"
        "3) \u041f\u0438\u0448\u0438 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u044f",
        parse_mode=ParseMode.HTML,
        reply_markup=menu_keyboard(),
    )
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>\u0423\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u0435</b>\n"
        "\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u043a\u043d\u043e\u043f\u043a\u0438.\n"
        "\u2022 \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0440\u043e\u043b\u044c \u2014 \u0437\u0430\u0434\u0430\u0451\u0442 \u0441\u0442\u0440\u043e\u0433\u0443\u044e \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044e \u043e\u0442\u0432\u0435\u0442\u0430\n"
        "\u2022 \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c \u2014 \u0432\u044b\u0431\u043e\u0440 \u043c\u043e\u0434\u0435\u043b\u0438\n"
        "\u2022 \u041f\u0440\u043e\u0444\u0438\u043b\u044c \u2014 \u0438\u0441\u0442\u043e\u0440\u0438\u044f \u0438 \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0430\n"
        "\u2022 \u041b\u0438\u043c\u0438\u0442\u044b \u2014 \u043e\u0441\u0442\u0430\u0442\u043a\u0438 \u043f\u043e \u043c\u043e\u0434\u0435\u043b\u044f\u043c\n"
        "\u2022 \u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0434\u0438\u0430\u043b\u043e\u0433 \u2014 \u0441\u0431\u0440\u043e\u0441 \u0438\u0441\u0442\u043e\u0440\u0438\u0438",
        parse_mode=ParseMode.HTML,
        reply_markup=menu_keyboard(),
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
        providers = entry.providers or [entry.provider_id]

        async def run_chat() -> tuple[ProviderResult | None, str | None, str | None]:
            last_warning_local: str | None = None
            catalog = context.bot_data.get("models_catalog", [])
            for provider_id in providers:
                client = context.bot_data["providers"].get(provider_id)
                if not client:
                    continue
                models = context.bot_data.get(current_models_key(provider_id), [])
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
                    if answer_looks_broken(text, attempt.answer):
                        for alt_entry in preferred_fallback_chat_entries(catalog, state.get("selected_model_key")):
                            alt_providers = alt_entry.providers or [alt_entry.provider_id]
                            for alt_provider_id in alt_providers:
                                alt_client = context.bot_data["providers"].get(alt_provider_id)
                                if not alt_client:
                                    continue
                                alt_models = context.bot_data.get(current_models_key(alt_provider_id), [])
                                if alt_entry.model_id not in alt_models:
                                    continue
                                try:
                                    alt_answer, alt_usage = await alt_client.chat_with_usage(alt_entry.model_id, history)
                                    if answer_looks_broken(text, alt_answer):
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
                                    )
                                except Exception:
                                    continue
                        last_warning_local = "Модель вернула некорректный ответ, попробуй другую модель."
                        continue
                    return attempt, provider_id, last_warning_local
                if attempt.warning:
                    last_warning_local = attempt.warning
            return None, None, last_warning_local

        semaphore: asyncio.Semaphore = context.bot_data["request_semaphore"]
        async with semaphore:
            result, used_provider, warning = await run_chat()

        if not result or not result.answer:
            await update.message.reply_text(
                "Ответ не получен. Попробуй еще раз.",
                reply_markup=menu_keyboard(),
            )
            return
        answer = result.answer
        update_usage_stats(state, used_provider, entry.model_id, result.usage, text, answer)
        if warning:
            await update.message.reply_text(warning, reply_markup=menu_keyboard())

        chunks = split_for_telegram(answer)
        for i, part in enumerate(chunks):
            if i == len(chunks) - 1:
                await update.message.reply_text(part, reply_markup=menu_keyboard())
            else:
                await update.message.reply_text(part)
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

    app = Application.builder().token(tg_token).build()
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

    app.add_handler(CommandHandler("start", start))
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

