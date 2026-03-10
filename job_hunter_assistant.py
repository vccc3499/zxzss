#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HH_BASE = "https://api.hh.ru"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OLLAMA_BASE = "http://127.0.0.1:11434"

DEFAULT_CITY = "Сочи"
DEFAULT_MIN_SALARY = 150_000
DEFAULT_QUERY = "работа"
DEFAULT_PAGES = 3
DEFAULT_PER_PAGE = 50

STATE_PATH = Path("job_state.json")
OUTBOX_PATH = Path("outbox_replies.jsonl")
CHAT_LOG_PATH = Path("chat_replies.jsonl")


def load_dotenv_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


INTERVIEW_KEYWORDS = (
    "собесед",
    "интервью",
    "приглаша",
    "приглашаем",
    "встреч",
    "связаться",
    "созвон",
)


@dataclass
class Vacancy:
    vacancy_id: str
    name: str
    employer: str
    area: str
    url: str
    salary_from: int | None
    salary_to: int | None
    salary_currency: str | None
    snippet: str
    published_at: str | None

    @property
    def salary_max(self) -> int | None:
        if self.salary_from is None and self.salary_to is None:
            return None
        return max(v for v in [self.salary_from, self.salary_to] if v is not None)


def http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Codex Job Assistant)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    body = None
    if payload is not None:
        req_headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, method=method, headers=req_headers, data=body)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_form(url: str, data: dict[str, str], timeout: int = 20) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "seen_vacancy_ids": [],
            "drafted_vacancy_ids": [],
            "notified_message_hashes": [],
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "seen_vacancy_ids": [],
            "drafted_vacancy_ids": [],
            "notified_message_hashes": [],
        }


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_html(text: str) -> str:
    return (
        text.replace("<highlighttext>", "")
        .replace("</highlighttext>", "")
        .replace("<br />", "\n")
        .replace("<br>", "\n")
    )


def parse_vacancy(item: dict[str, Any]) -> Vacancy:
    salary = item.get("salary") or {}
    snippet = item.get("snippet") or {}
    requirement = clean_html(snippet.get("requirement", "") or "")
    responsibility = clean_html(snippet.get("responsibility", "") or "")
    full_snippet = "\n".join(x for x in [requirement, responsibility] if x).strip()

    return Vacancy(
        vacancy_id=str(item.get("id", "")),
        name=item.get("name", "Без названия"),
        employer=(item.get("employer") or {}).get("name", "Неизвестно"),
        area=(item.get("area") or {}).get("name", ""),
        url=item.get("alternate_url", ""),
        salary_from=salary.get("from"),
        salary_to=salary.get("to"),
        salary_currency=salary.get("currency"),
        snippet=full_snippet,
        published_at=item.get("published_at"),
    )


def fetch_vacancies(city: str, min_salary: int, query: str, pages: int, per_page: int) -> list[Vacancy]:
    all_items: list[Vacancy] = []
    for page in range(pages):
        params = {
            "text": f"{query} {city}",
            "salary": min_salary,
            "only_with_salary": "true",
            "per_page": per_page,
            "page": page,
            "order_by": "publication_time",
        }
        url = f"{HH_BASE}/vacancies?{urllib.parse.urlencode(params)}"
        data = http_json(url)
        for raw in data.get("items", []):
            all_items.append(parse_vacancy(raw))
        time.sleep(0.2)
    return all_items


def pass_filters(v: Vacancy, city: str, min_salary: int) -> bool:
    if city.lower() not in (v.area or "").lower():
        return False
    if v.salary_max is None:
        return False
    if v.salary_max < min_salary:
        return False
    return True


def chat_with_openrouter(messages: list[dict[str, str]], model: str) -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY не задан")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 550,
    }
    data = http_json(
        f"{OPENROUTER_BASE}/chat/completions",
        method="POST",
        payload=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Job Auto Assistant",
        },
    )
    return data["choices"][0]["message"]["content"].strip()


def chat_with_ollama(messages: list[dict[str, str]], model: str) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    data = http_json(f"{OLLAMA_BASE}/api/chat", method="POST", payload=payload, timeout=120)
    return data.get("message", {}).get("content", "").strip()


def llm_chat(messages: list[dict[str, str]]) -> str:
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    openrouter_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct:free").strip()

    if ollama_model:
        return chat_with_ollama(messages, ollama_model)
    return chat_with_openrouter(messages, openrouter_model)


def safe_llm_reply(messages: list[dict[str, str]], fallback_text: str) -> str:
    try:
        return llm_chat(messages)
    except Exception as e:
        print(f"[WARN] LLM недоступна, использован шаблонный ответ: {e}")
        return fallback_text


def build_reply_prompt(v: Vacancy, profile_text: str) -> list[dict[str, str]]:
    system = (
        "Ты помощник по трудоустройству. Пиши кратко, по-русски, уверенно и вежливо. "
        "Нельзя выдумывать опыт, навыки и проекты. Если данных мало, акцентируй обучаемость, "
        "дисциплину и готовность быстро вникнуть в задачи."
    )
    user = (
        "Составь отклик на вакансию.\n\n"
        f"Профиль кандидата:\n{profile_text}\n\n"
        f"Вакансия: {v.name}\n"
        f"Компания: {v.employer}\n"
        f"Город: {v.area}\n"
        f"Зарплата: from={v.salary_from}, to={v.salary_to}, currency={v.salary_currency}\n"
        f"Описание:\n{v.snippet}\n\n"
        "Формат:\n"
        "1) Короткое приветствие\n"
        "2) Почему интересна вакансия\n"
        "3) Чем кандидат может быть полезен\n"
        "4) Готовность к собеседованию\n"
        "До 1100 символов."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_chat_reply_prompt(incoming_text: str, profile_text: str, employer: str, vacancy_name: str) -> list[dict[str, str]]:
    system = (
        "Ты отвечаешь работодателю от лица кандидата. Нельзя врать или приписывать опыт. "
        "Тон деловой и доброжелательный. Обязательно подчеркни обучаемость и готовность работать."
    )
    user = (
        f"Профиль кандидата:\n{profile_text}\n\n"
        f"Работодатель: {employer}\n"
        f"Вакансия: {vacancy_name}\n\n"
        f"Сообщение работодателя:\n{incoming_text}\n\n"
        "Сформируй ответ кандидата до 700 символов."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def detect_interview_invite(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in INTERVIEW_KEYWORDS)


def extract_interview_details(text: str) -> dict[str, str]:
    date_match = re.search(r"\b(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)\b", text)
    time_match = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", text)
    phone_match = re.search(r"(?:\+7|8)[\s(\-]*\d{3}[\s)\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}", text)

    return {
        "date": date_match.group(0) if date_match else "не указана",
        "time": time_match.group(0) if time_match else "не указано",
        "phone": phone_match.group(0) if phone_match else "не указан",
    }


def message_hash(*parts: str) -> str:
    base = "||".join(parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def send_telegram_notification(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    result = post_form(url, {"chat_id": chat_id, "text": text})
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def cmd_scan(args: argparse.Namespace) -> int:
    vacancies = fetch_vacancies(args.city, args.min_salary, args.query, args.pages, args.per_page)
    filtered = [v for v in vacancies if pass_filters(v, args.city, args.min_salary)]

    print(f"Найдено: {len(vacancies)}, после фильтра: {len(filtered)}")
    for v in filtered[: args.show]:
        print("=" * 80)
        print(f"[{v.vacancy_id}] {v.name}")
        print(f"Компания: {v.employer}")
        print(f"Локация: {v.area}")
        print(f"ЗП: from={v.salary_from} to={v.salary_to} {v.salary_currency}")
        print(f"URL: {v.url}")
        if v.snippet:
            short = f"{v.snippet[:320]}{'...' if len(v.snippet) > 320 else ''}"
            print(f"Описание: {short}")
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    profile_text = args.profile.strip()
    if not profile_text:
        raise RuntimeError("Передай профиль кандидата через --profile")

    state = load_state()
    seen = set(state.get("seen_vacancy_ids", []))
    drafted = set(state.get("drafted_vacancy_ids", []))

    vacancies = fetch_vacancies(args.city, args.min_salary, args.query, args.pages, args.per_page)
    filtered = [v for v in vacancies if pass_filters(v, args.city, args.min_salary)]
    new_vacancies = [v for v in filtered if v.vacancy_id not in drafted]

    print(f"После фильтра: {len(filtered)}; новых для обработки: {len(new_vacancies)}")
    for v in new_vacancies:
        seen.add(v.vacancy_id)

        fallback = (
            f"Здравствуйте! Меня заинтересовала вакансия {v.name} в компании {v.employer}. "
            "Я дисциплинированный кандидат, быстро обучаюсь, ответственно подхожу к задачам "
            "и готов быстро включиться в рабочий процесс. Буду рад(а) пройти собеседование "
            "в удобное для вас время и обсудить, как могу быть полезен(на) вашей команде."
        )
        draft = safe_llm_reply(build_reply_prompt(v, profile_text), fallback)

        record = {
            "vacancy_id": v.vacancy_id,
            "vacancy_name": v.name,
            "employer": v.employer,
            "area": v.area,
            "url": v.url,
            "salary_from": v.salary_from,
            "salary_to": v.salary_to,
            "salary_currency": v.salary_currency,
            "reply_text": draft,
            "created_at_unix": int(time.time()),
            "mode": "manual-send-required",
        }
        append_jsonl(OUTBOX_PATH, record)
        drafted.add(v.vacancy_id)
        print(f"[OK] Черновик добавлен в {OUTBOX_PATH}: {v.vacancy_id} {v.name}")

    state["seen_vacancy_ids"] = sorted(seen)
    state["drafted_vacancy_ids"] = sorted(drafted)
    save_state(state)
    print(f"Состояние обновлено: {STATE_PATH}")
    return 0

def cmd_reply_chat(args: argparse.Namespace) -> int:
    if not args.incoming:
        raise RuntimeError("Передай текст сообщения работодателя через --incoming")
    if not args.profile.strip():
        raise RuntimeError("Передай профиль кандидата через --profile")

    employer = args.employer.strip() or "Работодатель"
    vacancy_name = args.vacancy.strip() or "Вакансия"

    fallback = (
        "Здравствуйте! Спасибо за сообщение. Мне интересна вакансия, "
        "я быстро обучаюсь, ответственно подхожу к задачам и готов(а) развиваться в работе."
    )
    reply = safe_llm_reply(
        build_chat_reply_prompt(args.incoming, args.profile, employer, vacancy_name),
        fallback,
    )
    print(reply)

    details = extract_interview_details(args.incoming)
    is_invite = detect_interview_invite(args.incoming)
    event_hash = message_hash(employer, vacancy_name, args.incoming)

    state = load_state()
    notified = set(state.get("notified_message_hashes", []))

    if is_invite and event_hash not in notified:
        notify_text = (
            "Найдено приглашение на собеседование.\n"
            f"Компания: {employer}\n"
            f"Вакансия: {vacancy_name}\n"
            f"Дата: {details['date']}\n"
            f"Время: {details['time']}\n"
            f"Телефон: {details['phone']}\n\n"
            f"Текст: {args.incoming}"
        )
        if args.notify_telegram:
            send_telegram_notification(notify_text)
            print("[OK] Уведомление отправлено в Telegram")
            notified.add(event_hash)
            state["notified_message_hashes"] = sorted(notified)
            save_state(state)
        else:
            print("[INFO] Приглашение распознано, но отправка в Telegram отключена")

    append_jsonl(
        CHAT_LOG_PATH,
        {
            "incoming": args.incoming,
            "reply": reply,
            "employer": employer,
            "vacancy": vacancy_name,
            "is_interview_invite": is_invite,
            "details": details,
            "ts": int(time.time()),
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Автоассистент откликов на вакансии по Сочи")
    sub = p.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Найти вакансии и показать результаты фильтра")
    scan.add_argument("--city", default=DEFAULT_CITY)
    scan.add_argument("--min-salary", type=int, default=DEFAULT_MIN_SALARY)
    scan.add_argument("--query", default=DEFAULT_QUERY)
    scan.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    scan.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    scan.add_argument("--show", type=int, default=20)
    scan.set_defaults(func=cmd_scan)

    auto = sub.add_parser("auto", help="Сгенерировать отклики по новым вакансиям")
    auto.add_argument("--city", default=DEFAULT_CITY)
    auto.add_argument("--min-salary", type=int, default=DEFAULT_MIN_SALARY)
    auto.add_argument("--query", default=DEFAULT_QUERY)
    auto.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    auto.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    auto.add_argument("--profile", required=True, help="Краткий честный профиль кандидата")
    auto.set_defaults(func=cmd_auto)

    reply_chat = sub.add_parser("reply-chat", help="Сгенерировать ответ работодателю")
    reply_chat.add_argument("--incoming", required=True, help="Входящее сообщение работодателя")
    reply_chat.add_argument("--profile", required=True, help="Краткий честный профиль кандидата")
    reply_chat.add_argument("--employer", default="", help="Название работодателя")
    reply_chat.add_argument("--vacancy", default="", help="Название вакансии")
    reply_chat.add_argument("--notify-telegram", action="store_true", default=False)
    reply_chat.set_defaults(func=cmd_reply_chat)

    return p


def main() -> int:
    load_dotenv_file()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP {e.code}: {detail}")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
