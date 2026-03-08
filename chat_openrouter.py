#!/usr/bin/env python3
"""
Terminal chat with OpenRouter free models.

Usage:
  1) Set API key:
     PowerShell: $env:OPENROUTER_API_KEY="your_key_here"
  2) Run:
     python chat_openrouter.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_SITE_URL = "http://localhost"
DEFAULT_SITE_NAME = "Local OpenRouter Chat Test"


def http_json(url: str, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> dict:
    body = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def get_free_models(api_key: str) -> list[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": DEFAULT_SITE_URL,
        "X-Title": DEFAULT_SITE_NAME,
    }
    data = http_json(f"{OPENROUTER_BASE}/models", headers=headers)
    models = data.get("data", [])
    free_ids = []
    for m in models:
        model_id = m.get("id", "")
        if ":free" in model_id:
            free_ids.append(model_id)
    return sorted(set(free_ids))


def try_model(api_key: str, model_id: str) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": DEFAULT_SITE_URL,
        "X-Title": DEFAULT_SITE_NAME,
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 32,
        "temperature": 0,
    }
    try:
        data = http_json(f"{OPENROUTER_BASE}/chat/completions", method="POST", payload=payload, headers=headers)
        choices = data.get("choices", [])
        if not choices:
            return False, "No choices in response"
        text = choices[0].get("message", {}).get("content", "").strip()
        if text:
            return True, text
        return False, "Empty text response"
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        return False, f"HTTP {e.code}: {err}"
    except Exception as e:
        return False, str(e)


def chat(api_key: str, model_id: str) -> None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": DEFAULT_SITE_URL,
        "X-Title": DEFAULT_SITE_NAME,
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers concise.",
        }
    ]

    print(f"\nChat started with model: {model_id}")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        messages.append({"role": "user", "content": user_text})
        payload = {"model": model_id, "messages": messages, "temperature": 0.7}
        try:
            data = http_json(f"{OPENROUTER_BASE}/chat/completions", method="POST", payload=payload, headers=headers)
            answer = data["choices"][0]["message"]["content"].strip()
            print(f"AI: {answer}\n")
            messages.append({"role": "assistant", "content": answer})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            print(f"[HTTP {e.code}] {detail}\n")
        except Exception as e:
            print(f"[Error] {e}\n")


def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print('PowerShell example: $env:OPENROUTER_API_KEY="your_key_here"')
        return 1
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        print("OPENROUTER_API_KEY must be the real ASCII key (looks like sk-or-...).")
        print("You likely set placeholder text instead of the actual API key.")
        return 1
    if not api_key.startswith("sk-or-"):
        print("OPENROUTER_API_KEY does not look like an OpenRouter key (expected prefix: sk-or-).")
        print("Please set your real key and run again.")
        return 1

    print("Fetching free models from OpenRouter...")
    try:
        free_models = get_free_models(api_key)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"Failed to fetch models. HTTP {e.code}: {detail}")
        return 1
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return 1

    if not free_models:
        print("No free models found.")
        return 1

    print(f"Found {len(free_models)} free models. Testing availability...")
    for m in free_models[:20]:
        ok, info = try_model(api_key, m)
        status = "OK" if ok else "FAIL"
        print(f"- {m}: {status}")
        if ok:
            print(f"  sample reply: {info}")
            chat(api_key, m)
            return 0

    print("No tested free models responded successfully.")
    print("Try again later or test more models by increasing the slice in code.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
