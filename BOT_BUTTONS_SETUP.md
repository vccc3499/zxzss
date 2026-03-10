# Telegram Bot (Button-Only Mode)

## Start
1. Fill `.env` with `TELEGRAM_BOT_TOKEN` and provider keys.
2. Run:
   - `python tg_openrouter_bot.py`

## How it works
- After `/start`, bot auto-refreshes model lists for all providers.
- Auto-refresh is token-safe:
  - only model list endpoints are used;
  - no chat health-check requests are sent.
- Bot creates up to 50 agents from available models.

## Buttons
- `Обновить все` - refresh all providers and rebuild agents.
- `Агенты` - open agent picker and switch agent.
- `Роли` - pick specialization (teacher, coder, marketing, avitolog, biologist, etc.).
- `Провайдер` - switch provider for manual model mode.
- `Выбрать модель` - manual model picker for current provider.
- `Очистить диалог` - clear current chat history.
- `Профиль` - usage stats.
- `Помощь` - short in-bot help.
