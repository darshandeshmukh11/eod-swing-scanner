"""Send messages via Telegram Bot API (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

TELEGRAM_MAX_MESSAGE_LEN = 4096


class TelegramConfigError(RuntimeError):
    pass


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def load_dotenv_if_present(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE); does not override existing env vars."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_telegram_credentials(
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> tuple[str, str]:
    load_dotenv_if_present()
    bot_token = (token or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not bot_token:
        raise TelegramConfigError(
            "Set TELEGRAM_BOT_TOKEN (create a bot via @BotFather on Telegram)."
        )
    if not chat:
        raise TelegramConfigError(
            "Set TELEGRAM_CHAT_ID (message @userinfobot or your bot, then use getUpdates)."
        )
    return bot_token, chat


def split_telegram_message(text: str, max_len: int = TELEGRAM_MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest)
            break
        split_at = rest.rfind("\n\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = rest.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(rest[:split_at].rstrip())
        rest = rest[split_at:].lstrip()
    return chunks


def send_telegram_message(
    text: str,
    *,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    disable_preview: bool = True,
) -> list[int]:
    """Send `text` to Telegram; returns list of message_ids (one per chunk)."""
    bot_token, chat = get_telegram_credentials(token, chat_id)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message_ids: list[int] = []

    for chunk in split_telegram_message(text):
        payload = {
            "chat_id": chat,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }
        body = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram API HTTP {exc.code}: {err_body}") from exc

        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        result = data.get("result") or {}
        if "message_id" in result:
            message_ids.append(int(result["message_id"]))

    return message_ids
