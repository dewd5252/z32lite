"""Shared schema and helpers for Z32LITE dataset artifacts."""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_ACTION_PREFIX = "SYSTEM_ACTION:"
NOTIFY_USER_PREFIX = "NOTIFY_USER:"

KNOWN_SPLITS = {"train", "holdout", "eval"}
KNOWN_CATEGORIES = {
    "conversational_core",
    "tool_calling",
    "refusal_safety",
    "boundary_knowledge",
}
KNOWN_SOURCES = {"synthetic", "curated"}
KNOWN_LANGUAGES = {"ar-eg", "ar", "en", "mixed"}
KNOWN_ACTIONS = {
    "set_volume",
    "media_next_track",
    "media_prev_track",
    "media_play",
    "media_pause",
    "search_web",
    "search_contacts",
    "flashlight",
    "set_alarm",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def conversation_signature(example: dict[str, Any]) -> str:
    messages = example.get("conversations", [])
    flattened = "||".join(
        f"{message.get('from', '')}:{normalize_text(message.get('value', ''))}"
        for message in messages
    )
    return flattened


def parse_structured_response(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith(SYSTEM_ACTION_PREFIX):
        payload = json.loads(text[len(SYSTEM_ACTION_PREFIX) :].strip())
        return {"kind": "system_action", "payload": payload}
    if text.startswith(NOTIFY_USER_PREFIX):
        payload = json.loads(text[len(NOTIFY_USER_PREFIX) :].strip())
        return {"kind": "notify_user", "payload": payload}
    return {"kind": "text", "payload": None}


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not example.get("id"):
        errors.append("missing id")
    if example.get("split") not in KNOWN_SPLITS:
        errors.append(f"invalid split: {example.get('split')}")
    if example.get("category") not in KNOWN_CATEGORIES:
        errors.append(f"invalid category: {example.get('category')}")
    if example.get("source") not in KNOWN_SOURCES:
        errors.append(f"invalid source: {example.get('source')}")
    if example.get("language") not in KNOWN_LANGUAGES:
        errors.append(f"invalid language: {example.get('language')}")

    tags = example.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append("tags must be a list[str]")

    conversations = example.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        errors.append("conversations must contain exactly two turns")
        return errors

    expected_roles = ["user", "assistant"]
    for idx, (message, expected_role) in enumerate(zip(conversations, expected_roles)):
        if message.get("from") != expected_role:
            errors.append(f"conversation turn {idx} must be from {expected_role}")
        value = message.get("value")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"conversation turn {idx} has empty value")

    assistant_value = conversations[1]["value"]
    try:
        parsed = parse_structured_response(assistant_value)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid structured JSON: {exc}")
        parsed = {"kind": "invalid", "payload": None}

    if parsed["kind"] == "system_action":
        action = parsed["payload"].get("action")
        if action not in KNOWN_ACTIONS:
            errors.append(f"unknown system action: {action}")
    elif parsed["kind"] == "notify_user":
        payload = parsed["payload"]
        if not payload.get("message"):
            errors.append("notify payload missing message")
        if not payload.get("action_pending"):
            errors.append("notify payload missing action_pending")

    evaluation = example.get("evaluation")
    if evaluation is not None:
        if example.get("split") != "eval":
            errors.append("evaluation block is only allowed for eval split")
        elif evaluation.get("mode") not in {"structured_exact", "text_heuristic"}:
            errors.append(f"invalid evaluation mode: {evaluation.get('mode')}")

    return errors


def build_qwen_chat_text(example: dict[str, Any], system_prompt: str) -> str:
    user_text = example["conversations"][0]["value"].strip()
    assistant_text = example["conversations"][1]["value"].strip()
    return (
        "<|im_start|>system\n"
        f"{system_prompt.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_text}\n"
        "<|im_end|>"
    )
