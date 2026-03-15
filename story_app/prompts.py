from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

MULTIMODAL_SYSTEM_PROMPT = (
    "You carefully observe the image and follow the current request. When asked for a description, write only a "
    "grounded description of visible scene details. When asked for a story part, write only gentle bedtime-story "
    "prose that stays faithful to the same image and the earlier conversation. Reply only with the requested text. "
    "Do not add labels or meta commentary."
)

DESCRIPTION_USER_PROMPT_SUFFIX = " Reply only with the description text."

PART_1_USER_PROMPT = (
    "Using the same image and your description above, write only part 1 of a gentle three-part bedtime story. Keep "
    "it grounded in the visible scene, warm, concise, and clean. Write about 50 words. Reply only with the story "
    "text."
)

PART_2_USER_PROMPT = (
    "Using the same image and the story so far, write only part 2 of the same bedtime story. Continue directly, stay "
    "grounded in the visible scene, keep it gentle, and write about 50 words. Reply only with the story text."
)

PART_3_USER_PROMPT = (
    "Using the same image and the story so far, write only part 3 of the same bedtime story. Continue directly, keep "
    "it grounded in the visible scene, end with a calm hopeful feeling, and write about 50 words. Reply only with "
    "the story text."
)


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValidationError("Model output was empty.")
    return text


def build_description_prompt(config: GenerationConfig) -> str:
    return config.description_prompt_prefix


def build_image_text_content(prompt_text: str) -> list[dict[str, str]]:
    return [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]


def build_description_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_image_text_content(prompt_text)},
    ]


def build_story_messages(
    *,
    description_prompt: str,
    description_text: str,
    previous_parts: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_image_text_content(description_prompt)},
        {"role": "assistant", "content": description_text},
        {"role": "user", "content": build_image_text_content(PART_1_USER_PROMPT)},
    ]

    if not previous_parts:
        return messages

    messages.append({"role": "assistant", "content": previous_parts[0]})

    if len(previous_parts) == 1:
        messages.append({"role": "user", "content": build_image_text_content(PART_2_USER_PROMPT)})
        return messages

    messages.append({"role": "user", "content": build_image_text_content(PART_2_USER_PROMPT)})
    messages.append({"role": "assistant", "content": previous_parts[1]})
    messages.append({"role": "user", "content": build_image_text_content(PART_3_USER_PROMPT)})
    return messages


def _format_message_content(content: Any) -> str:
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                lines.append("[IMAGE]")
            elif item_type == "text":
                lines.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in lines if part).strip()
    return str(content).strip()


def format_story_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = _format_message_content(message.get("content", ""))
        lines.append(f"{role}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def validate_description_text(raw_text: str, config: GenerationConfig) -> str:
    text = normalize_text(raw_text)
    if len(text.split()) < config.min_description_words:
        raise ValidationError(
            f"Description must contain at least {config.min_description_words} words."
        )
    return text


def validate_story_part_text(raw_text: str, config: GenerationConfig) -> str:
    text = normalize_text(raw_text)
    if len(text.split()) < config.min_story_part_words:
        raise ValidationError(
            f"Story part must contain at least {config.min_story_part_words} words."
        )
    return text
