from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

DESCRIPTION_SYSTEM_PROMPT = (
    "You carefully observe the image and follow the current request. Write only a grounded description of the "
    "depicted scene. Do not add labels or meta commentary."
)

DESCRIPTION_USER_PROMPT_SUFFIX = " Reply only with the description text."

STORY_SYSTEM_PROMPT = (
    "You write gentle bedtime-story prose. Stay faithful to the provided scene description and the earlier story "
    "parts. Reply only with the requested story text. Do not add labels or meta commentary."
)

IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You turn one bedtime-story moment into one short image prompt sentence for a CLIP-limited image model. "
    "Reply with exactly one sentence describing only the unique visible details of the scene. Keep the sentence "
    "concrete and visual. Prioritize the main subject, distinctive objects, setting, time of day, and one visible "
    "action. Do not repeat the whole story. Do not add camera terms, artist names, text, captions, logos, "
    "watermarks, borders, frames, panels, or extra unrelated details."
)

PART_2_USER_PROMPT = "Write only part 2 of the same bedtime story. Continue directly, stay grounded in the same scene description, keep it gentle, and write about 50 words. Reply only with the story text."

PART_3_USER_PROMPT = "Write only part 3 of the same bedtime story. Continue directly, stay grounded in the same scene description, end with a calm hopeful feeling, and write about 50 words. Reply only with the story text."


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
        {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_image_text_content(prompt_text)},
    ]


def build_story_part_1_prompt(description_text: str) -> str:
    return (
        "Write only part 1 of a gentle three-part bedtime story based on the scene description below.\n\n"
        f"Scene description:\n{description_text}\n\n"
        "Keep the story grounded in those visible details, begin in that scene, stay warm and clean, and write about "
        "50 words. Reply only with the story text."
    )


def build_story_messages(
    *,
    description_text: str,
    previous_parts: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": build_story_part_1_prompt(description_text)},
    ]

    if not previous_parts:
        return messages

    messages.append({"role": "assistant", "content": previous_parts[0]})

    if len(previous_parts) == 1:
        messages.append({"role": "user", "content": PART_2_USER_PROMPT})
        return messages

    messages.append({"role": "user", "content": PART_2_USER_PROMPT})
    messages.append({"role": "assistant", "content": previous_parts[1]})
    messages.append({"role": "user", "content": PART_3_USER_PROMPT})
    return messages


def build_story_part_image_prompt(
    config: GenerationConfig,
    *,
    description_text: str,
    part_text: str,
) -> str:
    return (
        "Create one single polished storybook illustration for this exact bedtime story moment. "
        f"Scene grounding: {description_text} "
        f"Story moment: {part_text} "
        "Show only one continuous scene from this moment, keep the setting and characters consistent across parts, "
        "and avoid adding unrelated objects or extra characters. "
        f"{config.image_prompt_style_suffix}"
    )


def build_story_part_image_summary_messages(
    config: GenerationConfig,
    *,
    description_text: str,
    part_text: str,
    max_image_prompt_tokens: int,
) -> list[dict[str, Any]]:
    prompt_text = (
        "Write one short sentence prompt for a single illustration of this story moment.\n\n"
        f"Scene description:\n{description_text}\n\n"
        f"Story moment:\n{part_text}\n\n"
        f"Keep the final sentence within {max_image_prompt_tokens} image-model tokens. Focus only on the unique "
        "visible details from this one moment. Reply only with the final sentence."
    )
    return [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def finalize_image_prompt(config: GenerationConfig, scene_prompt_text: str) -> str:
    scene_prompt = normalize_text(scene_prompt_text)
    if config.image_prompt_style_suffix.strip():
        scene_prompt = scene_prompt.rstrip(".,;:")
        return f"{scene_prompt}, {config.image_prompt_style_suffix}"
    return scene_prompt


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


def validate_image_prompt_text(raw_text: str, config: GenerationConfig) -> str:
    text = normalize_text(raw_text)
    sentence_parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    if len(sentence_parts) > 1:
        raise ValidationError(
            "Image prompt summary must be exactly one sentence."
        )
    return text
