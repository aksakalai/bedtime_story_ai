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
    "You write gentle bedtime-story prose for three consecutive children's picture-book pages. Stay faithful to "
    "the scene description and earlier parts. Make each part easy to illustrate and clearly different from the "
    "others. Reply only with the requested story text."
)

IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You turn one bedtime-story moment into one very short image prompt sentence for a CLIP-limited image model. "
    "The image should feel like a children's picture-book illustration. Reply with one short sentence only. "
    "Name just the few visual details needed to recognize the scene and highlight the single visible event or "
    "change that makes this page different. Keep recurring background details to a minimum. Do not restate the "
    "whole scene description or list every object. Do not add camera terms, artist names, text, captions, logos, "
    "watermarks, borders, frames, panels, or extra unrelated details."
)

PART_2_USER_PROMPT = (
    "Write only part 2 of the same bedtime story. Continue directly from part 1. Stay grounded in the same scene "
    "description and introduce one clear visible change or event so this page looks noticeably different from part "
    "1. Keep it gentle. Write 45 to 55 words in 2 or 3 sentences. Reply only with the story text."
)

PART_3_USER_PROMPT = (
    "Write only part 3 of the same bedtime story. Continue directly from part 2. Stay grounded in the same scene "
    "description and conclude with a calm hopeful ending that shows another visible change and the settled final "
    "state. Write 45 to 55 words in 2 or 3 sentences. Reply only with the story text."
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
        {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_image_text_content(prompt_text)},
    ]


def build_story_part_1_prompt(description_text: str) -> str:
    return (
        "Write only part 1 of a gentle three-part bedtime story based on the scene description below.\n\n"
        f"Scene description:\n{description_text}\n\n"
        "Begin with a clear opening picture-book scene. Establish the setting and main subjects. Do not introduce "
        "the main change yet. Keep it warm and grounded in the visible details. Write 45 to 55 words in 2 or 3 "
        "sentences. Reply only with the story text."
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
    part_index: int,
    max_image_prompt_tokens: int,
) -> list[dict[str, Any]]:
    if part_index == 1:
        part_role = (
            "This is part 1, so emphasize the opening scene and the main subjects clearly."
        )
    elif part_index == 2:
        part_role = (
            "This is part 2, so make the new event or visible change the main focus."
        )
    else:
        part_role = (
            "This is part 3, so make the calm concluding change or ending state the main focus."
        )

    prompt_text = (
        "Write one short sentence prompt for a single children's picture-book illustration of this story moment.\n\n"
        f"Scene description:\n{description_text}\n\n"
        f"Story moment:\n{part_text}\n\n"
        f"{part_role}\n\n"
        f"Keep the final sentence within {max_image_prompt_tokens} image-model tokens. Aim well below that limit. "
        "Mention only the main subject, at most two recurring anchor details, and the one new visible event or "
        "ending change that matters most on this page. Reply only with the final sentence."
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
    return normalize_text(raw_text)
