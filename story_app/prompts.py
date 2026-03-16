from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

DESCRIPTION_SYSTEM_PROMPT = (
    "You carefully observe the image and follow the current request. Write only one grounded prose paragraph that "
    "defines the scene and one central actor for a children's story. If a notable character is clearly present, use "
    "that character as the actor. If no notable character is clearly present, invent one fitting scene-related actor "
    "using a descriptive role instead of a proper name. Do not add labels or meta commentary."
)

DESCRIPTION_USER_PROMPT_SUFFIX = " Reply only with the description text."

STORY_SYSTEM_PROMPT = (
    "You write gentle bedtime-story prose for three consecutive children's picture-book pages that follow the same "
    "central actor across one simple story arc. Stay faithful to the scene-and-actor description and to earlier "
    "parts. Part 1 introduces the scene and actor. Part 2 introduces one visible event, mystery, or noticeable "
    "change affecting that actor. Part 3 resolves that same event with a calm hopeful ending. If the setting "
    "shifts, move only to a directly related nearby place from the previous page. Keep the actor consistent and do "
    "not introduce proper names unless the description already uses one. Reply only with the requested story text."
)

IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You turn one bedtime-story moment into one very short image prompt sentence for a CLIP-limited image model. "
    "The image should feel like a children's picture-book illustration. Reply with one short sentence only. Keep "
    "the same central actor and story continuity, but describe only what must be visible on this page. Mention the "
    "actor first or early, keep recurring anchors to a bare minimum, and highlight the one visible event, mystery, "
    "or resolved state that makes this page different. Do not restage the previous page, restate the whole scene "
    "description, or list every object. Do not add camera terms, artist names, text, captions, logos, watermarks, "
    "borders, frames, panels, or extra unrelated details."
)

PART_2_USER_PROMPT = (
    "Write only part 2 of the same bedtime story. Continue directly from part 1. Keep following the same central "
    "actor, and introduce one visible event, mystery, or noticeable change that affects that actor and makes this "
    "page look clearly different from part 1. If the setting shifts, move only to a directly related nearby place "
    "from part 1. Do not resolve the event yet. Use exactly 3 sentences. Reply only with the story text."
)

PART_3_USER_PROMPT = (
    "Write only part 3 of the same bedtime story. Continue directly from part 2. Keep following the same central "
    "actor, resolve the same event or change from part 2, and show the calm final state that makes this page look "
    "clearly different from part 2. If the setting shifts, move only to a directly related nearby place from part "
    "2. Use exactly 3 sentences. Reply only with the story text."
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
        f"Scene and actor description:\n{description_text}\n\n"
        "Open with the same central actor in the described setting. Establish who the actor is, where they are, and "
        "the calm mood of the page. Let the actor notice or approach something gentle, but do not start the main "
        "event yet. Keep it warm, concrete, and easy to illustrate. Use exactly 3 sentences. Reply only with the "
        "story text."
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
    previous_image_prompt: str | None = None,
) -> list[dict[str, Any]]:
    if part_index == 1:
        part_role = (
            "This is part 1, so establish the actor in the opening scene and make the clearest recurring anchors "
            "easy to recognize."
        )
    elif part_index == 2:
        part_role = (
            "This is part 2, so keep the same actor and make the new visible event or change the main focus."
        )
    else:
        part_role = (
            "This is part 3, so keep the same actor and make the resolved final state the main focus."
        )

    previous_page_section = ""
    if previous_image_prompt is not None:
        previous_page_section = (
            f"Previous page final image prompt:\n{previous_image_prompt}\n\n"
            "Preserve continuity with that page, but describe the next page instead of repeating it.\n\n"
        )

    prompt_text = (
        "Write one short sentence prompt for a single children's picture-book illustration of this story moment.\n\n"
        f"Scene and actor description:\n{description_text}\n\n"
        f"Story moment:\n{part_text}\n\n"
        f"{previous_page_section}"
        f"{part_role}\n\n"
        f"Keep the final sentence within {max_image_prompt_tokens} image-model tokens. Mention the same actor first "
        "or early, use at most two recurring anchor details only if they help continuity, and focus on the single "
        "page-defining visible change or settled ending state. Reply only with the final sentence."
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
    return normalize_text(raw_text)


def validate_story_part_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_text(raw_text)


def validate_image_prompt_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_text(raw_text)
