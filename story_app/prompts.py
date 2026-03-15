from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

STORY_SYSTEM_PROMPT = (
    "You write only gentle bedtime-story prose grounded in the provided scene. Keep every part in the same setting "
    "and close to the visible details. Build from the description and the earlier story instead of inventing "
    "unrelated elements. Do not add major new characters, named companions, locations, props, backstory, or time "
    "jumps unless they clearly grow out of the described scene. Reply only with the story text. Do not include "
    "explanations, labels, or meta commentary."
)

PART_2_USER_PROMPT = (
    "Write only part 2 of the same bedtime story.\n\n"
    "Requirements:\n"
    "- Continue directly from part 1 in the same setting and the same moment or an immediate continuation.\n"
    "- Keep using the same visible objects, scenery, and mood already established.\n"
    "- Build one gentle event, discovery, mystery, or small adventure from those existing details.\n"
    "- Do not introduce major new characters, locations, props, or unrelated themes.\n"
    "- Avoid sudden time jumps.\n"
    "- Aim for roughly 45 to 65 words.\n"
    "- Write only the story text."
)

PART_3_USER_PROMPT = (
    "Write only part 3 of the same bedtime story.\n\n"
    "Requirements:\n"
    "- Continue directly from part 2 in the same setting and the same moment or an immediate continuation.\n"
    "- Resolve the gentle event using the same scene and details already established.\n"
    "- Do not introduce major new characters, locations, props, or unrelated themes.\n"
    "- End with a calm, hopeful, bedtime-safe feeling.\n"
    "- Avoid sudden time jumps.\n"
    "- Aim for roughly 45 to 65 words.\n"
    "- Write only the story text."
)


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValidationError("Model output was empty.")
    return text


def build_description_prompt(config: GenerationConfig) -> str:
    return config.description_prompt_prefix


def build_story_part_1_prompt(description_text: str) -> str:
    return (
        "The goal is to write part 1 of a three-part bedtime story grounded in the scene described below.\n\n"
        f"Description of the image:\n{description_text}\n\n"
        "Requirements:\n"
        "- Part 1 should feel like the opening scene of a story, not a caption or checklist.\n"
        "- Stay tightly anchored to the exact setting and visible details from the description.\n"
        "- Use the distinctive objects, colors, and spatial relationships naturally in the prose.\n"
        "- Do not copy the description sentence by sentence.\n"
        "- Do not introduce major new characters, locations, or props.\n"
        "- If no character is clearly visible, you may use one gentle unnamed focal character or softly personify an existing visible element, but do not add a cast.\n"
        "- End with a small point of curiosity that naturally leads into part 2.\n"
        "- Keep the tone warm, calm, and bedtime-safe.\n"
        "- Aim for roughly 45 to 65 words.\n"
        "- Reply only with the story itself.\n"
        "- Do not mention AI, prompts, instructions, or that this comes from a drawing.\n"
        "\nNow write only part 1 of the story."
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


def format_story_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
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
