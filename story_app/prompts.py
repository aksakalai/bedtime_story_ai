from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

STORY_SYSTEM_PROMPT = (
    "You write only clean bedtime-story prose. Reply only with the story text. "
    "Do not include explanations, labels, or meta commentary."
)

PART_2_USER_PROMPT = (
    "Write the second part of the same story. Continue directly from the story above. "
    "Write only the story text."
)

PART_3_USER_PROMPT = (
    "Write the third and final part of the same story. Continue directly from the story above "
    "and end conclusively. Write only the story text."
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
        "The goal is to generate a three-part bedtime story based on the description of a child's drawing.\n\n"
        "Below is the description of the image. Carefully observe the unique details, elements, objects, and scenery "
        "that identify this picture, and fully use all of them in the story without missing details.\n\n"
        f"Description of the image:\n{description_text}\n\n"
        "Story requirements:\n"
        "- The story has three parts.\n"
        "- Part 1 introduces the main objects, scenery, unique details, tone, and characters.\n"
        "- Part 2 builds an event, intrigue, mystery, or gentle adventure from the same scene.\n"
        "- Part 3 concludes the story with a hopeful, optimistic, and positive ending.\n"
        "- Use the unique objects, scenery, colors, and distinctive details from the description naturally in the story.\n"
        "- Reply only with the story itself.\n"
        "- Do not mention AI, prompts, instructions, or that this comes from a drawing.\n"
        "- Aim for roughly 45 to 65 words per part, but finish cleanly.\n\n"
        "Now write the first part of the story."
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
