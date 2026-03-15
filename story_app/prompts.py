from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

STORY_SYSTEM_PROMPT = (
    "You write only concise, gentle bedtime-story prose. The scene description is the entire story world. Use only "
    "the characters, animals, objects, colors, and scenery explicitly stated in that description. Do not add any new "
    "character, creature, object, scenery detail, location, off-screen space, backstory, or time jump. Add only "
    "small actions and feelings that directly involve the described elements. Reply only with the story text. Do not "
    "include explanations, labels, or meta commentary."
)

PART_2_USER_PROMPT = (
    "Write only part 2 of the same bedtime story.\n\n"
    "Requirements:\n"
    "- Continue directly from part 1.\n"
    "- Stay in the exact same scene.\n"
    "- Build one small gentle development using only details explicitly present in the description and part 1.\n"
    "- Do not add any new detail that is not explicit in the description.\n"
    "- Keep the tone calm, clear, and bedtime-safe.\n"
    "- Aim for roughly 35 to 55 words.\n"
    "- Write only the story text."
)

PART_3_USER_PROMPT = (
    "Write only part 3 of the same bedtime story.\n\n"
    "Requirements:\n"
    "- Continue directly from part 2.\n"
    "- Stay in the exact same scene.\n"
    "- Resolve the gentle development using only details explicitly present in the description and earlier parts.\n"
    "- Do not add any new detail that is not explicit in the description.\n"
    "- End with a calm, hopeful, bedtime-safe feeling.\n"
    "- Aim for roughly 35 to 55 words.\n"
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
        "Write only part 1 of a three-part bedtime story based on the scene description below.\n\n"
        f"Scene description:\n{description_text}\n\n"
        "Requirements:\n"
        "- Treat the scene description as the entire story world.\n"
        "- Begin in the exact same scene.\n"
        "- Use the specific described details actively in the story prose.\n"
        "- Let part 1 feel like the opening moment of a story, not a summary.\n"
        "- Do not add any new detail that is not explicit in the description.\n"
        "- Add only small actions or feelings that directly involve the described elements.\n"
        "- End with one small gentle point of curiosity that can continue into part 2.\n"
        "- Keep the tone warm, calm, and bedtime-safe.\n"
        "- Aim for roughly 35 to 55 words.\n"
        "- Reply only with the story itself.\n"
        "- Do not mention AI, prompts, instructions, or the source of the scene description.\n"
        "\nNow write only part 1."
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
