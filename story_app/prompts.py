from __future__ import annotations

import re
from textwrap import dedent

from .config import GenerationConfig
from .schemas import ValidationError


_META_PREFIX_PATTERNS = (
    r"^(here is|here's)\b",
    r"^sure\b",
    r"^of course\b",
    r"^certainly\b",
    r"^part\s*[123]\b",
    r"^part\s*(one|two|three)\b",
    r"^(first|second|third|final)\s+part\b",
    r"^the\s+(first|second|third|final)\s+part\b",
    r"^i\s+(wrote|have written|created)\b",
    r"^story\s*:",
)


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValidationError("Model output was empty.")
    return text


def build_description_prompt(config: GenerationConfig) -> str:
    return config.description_prompt_prefix


def build_story_part_prompt(
    *,
    description_text: str,
    step_name: str,
    previous_parts: list[str],
) -> str:
    step_instructions = {
        "part_1": (
            "Write only the first part of the story, which is the beginning."
            " Introduce the characters and setting."
        ),
        "part_2": (
            "Write only the second part of the same story, which is the middle."
            " Continue naturally from the story so far without restarting it."
        ),
        "part_3": (
            "Write only the third and final part of the same story, which is the ending."
            " Finish the story gently."
        ),
    }
    if step_name not in step_instructions:
        raise ValidationError(f"Unknown story step: {step_name}")

    story_so_far_block = ""
    if previous_parts:
        story_so_far_block = "\n".join(
            [
                "Accepted story so far:",
                *[
                    f"{index}. {part}"
                    for index, part in enumerate(previous_parts, start=1)
                ],
                "",
            ]
        )

    return dedent(
        f"""
        We are building a calm three-part bedtime story from one drawing description.

        Drawing description:
        {description_text}

        {story_so_far_block}Task:
        {step_instructions[step_name]}

        Output rules:
        - Write only the story text.
        - Do not explain anything.
        - Do not label the part.
        - Do not mention being an AI or assistant.
        - Stay grounded in the drawing description.
        - Keep the tone warm, gentle, and bedtime-friendly.
        - Write 45 to 65 words.

        Story text:
        """
    ).strip()


def validate_description_text(raw_text: str, config: GenerationConfig) -> str:
    text = normalize_text(raw_text)
    if len(text.split()) < config.min_description_words:
        raise ValidationError(
            f"Description must contain at least {config.min_description_words} words."
        )
    forbidden_markers = ("```", "{", "}", "<|", "|>", "<", ">")
    if any(marker in text for marker in forbidden_markers):
        raise ValidationError("Description contained structured output markers or special tokens.")
    return text


def validate_story_part_text(raw_text: str, config: GenerationConfig) -> str:
    text = normalize_text(raw_text)
    if len(text.split()) < config.min_story_part_words:
        raise ValidationError(
            f"Story part must contain at least {config.min_story_part_words} words."
        )
    lowered = text.lower()
    for pattern in _META_PREFIX_PATTERNS:
        if re.match(pattern, lowered):
            raise ValidationError("Story part started with assistant wrapper text.")
    if any(marker in text for marker in ("```", "{", "}", "<|", "|>")):
        raise ValidationError("Story part contained structured output markers or special tokens.")
    return text
