from __future__ import annotations

import re
from textwrap import dedent
from typing import Any

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
            "Write the beginning of a three-part bedtime story."
            " In one short paragraph, introduce the visible character or characters, the setting,"
            " and the calm starting situation from the drawing."
            " If no person or animal is visible, invent only one gentle main character and place that"
            " character inside this exact pictured setting."
            " Be sure to mention the uniquely identifiable details from the drawing description."
        ),
        "part_2": (
            "Write the middle of the same story."
            " Start exactly after the previous part ends."
            " In one short paragraph, let one specific gentle event happen."
            " The event must directly involve something clearly visible in the drawing."
            " Be sure to continue using the uniquely identifiable details from the drawing description."
            " Do not repeat or summarize the previous part."
        ),
        "part_3": (
            "Write the ending of the same story."
            " Start exactly after the previous part ends."
            " In one short paragraph, resolve the gentle event and finish with a clear final sentence."
            " Do not start a new event."
            " Keep the ending in the same setting unless the earlier parts already changed it."
            " Do not repeat or summarize the previous part."
        ),
    }
    if step_name not in step_instructions:
        raise ValidationError(f"Unknown story step: {step_name}")

    return dedent(
        f"""
        We are building a calm three-part bedtime story from one drawing description.

        Drawing description:
        {description_text}

        Task:
        {step_instructions[step_name]}

        Output rules:
        - Write exactly one paragraph.
        - Write only the story text.
        - Do not explain anything.
        - Do not label the part.
        - Do not mention being an AI or assistant.
        - The story must take place in the exact pictured scene described above.
        - Stay grounded in the drawing description and use concrete visual details from it.
        - Evaluate the uniquely identifiable details from the drawing description and make sure they appear naturally in the story, especially in part 1 and part 2.
        - Keep the same major objects, colors, and setting consistent across all three parts.
        - Keep the full arc clear: setup in part 1, event in part 2, conclusion in part 3.
        - Keep the tone warm, gentle, and bedtime-friendly.
        - Aim for about 45 to 65 words, but finish the paragraph cleanly.
        - End with a complete sentence.
        - Stop immediately after the paragraph.
        - Avoid a cliffhanger in part 3.

        Story text:
        """
    ).strip()


def build_story_messages(
    *,
    description_text: str,
    step_name: str,
    previous_parts: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You write only clean bedtime-story prose. Follow the user's formatting and length instructions exactly.",
        }
    ]

    part_1_prompt = build_story_part_prompt(
        description_text=description_text,
        step_name="part_1",
        previous_parts=[],
    )
    messages.append({"role": "user", "content": part_1_prompt})

    if step_name == "part_1":
        return messages

    if len(previous_parts) >= 1:
        messages.append({"role": "assistant", "content": previous_parts[0]})
        messages.append(
            {
                "role": "user",
                "content": build_story_part_prompt(
                    description_text=description_text,
                    step_name="part_2",
                    previous_parts=[previous_parts[0]],
                ),
            }
        )
    if step_name == "part_2":
        return messages

    if len(previous_parts) >= 2:
        messages.append({"role": "assistant", "content": previous_parts[1]})
        messages.append(
            {
                "role": "user",
                "content": build_story_part_prompt(
                    description_text=description_text,
                    step_name="part_3",
                    previous_parts=previous_parts[:2],
                ),
            }
        )
    return messages


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
    stripped = text.rstrip("\"')]} ")
    if not stripped.endswith((".", "!", "?")):
        raise ValidationError("Story part did not end with a complete sentence.")
    return text
