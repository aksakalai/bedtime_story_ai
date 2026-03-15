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

_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "black",
    "blue",
    "brown",
    "by",
    "child",
    "childs",
    "cloud",
    "clouds",
    "calm",
    "dark",
    "details",
    "drawing",
    "edge",
    "edges",
    "evening",
    "exact",
    "from",
    "green",
    "in",
    "is",
    "it",
    "its",
    "light",
    "lights",
    "line",
    "lines",
    "little",
    "morning",
    "of",
    "on",
    "one",
    "or",
    "outside",
    "picture",
    "pictured",
    "red",
    "scene",
    "setting",
    "small",
    "sky",
    "sunlight",
    "the",
    "this",
    "to",
    "tops",
    "two",
    "visible",
    "warm",
    "weather",
    "white",
    "with",
    "yellow",
}


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValidationError("Model output was empty.")
    return text


def _tokenize_words(raw_text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", raw_text.lower())


def extract_visual_anchor_words(description_text: str, max_words: int = 8) -> list[str]:
    anchors: list[str] = []
    for token in _tokenize_words(description_text):
        if len(token) < 3:
            continue
        if token in _ANCHOR_STOPWORDS:
            continue
        if token not in anchors:
            anchors.append(token)
        if len(anchors) >= max_words:
            break
    return anchors


def find_anchor_overlap(description_text: str, story_text: str) -> list[str]:
    anchors = extract_visual_anchor_words(description_text)
    story_words = set(_tokenize_words(story_text))
    return [anchor for anchor in anchors if anchor in story_words]


def validate_story_grounding(
    description_text: str,
    story_text: str,
    config: GenerationConfig,
) -> list[str]:
    anchors = extract_visual_anchor_words(description_text)
    if not anchors:
        return []
    overlap = find_anchor_overlap(description_text, story_text)
    required_overlap = min(config.min_story_anchor_overlap, len(anchors))
    if len(overlap) < required_overlap:
        raise ValidationError(
            "Story part drifted away from the drawing details. "
            f"Expected at least {required_overlap} anchor words from the description, got {len(overlap)}."
        )
    return overlap


def build_description_prompt(config: GenerationConfig) -> str:
    return config.description_prompt_prefix


def build_story_part_prompt(
    *,
    description_text: str,
    step_name: str,
    previous_parts: list[str],
) -> str:
    visual_anchors = extract_visual_anchor_words(description_text)
    step_instructions = {
        "part_1": (
            "Write the beginning of a three-part bedtime story."
            " In one short paragraph, introduce the visible character or characters, the setting,"
            " and the calm starting situation from the drawing."
            " If no person or animal is visible, invent only one gentle main character and place that"
            " character inside this exact pictured setting."
        ),
        "part_2": (
            "Write the middle of the same story."
            " Continue directly from the accepted story so far without restarting it."
            " In one short paragraph, let one specific gentle event happen."
            " The event must directly involve something clearly visible in the drawing."
        ),
        "part_3": (
            "Write the ending of the same story."
            " Continue directly from the accepted story so far."
            " In one short paragraph, resolve the gentle event and finish with a clear final sentence."
            " Do not start a new event."
            " Keep the ending in the same setting unless the earlier parts already changed it."
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

        Required visual anchors:
        {", ".join(visual_anchors) if visual_anchors else "Use the main visible objects from the description."}

        {story_so_far_block}Task:
        {step_instructions[step_name]}

        Output rules:
        - Write exactly one paragraph.
        - Write only the story text.
        - Do not explain anything.
        - Do not label the part.
        - Do not mention being an AI or assistant.
        - The story must take place in the exact pictured scene described above.
        - Stay grounded in the drawing description and use concrete visual details from it.
        - Use at least three concrete details from the drawing description in this paragraph whenever natural.
        - Keep the same major objects and setting consistent across all three parts.
        - Keep the action physically near the pictured objects instead of moving to a different place.
        - Do not introduce a new place, weather pattern, or major object that is not supported by the drawing description.
        - Keep the full arc clear: setup in part 1, event in part 2, conclusion in part 3.
        - Keep the tone warm, gentle, and bedtime-friendly.
        - Aim for about 45 to 55 words, but finish the paragraph cleanly.
        - End with a complete sentence.
        - Stop immediately after the paragraph.
        - Avoid a cliffhanger in part 3.

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
    stripped = text.rstrip("\"')]} ")
    if not stripped.endswith((".", "!", "?")):
        raise ValidationError("Story part did not end with a complete sentence.")
    return text
