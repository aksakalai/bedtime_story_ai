from __future__ import annotations

import re
from textwrap import dedent

from .config import GenerationConfig
from .schemas import DrawingDescription, SchemaError, StoryPackage, StoryPart


def _clip_words(text: str, max_words: int) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def _normalize_text_block(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise SchemaError("Model output was empty.")
    return text


def _word_count(text: str) -> int:
    return len(text.split())


def build_description_prompt(_config: GenerationConfig) -> str:
    return "<MORE_DETAILED_CAPTION>"


def build_story_prompt(description: DrawingDescription, config: GenerationConfig) -> str:
    return dedent(
        f"""
        You write warm bedtime stories for children ages {config.age_range}.
        Use the drawing description below to create one story with exactly 3 parts.

        Return exactly this structure and nothing else:
        TITLE: <short title on one line>
        PARTS:
        <entrance part>
        {config.story_separator_token}
        <buildup part>
        {config.story_separator_token}
        <ending part>

        Rules:
        - English only.
        - Keep the same core characters and setting from the drawing description.
        - Part 1 must be the entrance.
        - Part 2 must be the buildup.
        - Part 3 must be the ending.
        - Each part should be 35 to 55 words.
        - Keep the tone calm, gentle, cozy, and bedtime-friendly.
        - Avoid scary, violent, or high-stakes conflict.
        - Use the separator token exactly two times.
        - Do not use JSON, markdown, bullets, or extra commentary.

        Drawing description:
        {description.text}
        """
    ).strip()


def parse_description_response(raw_text: str, config: GenerationConfig) -> DrawingDescription:
    text = _normalize_text_block(raw_text)
    if _word_count(text) < config.min_description_words:
        raise SchemaError(
            f"Description must contain at least {config.min_description_words} words."
        )
    return DrawingDescription(text=text)


def parse_story_response(raw_text: str, config: GenerationConfig) -> StoryPackage:
    stripped = raw_text.strip()
    if not stripped:
        raise SchemaError("Story response was empty.")

    lines = stripped.splitlines()
    if len(lines) < 4:
        raise SchemaError("Story response must include a title and three parts.")

    title_line = lines[0].strip()
    if not title_line.startswith("TITLE:"):
        raise SchemaError("Story response must start with 'TITLE:'.")
    title = title_line[len("TITLE:"):].strip()
    if not title:
        raise SchemaError("Story title must not be empty.")

    if lines[1].strip() != "PARTS:":
        raise SchemaError("Story response must include a 'PARTS:' line after the title.")

    body = "\n".join(lines[2:]).strip()
    separator_count = body.count(config.story_separator_token)
    if separator_count != 2:
        raise SchemaError(
            f"Story response must include exactly 2 '{config.story_separator_token}' tokens."
        )

    raw_parts = [part.strip() for part in body.split(config.story_separator_token)]
    if len(raw_parts) != 3 or any(not part for part in raw_parts):
        raise SchemaError("Story response must contain exactly 3 non-empty parts.")

    parts = [
        StoryPart(scene_goal=scene_goal, story_text=_normalize_text_block(part))
        for scene_goal, part in zip(("entrance", "buildup", "ending"), raw_parts)
    ]
    for part in parts:
        if _word_count(part.story_text) < config.min_story_part_words:
            raise SchemaError(
                f"Story part '{part.scene_goal}' must contain at least {config.min_story_part_words} words."
            )

    return StoryPackage(
        title=title,
        age_range=config.age_range,
        parts=parts,
    )


def build_character_bible(description: DrawingDescription) -> str:
    return (
        f"storybook illustration, same recurring visual world as this drawing description: "
        f"{_clip_words(description.text, 40)}"
    )


def enrich_story_with_image_prompts(
    story: StoryPackage,
    description: DrawingDescription,
    config: GenerationConfig,
) -> StoryPackage:
    description_context = build_character_bible(description)
    enriched_parts: list[StoryPart] = []
    for index, part in enumerate(story.parts, start=1):
        scene_focus = _clip_words(part.story_text, 24)
        prompt = (
            f"{description_context}, scene {index}, {part.scene_goal}, "
            f"scene details: {scene_focus}, bedtime mood, cohesive children's book art, no text"
        )
        enriched_parts.append(
            StoryPart(
                scene_goal=part.scene_goal,
                story_text=part.story_text,
                image_prompt=prompt,
                image_path=part.image_path,
                audio_path=part.audio_path,
                duration_sec=part.duration_sec,
            )
        )
    return StoryPackage(title=story.title, age_range=config.age_range, parts=enriched_parts)


def build_story_markdown(story: StoryPackage) -> str:
    sections = [f"# {story.title}", f"*Recommended age:* {story.age_range}"]
    for index, part in enumerate(story.parts, start=1):
        sections.append(f"## Part {index}")
        sections.append(part.story_text)
    return "\n\n".join(sections)
