from __future__ import annotations

from textwrap import dedent

from .config import GenerationConfig
from .schemas import DrawingDescription, SchemaError, StoryPackage, StoryPart

DESCRIPTION_RESPONSE_LABELS = (
    "SUMMARY",
    "CHARACTERS",
    "SETTING",
    "STYLE",
    "COLORS",
    "SAFETY",
)

STORY_RESPONSE_LABELS = (
    "TITLE",
    "PART1_ENTRANCE",
    "PART2_BUILDUP",
    "PART3_ENDING",
)


def _clip_words(text: str, max_words: int) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def _parse_exact_labeled_lines(raw_text: str, expected_labels: tuple[str, ...]) -> dict[str, str]:
    stripped = raw_text.strip()
    if not stripped:
        raise SchemaError("Model output was empty.")

    lines = stripped.splitlines()
    if len(lines) != len(expected_labels):
        raise SchemaError(
            f"Expected exactly {len(expected_labels)} labeled lines, received {len(lines)}."
        )

    parsed: dict[str, str] = {}
    for expected_label, line in zip(expected_labels, lines):
        prefix = f"{expected_label}:"
        if not line.startswith(prefix):
            raise SchemaError(f"Expected line to start with '{prefix}'.")
        value = line[len(prefix):].strip()
        if not value:
            raise SchemaError(f"{expected_label} must not be empty.")
        parsed[expected_label] = value
    return parsed


def _parse_comma_separated_values(value: str, label: str) -> list[str]:
    if ";" in value:
        raise SchemaError(f"{label} must use commas only.")

    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise SchemaError(f"{label} must be a comma-separated list of non-empty strings.")
    return items


def build_description_prompt(config: GenerationConfig) -> str:
    return dedent(
        f"""
        You are helping a deterministic bedtime-story pipeline describe a child's drawing.
        Return exactly 6 lines and nothing else.
        Keep every value on the same line as its label.

        Use this exact format and label order:
        SUMMARY: <one calm summary sentence>
        CHARACTERS: <comma-separated recurring characters or key objects>
        SETTING: <short setting phrase>
        STYLE: <short visual style phrase>
        COLORS: <comma-separated main colors>
        SAFETY: <comma-separated bedtime-safety notes>

        Rules:
        - Keep the content kid-safe, gentle, and bedtime-friendly.
        - Infer the recurring characters and the main setting from the drawing.
        - Do not use JSON, bullets, markdown, code fences, or extra commentary.
        - Use English only.
        - Assume the final audience is ages {config.age_range}.
        """
    ).strip()


def build_story_prompt(description: DrawingDescription, config: GenerationConfig) -> str:
    safety_guidance = "; ".join(description.safety_notes)
    characters = ", ".join(description.characters)
    colors = ", ".join(description.color_palette)
    return dedent(
        f"""
        You write warm bedtime stories for children ages {config.age_range}.
        Return exactly 4 lines and nothing else.
        Keep every value on the same line as its label.

        Use this exact format and label order:
        TITLE: <short story title>
        PART1_ENTRANCE: <the story entrance in 1 or 2 sentences>
        PART2_BUILDUP: <the gentle buildup in 1 or 2 sentences>
        PART3_ENDING: <the calm ending in 1 or 2 sentences>

        Story rules:
        - English only.
        - Use one or more recurring characters from the drawing.
        - Make PART1_ENTRANCE the entrance of the story.
        - Make PART2_BUILDUP the middle buildup of the story.
        - Make PART3_ENDING the bedtime-ready ending of the story.
        - Each part should be around 35 to 55 words.
        - Keep the tone calm, cozy, and bedtime-friendly.
        - Avoid scary, violent, or high-stakes conflict.
        - Do not use JSON, bullets, markdown, code fences, or extra commentary.

        Drawing description:
        SUMMARY: {description.summary}
        CHARACTERS: {characters}
        SETTING: {description.setting}
        STYLE: {description.visual_style}
        COLORS: {colors}
        SAFETY: {safety_guidance}
        """
    ).strip()


def parse_description_response(raw_text: str) -> DrawingDescription:
    parsed = _parse_exact_labeled_lines(raw_text, DESCRIPTION_RESPONSE_LABELS)
    return DrawingDescription(
        summary=parsed["SUMMARY"],
        characters=_parse_comma_separated_values(parsed["CHARACTERS"], "CHARACTERS"),
        setting=parsed["SETTING"],
        visual_style=parsed["STYLE"],
        color_palette=_parse_comma_separated_values(parsed["COLORS"], "COLORS"),
        safety_notes=_parse_comma_separated_values(parsed["SAFETY"], "SAFETY"),
    )


def parse_story_response(raw_text: str, config: GenerationConfig) -> StoryPackage:
    parsed = _parse_exact_labeled_lines(raw_text, STORY_RESPONSE_LABELS)
    return StoryPackage(
        title=parsed["TITLE"],
        age_range=config.age_range,
        parts=[
            StoryPart(scene_goal="entrance", story_text=parsed["PART1_ENTRANCE"]),
            StoryPart(scene_goal="buildup", story_text=parsed["PART2_BUILDUP"]),
            StoryPart(scene_goal="ending", story_text=parsed["PART3_ENDING"]),
        ],
    )


def build_character_bible(description: DrawingDescription) -> str:
    characters = ", ".join(description.characters[:3])
    colors = ", ".join(description.color_palette[:4])
    return (
        f"storybook illustration, { _clip_words(description.visual_style, 6) }, "
        f"{ _clip_words(description.setting, 8) }, "
        f"consistent characters: {characters}, colors: {colors}, "
        f"gentle mood, bedtime atmosphere"
    )


def enrich_story_with_image_prompts(
    story: StoryPackage,
    description: DrawingDescription,
    config: GenerationConfig,
) -> StoryPackage:
    character_bible = build_character_bible(description)
    enriched_parts: list[StoryPart] = []
    for index, part in enumerate(story.parts, start=1):
        scene_focus = _clip_words(part.story_text, 20)
        prompt = (
            f"{character_bible}, scene {index}, {part.scene_goal}, "
            f"{scene_focus}, cohesive storybook scene, no text"
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
