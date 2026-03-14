from __future__ import annotations

from textwrap import dedent

from .config import GenerationConfig
from .schemas import DrawingDescription, StoryPackage, StoryPart


def _clip_words(text: str, max_words: int) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def build_description_prompt(config: GenerationConfig) -> str:
    return dedent(
        f"""
        You are helping turn a child's drawing into a safe bedtime story experience.
        Look at the drawing and return exactly one JSON object with these keys:
        summary, characters, setting, visual_style, color_palette, safety_notes.

        Requirements:
        - Keep the content kid-safe and gentle.
        - Infer recurring characters and the main setting from the drawing.
        - `characters` and `color_palette` must be arrays of short strings.
        - `safety_notes` must be an array of short strings describing any themes to keep calm and age-appropriate.
        - Do not include markdown, code fences, or any text outside the JSON object.
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
        Based on the description below, return exactly one JSON object with keys:
        title, age_range, parts.

        The `parts` value must be an array of exactly 3 objects.
        Each part must contain:
        - scene_goal
        - story_text

        Story rules:
        - English only.
        - Calm, cozy, and bedtime-friendly.
        - No scary, violent, or high-stakes conflict.
        - Each `story_text` must be around 40 to 60 words.
        - The 3 parts should flow from beginning, middle, to gentle ending.
        - Do not include markdown, code fences, or extra commentary.

        Drawing description:
        - Summary: {description.summary}
        - Characters: {characters}
        - Setting: {description.setting}
        - Visual style: {description.visual_style}
        - Color palette: {colors}
        - Safety guidance: {safety_guidance}
        """
    ).strip()


def build_character_bible(description: DrawingDescription) -> str:
    characters = ", ".join(description.characters[:3])
    colors = ", ".join(description.color_palette[:4])
    return (
        f"storybook illustration, { _clip_words(description.visual_style, 6) }, "
        f"{ _clip_words(description.setting, 8) }, "
        f"characters: {characters}, colors: {colors}"
    )


def enrich_story_with_image_prompts(
    story: StoryPackage,
    description: DrawingDescription,
    config: GenerationConfig,
) -> StoryPackage:
    character_bible = build_character_bible(description)
    enriched_parts: list[StoryPart] = []
    for index, part in enumerate(story.parts, start=1):
        scene_focus = _clip_words(part.image_prompt or part.story_text, 18)
        prompt = (
            f"{character_bible}, scene {index}, { _clip_words(part.scene_goal, 6) }, "
            f"{scene_focus}, bedtime mood, no text"
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
