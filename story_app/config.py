from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-story-drafting-v7-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2-VL-2B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe only the visible scene in one concise paragraph. Include uniquely identifiable objects, characters "
        "or animals if present, colors, counts, relative positions, clothing, facial expressions, background "
        "elements, and clear shapes or markings. Use only directly visible facts. Be specific and concrete. Use "
        "simple spatial words such as left, right, above, below, beside, behind, or in front of when helpful. Do not "
        "mention the image, medium, artist, style, composition, symbolism, mood, or your opinion. Do not infer "
        "hidden actions, relationships, backstory, or events. Leave out anything not clearly visible."
    )
    description_max_tokens: int = 192
    story_part_max_tokens: int = 128
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
