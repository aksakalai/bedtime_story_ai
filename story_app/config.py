from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-story-drafting-v2-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2-VL-2B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe this child's drawing as a plain visual scene in one clean paragraph. Report only directly visible "
        "details. Focus on scenery and layout that a later story can reuse: main objects, characters if any, colors, "
        "counts, relative positions, facial expressions, clothing, background elements, shapes, markings, and "
        "anything visually distinctive. Prefer concrete spatial wording such as left, right, above, below, beside, "
        "in front of, or behind when helpful. Do not add opinions, art critique, atmosphere labels, style "
        "commentary, symbolism, hidden actions, emotions that are not visibly shown, or story events."
    )
    description_max_tokens: int = 384
    story_part_max_tokens: int = 256
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
