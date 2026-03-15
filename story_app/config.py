from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-v12-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2-VL-2B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe this child's drawing in one clean paragraph using exact visible details only. Focus on uniquely "
        "identifiable details that can later be reused in a story: main objects, characters if any, colors, counts, "
        "positions, facial expressions, clothing, background elements, and anything visually distinctive. Do not "
        "invent hidden actions, emotions, or story events."
    )
    description_max_tokens: int = 320
    story_part_max_tokens: int = 256
    min_description_words: int = 10
    min_story_part_words: int = 20
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
