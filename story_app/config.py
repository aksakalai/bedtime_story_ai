from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-v6-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe this child's drawing in exact visible detail. Mention the characters, important objects, "
        "colors, positions, facial expressions, clothing, background elements, and notable shapes or patterns. "
        "Do not invent hidden story events. Write one clean paragraph only."
    )
    description_max_tokens: int = 140
    story_part_max_tokens: int = 192
    min_description_words: int = 10
    min_story_part_words: int = 20
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
