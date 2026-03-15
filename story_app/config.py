from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-v2-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Salesforce/blip-image-captioning-base"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = "a child's drawing of"
    description_max_tokens: int = 64
    story_part_max_tokens: int = 120
    min_description_words: int = 6
    min_story_part_words: int = 20
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
