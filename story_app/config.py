from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-v1-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Salesforce/blip-image-captioning-large"
    story_writer: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = "a child's drawing of"
    description_max_tokens: int = 80
    story_part_max_tokens: int = 140
    min_description_words: int = 6
    min_story_part_words: int = 20
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
