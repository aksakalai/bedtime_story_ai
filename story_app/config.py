from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelIds:
    drawing_describer: str = "Salesforce/blip-image-captioning-base"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    scene_generator: str = "stabilityai/sd-turbo"
    narrator: str = "hexgrad/Kokoro-82M"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    image_width: int = 512
    image_height: int = 512
    diffusion_steps: int = 4
    guidance_scale: float = 0.0
    story_max_tokens: int = 280
    description_max_tokens: int = 80
    narrator_voice: str = "af_heart"
    narrator_speed: float = 0.96
    random_seed: int = 42
    age_range: str = "5-10"
    story_separator_token: str = "<PART_BREAK>"
    min_description_words: int = 8
    min_story_part_words: int = 20
    image_prompt_description_words: int = 18
    image_prompt_story_words: int = 18
    image_negative_prompt: str = ""
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
