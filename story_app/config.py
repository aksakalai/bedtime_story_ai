from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelIds:
    drawing_describer: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    scene_generator: str = "segmind/SSD-1B"
    narrator: str = "hexgrad/Kokoro-82M"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    image_width: int = 768
    image_height: int = 768
    diffusion_steps: int = 16
    guidance_scale: float = 9.0
    story_max_tokens: int = 280
    description_max_tokens: int = 160
    narrator_voice: str = "af_heart"
    narrator_speed: float = 0.96
    random_seed: int = 42
    age_range: str = "5-10"
    image_negative_prompt: str = (
        "blurry, scary, horror, violence, gore, text, watermark, logo, low quality, deformed"
    )
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
