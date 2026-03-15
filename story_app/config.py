from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase2-text-to-image-v12-20260316"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    part_image_generator: str = "segmind/SSD-1B"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe the depicted scene itself in one concise paragraph. Include distinctive objects, colors, counts, "
        "positions, characters or animals if clearly visible, and other uniquely identifiable details. Use only "
        "visible scene facts. Do not mention the image, drawing, painting, paper, style, artist, or composition "
        "unless those are part of the depicted scene itself. Reply only with the description text."
    )
    description_max_tokens: int = 192
    story_part_max_tokens: int = 128
    image_width: int = 1024
    image_height: int = 1024
    image_num_inference_steps: int = 20
    image_guidance_scale: float = 9.0
    image_prompt_style_suffix: str = (
        "Warm, polished children's-book illustration, soft cinematic lighting, expressive characters, "
        "clean composition, bedtime mood, rich color harmony, high detail. No visible text, captions, logos, "
        "watermarks, frames, split panels, speech bubbles, collages, or page borders."
    )
    image_negative_prompt: str = (
        "low quality, blurry, muddy colors, flat lighting, deformed anatomy, extra limbs, duplicate characters, "
        "cropped face, text, letters, watermark, logo, signature, frame, border, collage, split panel"
    )
    image_seed_stride: int = 1000
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
