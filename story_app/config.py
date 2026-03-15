from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-story-drafting-v4-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2-VL-2B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "This image is a child's drawing. Describe only the scene depicted in it, not the drawing as an artwork. "
        "Write one compact paragraph containing only directly visible scene details that a story model could reuse. "
        "Include concrete objects, characters or animals if present, colors, counts, relative positions, clothing, "
        "facial expressions, background elements, notable shapes or markings, and any uniquely identifiable features. "
        "Use simple spatial wording such as left, right, above, below, beside, behind, or in front of when helpful. "
        "Do not mention the drawing, the artist, style, brushstrokes, composition, quality, symbolism, or your "
        "opinion of the image. Do not infer hidden actions, relationships, emotions, backstory, or story events. If "
        "a detail is not clearly visible, leave it out."
    )
    description_max_tokens: int = 384
    story_part_max_tokens: int = 256
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
