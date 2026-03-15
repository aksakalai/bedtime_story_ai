from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-story-drafting-v11-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "google/gemma-3-4b-it"
    story_writer: str = "google/gemma-3-4b-it"


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
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
