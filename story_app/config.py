from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase1-story-drafting-v6-20260315"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2-VL-2B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe the visible scene in one concise, factual paragraph. Include only directly visible, uniquely "
        "identifiable details that a story writer could reuse exactly: objects, characters or animals if present, "
        "colors, counts, clothing, facial expressions, relative positions, foreground and background elements, and "
        "clear markings or shapes. Be concrete and specific. Prefer exact visible facts over atmosphere or summary. "
        "Use simple spatial wording such as left, right, above, below, beside, behind, or in front of when helpful. "
        "Do not mention the image itself, the medium, the artist, style, composition, symbolism, or your opinion. Do "
        "not infer hidden actions, relationships, backstory, or story events. If a detail is not clearly visible, "
        "leave it out."
    )
    description_max_tokens: int = 192
    story_part_max_tokens: int = 112
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
