from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ValidationError(ValueError):
    """Raised when a model output breaks the deterministic phase-1 contract."""


@dataclass(frozen=True)
class DescriptionResult:
    image_path: str
    prompt_text: str
    description_text: str


@dataclass(frozen=True)
class StoryDraft:
    full_conversation_text: str
    part_1_text: str
    part_2_text: str
    part_3_text: str


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    input_image_path: Path
    description_prompt_path: Path
    description_path: Path
    story_conversation_path: Path
    story_part_1_path: Path
    story_part_2_path: Path
    story_part_3_path: Path


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    input_image_path: str
    description: DescriptionResult
    draft: StoryDraft
    full_conversation_text: str
    part_1_text: str
    part_2_text: str
    part_3_text: str
