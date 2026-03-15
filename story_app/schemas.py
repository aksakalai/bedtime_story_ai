from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    image_prompt_part_1_path: Path
    image_prompt_part_2_path: Path
    image_prompt_part_3_path: Path
    story_part_1_image_path: Path
    story_part_2_image_path: Path
    story_part_3_image_path: Path
    story_part_1_audio_path: Path
    story_part_2_audio_path: Path
    story_part_3_audio_path: Path
    storyboard_manifest_path: Path


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
    image_prompt_part_1_path: str | None = None
    image_prompt_part_2_path: str | None = None
    image_prompt_part_3_path: str | None = None
    story_part_1_image_path: str | None = None
    story_part_2_image_path: str | None = None
    story_part_3_image_path: str | None = None
    story_part_1_audio_path: str | None = None
    story_part_2_audio_path: str | None = None
    story_part_3_audio_path: str | None = None
    storyboard_manifest_path: str | None = None


@dataclass(frozen=True)
class StoryboardManifestPart:
    index: int
    text: str
    image_prompt: str
    seed: int
    image_path: str
    audio_path: str | None = None
    audio_duration_seconds: float | None = None


@dataclass(frozen=True)
class StoryboardManifest:
    run_id: str
    input_image_path: str
    description_text: str
    parts: list[StoryboardManifestPart]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input_image_path": self.input_image_path,
            "description_text": self.description_text,
            "parts": [
                {
                    "index": part.index,
                    "text": part.text,
                    "image_prompt": part.image_prompt,
                    "seed": part.seed,
                    "image_path": part.image_path,
                    "audio_path": part.audio_path,
                    "audio_duration_seconds": part.audio_duration_seconds,
                }
                for part in self.parts
            ],
        }
