from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SchemaError(ValueError):
    """Raised when structured model output cannot be validated."""


def _ensure_list_of_strings(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise SchemaError(f"{field_name} must be a list of non-empty strings.")
    return [item.strip() for item in value]


def _ensure_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass
class DrawingDescription:
    summary: str
    characters: list[str]
    setting: str
    visual_style: str
    color_palette: list[str]
    safety_notes: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DrawingDescription":
        return cls(
            summary=_ensure_string(data.get("summary"), "summary"),
            characters=_ensure_list_of_strings(data.get("characters"), "characters"),
            setting=_ensure_string(data.get("setting"), "setting"),
            visual_style=_ensure_string(data.get("visual_style"), "visual_style"),
            color_palette=_ensure_list_of_strings(data.get("color_palette"), "color_palette"),
            safety_notes=_ensure_list_of_strings(data.get("safety_notes"), "safety_notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoryPart:
    scene_goal: str
    story_text: str
    image_prompt: str = ""
    image_path: str = ""
    audio_path: str = ""
    duration_sec: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryPart":
        duration = data.get("duration_sec", 0.0)
        if not isinstance(duration, (int, float)) or duration < 0:
            raise SchemaError("duration_sec must be a non-negative number.")
        return cls(
            scene_goal=_ensure_string(data.get("scene_goal"), "scene_goal"),
            story_text=_ensure_string(data.get("story_text"), "story_text"),
            image_prompt=str(data.get("image_prompt", "")).strip(),
            image_path=str(data.get("image_path", "")).strip(),
            audio_path=str(data.get("audio_path", "")).strip(),
            duration_sec=float(duration),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoryPackage:
    title: str
    age_range: str
    parts: list[StoryPart]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryPackage":
        parts_raw = data.get("parts")
        if not isinstance(parts_raw, list):
            raise SchemaError("parts must be a list.")
        parts = [StoryPart.from_dict(part) for part in parts_raw]
        if len(parts) != 3:
            raise SchemaError("StoryPackage must contain exactly 3 parts.")
        return cls(
            title=_ensure_string(data.get("title"), "title"),
            age_range=_ensure_string(data.get("age_range"), "age_range"),
            parts=parts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "age_range": self.age_range,
            "parts": [part.to_dict() for part in self.parts],
        }


@dataclass
class TimelineSegment:
    index: int
    start_sec: float
    end_sec: float
    story_text: str
    image_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineManifest:
    audio_path: str
    total_duration_sec: float
    segments: list[TimelineSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "total_duration_sec": self.total_duration_sec,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass
class RunManifest:
    run_id: str
    run_dir: str
    input_image_path: str
    description_path: str
    story_path: str
    timeline_path: str
    narration_audio_path: str
    scene_image_paths: list[str]
    part_audio_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    run_id: str
    run_dir: Path
    description: DrawingDescription
    story: StoryPackage
    timeline: TimelineManifest
    manifest: RunManifest
    story_markdown: str
    playback_html: str
    image_gallery: list[str]
    narration_audio_path: str
    manifest_path: str
