from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .schemas import RunManifest, StoryPackage


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    input_image_path: Path
    description_path: Path
    description_prompt_path: Path
    description_response_path: Path
    story_path: Path
    story_prompt_path: Path
    story_response_path: Path
    timeline_path: Path
    narration_audio_path: Path
    video_path: Path
    manifest_path: Path
    images_dir: Path
    audio_dir: Path
    video_dir: Path


def prepare_run_paths(image_source: str | Path, outputs_root: Path) -> RunPaths:
    from PIL import Image

    outputs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = outputs_root / f"run_{run_id}"
    images_dir = run_dir / "images"
    audio_dir = run_dir / "audio"
    video_dir = run_dir / "video"
    images_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(image_source)
    input_image_path = run_dir / "input_drawing.png"

    try:
        with Image.open(source_path) as image:
            image.convert("RGB").save(input_image_path)
    except Exception:
        shutil.copy2(source_path, input_image_path)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        input_image_path=input_image_path,
        description_path=run_dir / "description.json",
        description_prompt_path=run_dir / "description_prompt.txt",
        description_response_path=run_dir / "description_response.txt",
        story_path=run_dir / "story.json",
        story_prompt_path=run_dir / "story_prompt.txt",
        story_response_path=run_dir / "story_response.txt",
        timeline_path=run_dir / "timeline.json",
        narration_audio_path=audio_dir / "story_narration.wav",
        video_path=video_dir / "story_video.mp4",
        manifest_path=run_dir / "manifest.json",
        images_dir=images_dir,
        audio_dir=audio_dir,
        video_dir=video_dir,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def build_run_manifest(run_paths: RunPaths, story: StoryPackage) -> RunManifest:
    return RunManifest(
        run_id=run_paths.run_id,
        run_dir=str(run_paths.run_dir.resolve()),
        input_image_path=str(run_paths.input_image_path.resolve()),
        description_path=str(run_paths.description_path.resolve()),
        description_prompt_path=str(run_paths.description_prompt_path.resolve()),
        description_response_path=str(run_paths.description_response_path.resolve()),
        story_path=str(run_paths.story_path.resolve()),
        story_prompt_path=str(run_paths.story_prompt_path.resolve()),
        story_response_path=str(run_paths.story_response_path.resolve()),
        timeline_path=str(run_paths.timeline_path.resolve()),
        narration_audio_path=str(run_paths.narration_audio_path.resolve()),
        video_path=str(run_paths.video_path.resolve()),
        scene_image_paths=[str(Path(part.image_path).resolve()) for part in story.parts if part.image_path],
        part_audio_paths=[str(Path(part.audio_path).resolve()) for part in story.parts if part.audio_path],
    )
