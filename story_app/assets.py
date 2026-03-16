from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from .schemas import RunPaths


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_input_image(source_image_path: str | Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(source_image_path, destination_path)
    return destination_path


def prepare_run_paths(source_image_path: str | Path, outputs_root: Path) -> RunPaths:
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir = outputs_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(source_image_path).suffix or ".png"
    input_image_path = run_dir / f"input_image{suffix}"
    copy_input_image(source_image_path, input_image_path)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        input_image_path=input_image_path,
        description_prompt_path=run_dir / "description_prompt.txt",
        description_path=run_dir / "description.txt",
        story_part_1_anchor_path=run_dir / "story_part_1_anchor.txt",
        story_part_2_anchor_path=run_dir / "story_part_2_anchor.txt",
        story_part_3_anchor_path=run_dir / "story_part_3_anchor.txt",
        story_conversation_path=run_dir / "story_conversation.txt",
        story_part_1_path=run_dir / "story_part_1.txt",
        story_part_2_path=run_dir / "story_part_2.txt",
        story_part_3_path=run_dir / "story_part_3.txt",
        image_prompt_part_1_path=run_dir / "image_prompt_part_1.txt",
        image_prompt_part_2_path=run_dir / "image_prompt_part_2.txt",
        image_prompt_part_3_path=run_dir / "image_prompt_part_3.txt",
        story_part_1_image_path=run_dir / "story_part_1_image.png",
        story_part_2_image_path=run_dir / "story_part_2_image.png",
        story_part_3_image_path=run_dir / "story_part_3_image.png",
        story_part_1_audio_path=run_dir / "story_part_1_audio.wav",
        story_part_2_audio_path=run_dir / "story_part_2_audio.wav",
        story_part_3_audio_path=run_dir / "story_part_3_audio.wav",
        story_part_1_subtitle_path=run_dir / "story_part_1.ass",
        story_part_2_subtitle_path=run_dir / "story_part_2.ass",
        story_part_3_subtitle_path=run_dir / "story_part_3.ass",
        story_part_1_clip_path=run_dir / "story_part_1_clip.mp4",
        story_part_2_clip_path=run_dir / "story_part_2_clip.mp4",
        story_part_3_clip_path=run_dir / "story_part_3_clip.mp4",
        final_story_video_path=run_dir / "final_story_video.mp4",
        storyboard_manifest_path=run_dir / "storyboard_manifest.json",
    )
