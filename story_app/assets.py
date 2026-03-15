from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from .schemas import RunPaths


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        story_part_1_prompt_path=run_dir / "story_part_1_prompt.txt",
        story_part_1_path=run_dir / "story_part_1.txt",
        story_part_2_prompt_path=run_dir / "story_part_2_prompt.txt",
        story_part_2_path=run_dir / "story_part_2.txt",
        story_part_3_prompt_path=run_dir / "story_part_3_prompt.txt",
        story_part_3_path=run_dir / "story_part_3.txt",
    )
