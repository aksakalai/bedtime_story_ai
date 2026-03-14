from __future__ import annotations

import shutil
import subprocess
import textwrap
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schemas import StoryPackage, TimelineManifest

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 4


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_story_frame(
    image_path: str,
    title: str,
    story_text: str,
    output_path: Path,
) -> None:
    with Image.open(image_path) as source_image:
        background = ImageOps.fit(
            source_image.convert("RGB"),
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )

    overlay = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = _load_font(46)
    body_font = _load_font(30)
    panel_height = 240
    panel_top = VIDEO_HEIGHT - panel_height - 32
    panel_rect = (40, panel_top, VIDEO_WIDTH - 40, VIDEO_HEIGHT - 32)
    draw.rounded_rectangle(panel_rect, radius=28, fill=(14, 20, 32, 190))
    draw.text((72, panel_top + 26), title, fill=(255, 246, 235), font=title_font)

    wrapped_text = textwrap.fill(story_text, width=54)
    draw.multiline_text(
        (72, panel_top + 100),
        wrapped_text,
        fill=(255, 255, 255),
        font=body_font,
        spacing=10,
    )

    combined = Image.alpha_composite(background.convert("RGBA"), overlay)
    combined.convert("RGB").save(output_path)


def render_story_video(
    story: StoryPackage,
    timeline: TimelineManifest,
    output_path: Path,
    work_dir: Path,
) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to render the story video but was not found on PATH.")

    work_dir.mkdir(parents=True, exist_ok=True)
    for path in work_dir.glob("frame_*.png"):
        path.unlink()
    for path in work_dir.glob("segment_*.png"):
        path.unlink()

    frame_index = 0
    for segment in timeline.segments:
        duration = max(segment.end_sec - segment.start_sec, 0.8)
        segment_frame_count = max(1, ceil(duration * VIDEO_FPS))
        rendered_frame = work_dir / f"segment_{segment.index}.png"
        _render_story_frame(
            image_path=segment.image_path,
            title=story.title,
            story_text=segment.story_text,
            output_path=rendered_frame,
        )
        print(
            f"[video] Segment {segment.index}: duration={duration:.2f}s, "
            f"frames={segment_frame_count}, image={segment.image_path}"
        )
        with Image.open(rendered_frame) as frame_image:
            frame_image.load()
            for _ in range(segment_frame_count):
                sequence_frame = work_dir / f"frame_{frame_index:05d}.png"
                frame_image.save(sequence_frame)
                frame_index += 1

    if frame_index == 0:
        raise RuntimeError("No video frames were rendered for the story output.")
    print(f"[video] Total rendered frames: {frame_index}")

    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        str(work_dir / "frame_%05d.png"),
        "-i",
        timeline.audio_path,
    ]

    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(VIDEO_FPS),
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to render the story video.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return output_path
