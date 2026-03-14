from __future__ import annotations

import shutil
import subprocess
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schemas import StoryPackage, TimelineManifest

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 12


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_words(
    draw: ImageDraw.ImageDraw,
    words: list[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> list[list[str]]:
    lines: list[list[str]] = []
    current_line: list[str] = []
    for word in words:
        trial = " ".join(current_line + [word])
        if not current_line or draw.textlength(trial, font=font) <= max_width:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
    if current_line:
        lines.append(current_line)
    return lines


def _draw_highlighted_text(
    draw: ImageDraw.ImageDraw,
    story_text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    origin_x: int,
    origin_y: int,
    max_width: int,
    highlighted_words: int,
) -> None:
    words = story_text.split()
    lines = _wrap_words(draw, words, font, max_width)
    highlighted_remaining = highlighted_words
    y = origin_y
    line_height = int(getattr(font, "size", 30) * 1.55)

    for line_words in lines:
        x = origin_x
        for word in line_words:
            word_text = f"{word} "
            is_highlighted = highlighted_remaining > 0
            bbox = draw.textbbox((x, y), word_text, font=font)
            if is_highlighted:
                draw.rounded_rectangle(
                    (
                        bbox[0] - 6,
                        bbox[1] - 4,
                        bbox[2] + 4,
                        bbox[3] + 3,
                    ),
                    radius=10,
                    fill=(255, 215, 150, 190),
                )
            draw.text(
                (x, y),
                word_text,
                font=font,
                fill=(40, 30, 18) if is_highlighted else (205, 214, 229),
            )
            x += int(draw.textlength(word_text, font=font))
            if highlighted_remaining > 0:
                highlighted_remaining -= 1
        y += line_height


def _render_story_frame(
    image_path: str,
    title: str,
    story_text: str,
    part_label: str,
    highlighted_words: int,
    progress_ratio: float,
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
    label_font = _load_font(22)
    panel_height = 268
    panel_top = VIDEO_HEIGHT - panel_height - 32
    panel_rect = (40, panel_top, VIDEO_WIDTH - 40, VIDEO_HEIGHT - 32)
    draw.rounded_rectangle(panel_rect, radius=28, fill=(14, 20, 32, 190))
    draw.text((72, panel_top + 22), title, fill=(255, 246, 235), font=title_font)
    draw.text((72, panel_top + 74), part_label, fill=(255, 215, 150), font=label_font)

    _draw_highlighted_text(
        draw=draw,
        story_text=story_text,
        font=body_font,
        origin_x=72,
        origin_y=panel_top + 114,
        max_width=VIDEO_WIDTH - 144,
        highlighted_words=highlighted_words,
    )

    progress_width = int((VIDEO_WIDTH - 144) * max(0.0, min(progress_ratio, 1.0)))
    progress_y = VIDEO_HEIGHT - 58
    draw.rounded_rectangle((72, progress_y, VIDEO_WIDTH - 72, progress_y + 12), radius=6, fill=(77, 89, 109, 180))
    draw.rounded_rectangle((72, progress_y, 72 + progress_width, progress_y + 12), radius=6, fill=(255, 215, 150, 255))

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
        story_words = segment.story_text.split()
        print(
            f"[video] Segment {segment.index}: duration={duration:.2f}s, "
            f"frames={segment_frame_count}, words={len(story_words)}, image={segment.image_path}"
        )
        for local_frame_index in range(segment_frame_count):
            progress_ratio = (local_frame_index + 1) / segment_frame_count
            highlighted_words = max(1, ceil(progress_ratio * max(len(story_words), 1)))
            sequence_frame = work_dir / f"frame_{frame_index:05d}.png"
            _render_story_frame(
                image_path=segment.image_path,
                title=story.title,
                story_text=segment.story_text,
                part_label=f"Part {segment.index} of {len(timeline.segments)}",
                highlighted_words=highlighted_words,
                progress_ratio=progress_ratio,
                output_path=sequence_frame,
            )
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
