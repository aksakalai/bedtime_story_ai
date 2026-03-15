from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont

from .config import GenerationConfig
from .schemas import ValidationError


@dataclass(frozen=True)
class TimedToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class OverlayLayout:
    font_name: str
    font_size: int
    panel_height: int
    margin_l: int
    margin_r: int
    margin_v: int
    line_height: int
    lines: list[list[TimedToken]]


def _normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower())


def _tokenize_display_text(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _resolve_story_font(config: GenerationConfig) -> tuple[str, str | None]:
    candidates = [
        (config.overlay_font_name, "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ("Liberation Serif", "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        ("Georgia", "C:\\Windows\\Fonts\\georgia.ttf"),
        ("Times New Roman", "C:\\Windows\\Fonts\\times.ttf"),
    ]
    for font_name, font_path in candidates:
        if Path(font_path).exists():
            return font_name, font_path
    return config.overlay_font_name, None


def _load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _measure_text(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_tokens(
    tokens: list[TimedToken],
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[list[TimedToken]]:
    lines: list[list[TimedToken]] = []
    current_line: list[TimedToken] = []
    for token in tokens:
        candidate_tokens = [*current_line, token]
        candidate_text = " ".join(item.text for item in candidate_tokens)
        candidate_width, _ = _measure_text(font, candidate_text)
        if current_line and candidate_width > max_width:
            lines.append(current_line)
            current_line = [token]
            continue
        current_line = candidate_tokens
    if current_line:
        lines.append(current_line)
    return lines


def fit_overlay_layout(tokens: list[TimedToken], config: GenerationConfig) -> OverlayLayout:
    font_name, font_path = _resolve_story_font(config)
    max_width = int(config.video_width * config.overlay_text_width_ratio)
    margin_h = int(config.video_width * config.overlay_horizontal_padding_ratio)
    padding_v = int(config.video_height * config.overlay_vertical_padding_ratio)
    min_panel_height = int(config.video_height * config.overlay_min_panel_height_ratio)
    max_panel_height = int(config.video_height * config.overlay_max_panel_height_ratio)
    best_layout: OverlayLayout | None = None

    for font_size in range(config.overlay_max_font_size, config.overlay_min_font_size - 1, -2):
        font = _load_font(font_path, font_size)
        lines = _wrap_tokens(tokens, font=font, max_width=max_width)
        _, sample_height = _measure_text(font, "Ag")
        line_height = max(sample_height, font_size)
        line_spacing = int(line_height * config.overlay_line_spacing_ratio)
        text_height = (len(lines) * line_height) + (max(0, len(lines) - 1) * line_spacing)
        panel_height = max(min_panel_height, text_height + (padding_v * 2))
        if panel_height > max_panel_height:
            continue
        best_layout = OverlayLayout(
            font_name=font_name,
            font_size=font_size,
            panel_height=panel_height,
            margin_l=margin_h,
            margin_r=margin_h,
            margin_v=padding_v,
            line_height=line_height,
            lines=lines,
        )
        break

    if best_layout is not None:
        return best_layout

    font = _load_font(font_path, config.overlay_min_font_size)
    lines = _wrap_tokens(tokens, font=font, max_width=max_width)
    _, sample_height = _measure_text(font, "Ag")
    line_height = max(sample_height, config.overlay_min_font_size)
    panel_height = max_panel_height
    return OverlayLayout(
        font_name=font_name,
        font_size=config.overlay_min_font_size,
        panel_height=panel_height,
        margin_l=margin_h,
        margin_r=margin_h,
        margin_v=padding_v,
        line_height=line_height,
        lines=lines,
    )


def align_story_text_to_timestamps(
    *,
    display_text: str,
    whisper_words: list[dict[str, float | str]],
    fallback_total_duration: float,
) -> list[TimedToken]:
    tokens = _tokenize_display_text(display_text)
    aligned: list[TimedToken] = []
    whisper_index = 0

    for token in tokens:
        normalized = _normalize_word(token)
        if not normalized:
            aligned.append(TimedToken(text=token, start=0.0, end=0.01))
            continue

        matched_word: dict[str, float | str] | None = None
        for candidate_index in range(whisper_index, min(whisper_index + 4, len(whisper_words))):
            candidate_word = whisper_words[candidate_index]
            candidate_normalized = _normalize_word(str(candidate_word.get("word", "")))
            if candidate_normalized == normalized or candidate_normalized in normalized or normalized in candidate_normalized:
                matched_word = candidate_word
                whisper_index = candidate_index + 1
                break

        if matched_word is None and whisper_index < len(whisper_words):
            matched_word = whisper_words[whisper_index]
            whisper_index += 1

        if matched_word is None:
            aligned.append(TimedToken(text=token, start=0.0, end=0.0))
            continue

        start = float(matched_word.get("start", 0.0) or 0.0)
        end = float(matched_word.get("end", start) or start)
        if end <= start:
            end = start + 0.01
        aligned.append(TimedToken(text=token, start=start, end=end))

    if not aligned:
        raise ValidationError("Could not align any story words to timestamps.")

    missing_indexes = [index for index, token in enumerate(aligned) if token.end <= token.start]
    if missing_indexes:
        estimated_step = fallback_total_duration / max(1, len(aligned))
        rebuilt: list[TimedToken] = []
        for index, token in enumerate(aligned):
            if token.end > token.start:
                rebuilt.append(token)
                continue
            start = estimated_step * index
            end = min(fallback_total_duration, start + estimated_step)
            rebuilt.append(TimedToken(text=token.text, start=start, end=max(end, start + 0.01)))
        aligned = rebuilt

    return aligned


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", "(").replace("}", ")")


def _format_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_color_from_rgb_hex(rgb_hex: str) -> str:
    rgb_hex = rgb_hex.lstrip("#")
    if len(rgb_hex) != 6:
        raise ValidationError(f"Expected a 6-digit RGB hex color, got `{rgb_hex}`.")
    red = rgb_hex[0:2]
    green = rgb_hex[2:4]
    blue = rgb_hex[4:6]
    return f"&H00{blue}{green}{red}"


def build_story_part_ass(
    *,
    timed_tokens: list[TimedToken],
    total_duration_seconds: float,
    config: GenerationConfig,
) -> tuple[str, int]:
    layout = fit_overlay_layout(timed_tokens, config)
    primary_color = _ass_color_from_rgb_hex("F4C46A")
    secondary_color = _ass_color_from_rgb_hex("FFF7EA")
    outline_color = _ass_color_from_rgb_hex("1A0F0A")
    back_color = _ass_color_from_rgb_hex("000000")
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {config.video_width}",
        f"PlayResY: {config.video_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Story,"
        f"{layout.font_name},{layout.font_size},{primary_color},{secondary_color},{outline_color},{back_color},"
        "0,0,0,0,100,100,0,0,1,2.6,0.8,2,"
        f"{layout.margin_l},{layout.margin_r},{layout.margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    text_parts: list[str] = []
    cursor_seconds = 0.0
    for line_index, line_tokens in enumerate(layout.lines):
        if line_index:
            text_parts.append(r"\N")
        for token_index, token in enumerate(line_tokens):
            if token_index:
                text_parts.append(" ")
            gap_cs = max(0, round((token.start - cursor_seconds) * 100))
            if gap_cs:
                text_parts.append(f"{{\\k{gap_cs}}}")
            duration_cs = max(1, round((token.end - token.start) * 100))
            text_parts.append(f"{{\\kf{duration_cs}}}{_escape_ass_text(token.text)}")
            cursor_seconds = max(cursor_seconds, token.end)

    dialogue_text = "".join(text_parts)
    event_end = total_duration_seconds + config.video_tail_padding_seconds
    lines.append(
        "Dialogue: 0,"
        f"{_format_ass_time(0.0)},{_format_ass_time(event_end)},Story,,0,0,0,,{dialogue_text}"
    )
    return "\n".join(lines) + "\n", layout.panel_height


def _escape_subtitles_path(path: str | Path) -> str:
    value = str(Path(path).resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


class FFmpegVideoAssembler:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def _ensure_ffmpeg(self) -> str:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValidationError(
                "ffmpeg is required for video assembly. In Colab it is usually already installed; "
                "if not, run `!apt-get -qq -y install ffmpeg` and restart the session."
            )
        return executable

    def _run_ffmpeg(self, arguments: list[str]) -> None:
        executable = self._ensure_ffmpeg()
        command = [executable, "-y", *arguments]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValidationError(
                "ffmpeg video assembly failed.\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

    def render_story_part_clip(
        self,
        *,
        image_path: str | Path,
        audio_path: str | Path,
        subtitle_path: str | Path,
        panel_height: int,
        total_duration_seconds: float,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel_top = self.config.video_height - panel_height
        filter_graph = (
            f"scale={self.config.video_width}:{self.config.video_height}:force_original_aspect_ratio=cover,"
            f"crop={self.config.video_width}:{self.config.video_height},"
            f"drawbox=x=0:y={panel_top}:w=iw:h={panel_height}:"
            f"color=0x{self.config.overlay_panel_color_hex}@{self.config.overlay_panel_opacity}:t=fill,"
            f"subtitles='{_escape_subtitles_path(subtitle_path)}'"
        )
        clip_duration = total_duration_seconds + self.config.video_tail_padding_seconds
        self._run_ffmpeg(
            [
                "-loop",
                "1",
                "-framerate",
                str(self.config.video_fps),
                "-i",
                str(Path(image_path).resolve()),
                "-i",
                str(Path(audio_path).resolve()),
                "-vf",
                filter_graph,
                "-af",
                f"apad=pad_dur={self.config.video_tail_padding_seconds}",
                "-t",
                f"{clip_duration:.3f}",
                "-r",
                str(self.config.video_fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path.resolve()),
            ]
        )
        return output_path

    def concatenate_story_clips(
        self,
        *,
        clip_paths: list[str | Path],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            concat_file = Path(handle.name)
            for clip_path in clip_paths:
                handle.write(f"file '{Path(clip_path).resolve().as_posix()}'\n")

        try:
            self._run_ffmpeg(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file.resolve()),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(output_path.resolve()),
                ]
            )
        finally:
            concat_file.unlink(missing_ok=True)

        return output_path
