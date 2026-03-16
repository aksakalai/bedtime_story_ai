from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase2-image-prompt-soft-guards-v21-20260316"


@dataclass(frozen=True)
class ModelIds:
    image_describer: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    story_writer: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    part_image_generator: str = "segmind/SSD-1B"
    part_narrator: str = "hexgrad/Kokoro-82M"
    word_aligner: str = "tiny.en"


@dataclass(frozen=True)
class GenerationConfig:
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    description_prompt_prefix: str = (
        "Describe the depicted scene itself in one concise paragraph. Include distinctive objects, colors, counts, "
        "positions, characters or animals if clearly visible, and other uniquely identifiable details. Use only "
        "visible scene facts. Do not mention the image, drawing, painting, paper, style, artist, or composition "
        "unless those are part of the depicted scene itself. Reply only with the description text."
    )
    description_max_tokens: int = 192
    story_part_max_tokens: int = 128
    image_width: int = 1024
    image_height: int = 1024
    image_num_inference_steps: int = 20
    image_guidance_scale: float = 9.0
    image_prompt_style_suffix: str = ""
    image_negative_prompt: str = (
        "low quality, blurry, muddy colors, flat lighting, deformed anatomy, extra limbs, duplicate characters, "
        "cropped face, text, letters, watermark, logo, signature, frame, border, collage, split panel"
    )
    image_prompt_summary_max_tokens: int = 76
    image_prompt_token_buffer: int = 1
    image_seed_stride: int = 1000
    narration_lang_code: str = "a"
    narration_voice: str = "af_heart"
    narration_speed: float = 1.0
    narration_sample_rate: int = 24000
    video_width: int = 1024
    video_height: int = 1024
    video_fps: int = 24
    video_tail_padding_seconds: float = 0.25
    overlay_text_width_ratio: float = 0.78
    overlay_box_width_ratio: float = 0.88
    overlay_box_bottom_margin_ratio: float = 0.032
    overlay_min_panel_height_ratio: float = 0.22
    overlay_max_panel_height_ratio: float = 0.34
    overlay_horizontal_padding_ratio: float = 0.045
    overlay_vertical_padding_ratio: float = 0.024
    overlay_min_font_size: int = 18
    overlay_max_font_size: int = 34
    overlay_line_spacing_ratio: float = 0.16
    overlay_font_name: str = "DejaVu Sans"
    overlay_panel_color_hex: str = "16202D"
    overlay_panel_opacity: float = 0.84
    overlay_panel_border_color_hex: str = "E6D2A8"
    overlay_panel_border_opacity: float = 0.36
    overlay_panel_border_thickness: int = 2
    overlay_panel_shadow_opacity: float = 0.18
    overlay_panel_shadow_offset: int = 8
    min_description_words: int = 10
    min_story_part_words: int = 1
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
