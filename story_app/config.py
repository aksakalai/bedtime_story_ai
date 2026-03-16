from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_BUILD = "phase2-anchor-snapshots-v30-20260316"


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
        "Observe the uploaded scene and fill one compact anchor sheet for page 1 of a children's story. Treat the "
        "depicted content as a real scene, never as a drawing, picture, sketch, illustration, painting, or child "
        "art. Identify one central actor. If no clear actor exists, create one fitting general actor for the scene, "
        "such as a child, bunny, fish, duckling, fox, or similar simple character. Include explicit colors for the "
        "actor, the scene, and each important object whenever visible. Output exactly these lines in this exact "
        "order, using `none` when needed:\n"
        "actor:\n"
        "actor_colors:\n"
        "actor_traits:\n"
        "scene:\n"
        "scene_colors:\n"
        "object_1:\n"
        "object_1_colors:\n"
        "object_2:\n"
        "object_2_colors:\n"
        "object_3:\n"
        "object_3_colors:\n"
        "secondary_actor:\n"
        "secondary_actor_colors:\n"
        "page_event:\n"
        "mood:\n"
        "Reply only with the anchor sheet text."
    )
    initial_anchor_max_tokens: int = 224
    anchor_update_max_tokens: int = 224
    story_part_max_tokens: int = 384
    image_width: int = 1024
    image_height: int = 1024
    image_num_inference_steps: int = 20
    image_guidance_scale: float = 9.0
    image_prompt_style_suffix: str = "children's picture-book illustration"
    image_negative_prompt: str = (
        "low quality, blurry, muddy colors, flat lighting, deformed anatomy, extra limbs, duplicate characters, "
        "cropped face, text, letters, watermark, logo, signature, frame, border, collage, split panel"
    )
    image_prompt_summary_max_tokens: int = 256
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
    min_description_words: int = 0
    min_story_part_words: int = 0
    random_seed: int = 42
    app_build: str = APP_BUILD
    models: ModelIds = field(default_factory=ModelIds)


DEFAULT_CONFIG = GenerationConfig()
