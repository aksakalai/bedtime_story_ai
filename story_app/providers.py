from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .config import GenerationConfig
from .json_utils import parse_json_response
from .prompts import build_description_prompt, build_story_prompt, enrich_story_with_image_prompts
from .schemas import DrawingDescription, StoryPackage, StoryPart


def _clear_torch_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _torch_dtype():
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32


def _clean_model_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = re.sub(r"^(assistant|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _parse_text_list(raw_text: str, max_items: int) -> list[str]:
    text = _clean_model_text(raw_text)
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    text = text.replace("\n", ",")
    parts = re.split(r"[,;|•]", text)
    items: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*[-*0-9.)]+\s*", "", part).strip(" .")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in {"none", "n/a", "not sure", "unknown"}:
            continue
        if cleaned not in items:
            items.append(cleaned)
        if len(items) >= max_items:
            break
    return items


def _parse_single_line(raw_text: str, fallback: str) -> str:
    text = _clean_model_text(raw_text)
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    line = next((line.strip(" -.") for line in text.splitlines() if line.strip()), "")
    return line or fallback


def _parse_summary_text(raw_text: str, fallback: str) -> str:
    text = _clean_model_text(raw_text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2]).strip()


class SmolVLMDescriber:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.config.models.drawing_describer)
        self.model = model_cls.from_pretrained(
            self.config.models.drawing_describer,
            dtype=_torch_dtype(),
        )
        self.model.to(self.device)

    def _generate_text(self, image_path: str | Path, prompt: str, max_new_tokens: int) -> str:
        self._load()
        assert self.processor is not None
        assert self.model is not None

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=rendered_prompt,
                images=[image],
                return_tensors="pt",
            )

        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_length = model_inputs["input_ids"].shape[1]
        completion = generated_ids[:, prompt_length:]
        return self.processor.batch_decode(completion, skip_special_tokens=True)[0]

    def _describe_fieldwise(self, image_path: str | Path) -> DrawingDescription:
        summary = _parse_summary_text(
            self._generate_text(
                image_path,
                (
                    "Describe this child's drawing in 1 or 2 gentle bedtime-story sentences. "
                    "Mention the main character, setting, mood, and key visual details."
                ),
                max_new_tokens=120,
            ),
            fallback="A child-friendly drawing with a gentle bedtime-story mood.",
        )
        characters = _parse_text_list(
            self._generate_text(
                image_path,
                "List the main characters or important objects as a short comma-separated list. Max 5 items.",
                max_new_tokens=60,
            ),
            max_items=5,
        ) or ["main child-drawn character"]
        setting = _parse_single_line(
            self._generate_text(
                image_path,
                "Describe the setting in one short phrase.",
                max_new_tokens=40,
            ),
            fallback="a cozy imaginative setting",
        )
        visual_style = _parse_single_line(
            self._generate_text(
                image_path,
                "Describe the art style or medium in one short phrase.",
                max_new_tokens=40,
            ),
            fallback="child-made storybook drawing",
        )
        color_palette = _parse_text_list(
            self._generate_text(
                image_path,
                "List the main colors in the drawing as a short comma-separated list. Max 6 items.",
                max_new_tokens=40,
            ),
            max_items=6,
        ) or ["soft mixed colors"]
        safety_notes = _parse_text_list(
            self._generate_text(
                image_path,
                (
                    "Give 3 short bedtime-safety guidance notes for turning this drawing into a story. "
                    "Use a comma-separated list."
                ),
                max_new_tokens=60,
            ),
            max_items=3,
        ) or [
            "keep the tone gentle",
            "avoid scary or intense conflict",
            "end with calm reassurance",
        ]

        return DrawingDescription(
            summary=summary,
            characters=characters,
            setting=setting,
            visual_style=visual_style,
            color_palette=color_palette,
            safety_notes=safety_notes,
        )

    def describe(self, image_path: str | Path, attempt_index: int = 0) -> DrawingDescription:
        prompt = build_description_prompt(self.config)
        if attempt_index:
            prompt += "\nPrevious output was invalid. Return only valid JSON with the required keys."

        raw_text = self._generate_text(
            image_path,
            prompt,
            max_new_tokens=self.config.description_max_tokens,
        )
        try:
            return parse_json_response(raw_text, DrawingDescription)
        except Exception:
            return self._describe_fieldwise(image_path)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        _clear_torch_memory()


class QwenStoryWriter:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.input_device = "cpu"

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.models.story_writer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.models.story_writer,
            **model_kwargs,
        )
        try:
            self.input_device = str(next(self.model.parameters()).device)
        except StopIteration:
            self.input_device = "cpu"

    def write_story(self, description: DrawingDescription, attempt_index: int = 0) -> StoryPackage:
        self._load()
        assert self.tokenizer is not None
        assert self.model is not None

        prompt = build_story_prompt(description, self.config)
        if attempt_index:
            prompt += "\nThe previous response was invalid. Return one valid JSON object only."

        messages = [
            {
                "role": "system",
                "content": "You are a careful bedtime story writer who follows output schemas exactly.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.story_max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        raw_text = self.tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        story = parse_json_response(raw_text, StoryPackage)
        return enrich_story_with_image_prompts(story, description, self.config)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        _clear_torch_memory()


class SSD1BSceneGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipeline: Any | None = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.pipeline is not None:
            return

        import torch
        from diffusers import StableDiffusionXLPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(),
            "use_safetensors": True,
        }
        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.models.scene_generator,
            **load_kwargs,
        )
        self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "vae") and hasattr(self.pipeline.vae, "enable_slicing"):
            self.pipeline.vae.enable_slicing()

        if self.device == "cuda":
            self.pipeline = self.pipeline.to(self.device)
        else:
            self.pipeline = self.pipeline.to("cpu")

    def generate(self, story: StoryPackage, images_dir: Path) -> StoryPackage:
        self._load()
        assert self.pipeline is not None

        import torch

        updated_parts: list[StoryPart] = []
        for index, part in enumerate(story.parts, start=1):
            generator = None
            if self.device == "cuda":
                generator = torch.Generator(device=self.device).manual_seed(self.config.random_seed + index)

            result = self.pipeline(
                prompt=part.image_prompt,
                negative_prompt=self.config.image_negative_prompt,
                width=self.config.image_width,
                height=self.config.image_height,
                num_inference_steps=self.config.diffusion_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )
            image_path = images_dir / f"scene_{index}.png"
            result.images[0].save(image_path)
            updated_parts.append(
                StoryPart(
                    scene_goal=part.scene_goal,
                    story_text=part.story_text,
                    image_prompt=part.image_prompt,
                    image_path=str(image_path.resolve()),
                    audio_path=part.audio_path,
                    duration_sec=part.duration_sec,
                )
            )
        return StoryPackage(title=story.title, age_range=story.age_range, parts=updated_parts)

    def unload(self) -> None:
        self.pipeline = None
        _clear_torch_memory()


class KokoroNarrator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipeline: Any | None = None

    def _load(self) -> None:
        if self.pipeline is not None:
            return

        from kokoro import KPipeline

        # KPipeline wraps the Kokoro-82M voice model and downloads assets as needed.
        self.pipeline = KPipeline(lang_code="a")

    def narrate(self, story: StoryPackage, audio_dir: Path, merged_audio_path: Path) -> StoryPackage:
        self._load()
        assert self.pipeline is not None

        import numpy as np
        import soundfile as sf

        sample_rate = 24000
        merged_segments: list[Any] = []
        updated_parts: list[StoryPart] = []

        for index, part in enumerate(story.parts, start=1):
            generator = self.pipeline(
                part.story_text,
                voice=self.config.narrator_voice,
                speed=self.config.narrator_speed,
                split_pattern=r"\n+",
            )
            chunks = [segment_audio for _, _, segment_audio in generator]
            if not chunks:
                raise RuntimeError(f"Kokoro returned no audio for part {index}.")

            audio = np.concatenate(chunks).astype("float32")
            audio_path = audio_dir / f"part_{index}.wav"
            sf.write(audio_path, audio, sample_rate)
            duration_sec = len(audio) / sample_rate
            merged_segments.append(audio)
            updated_parts.append(
                StoryPart(
                    scene_goal=part.scene_goal,
                    story_text=part.story_text,
                    image_prompt=part.image_prompt,
                    image_path=part.image_path,
                    audio_path=str(audio_path.resolve()),
                    duration_sec=round(duration_sec, 3),
                )
            )

        merged_audio = np.concatenate(merged_segments).astype("float32")
        sf.write(merged_audio_path, merged_audio, sample_rate)
        return StoryPackage(title=story.title, age_range=story.age_range, parts=updated_parts)
