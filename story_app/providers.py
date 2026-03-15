from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

from .assets import write_text
from .config import GenerationConfig
from .prompts import (
    build_description_prompt,
    build_story_prompt,
    enrich_story_with_image_prompts,
    parse_description_response,
    parse_story_response,
)
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


def _preview_text(raw_text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


class BlipDrawingDescriber:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[describe] Loading model: {self.config.models.drawing_describer} on {self.device}")
        self.processor = BlipProcessor.from_pretrained(self.config.models.drawing_describer)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.config.models.drawing_describer,
            torch_dtype=_torch_dtype(),
        )
        self.model.to(self.device)

    def _generate_text(self, image_path: str | Path, prompt: str, max_new_tokens: int) -> str:
        from PIL import Image

        self._load()
        assert self.processor is not None
        assert self.model is not None

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            model_inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            prepared_inputs: dict[str, Any] = {}
            for key, value in model_inputs.items():
                if getattr(value, "dtype", None) is not None and value.dtype.is_floating_point:
                    prepared_inputs[key] = value.to(self.device, dtype=_torch_dtype())
                else:
                    prepared_inputs[key] = value.to(self.device)

            generated_ids = self.model.generate(
                **prepared_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
            )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def describe(
        self,
        image_path: str | Path,
        prompt_path: Path | None = None,
        response_path: Path | None = None,
    ) -> DrawingDescription:
        prompt = build_description_prompt(self.config)
        if prompt_path is not None:
            write_text(prompt_path, prompt)
            print(f"[describe] Prompt saved: {prompt_path}")
        print(f"[describe] Prompt task: {prompt}")

        raw_text = self._generate_text(
            image_path,
            prompt,
            max_new_tokens=self.config.description_max_tokens,
        )
        if response_path is not None:
            write_text(response_path, raw_text)
            print(f"[describe] Response saved: {response_path}")

        print(f"[describe] Raw preview: {_preview_text(raw_text)}")
        description = parse_description_response(raw_text, self.config)
        print(f"[describe] Parsed description word count: {len(description.text.split())}")
        return description

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

        print(f"[story] Loading model: {self.config.models.story_writer}")
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

    def write_story(
        self,
        description: DrawingDescription,
        prompt_path: Path | None = None,
        response_path: Path | None = None,
    ) -> StoryPackage:
        self._load()
        assert self.tokenizer is not None
        assert self.model is not None

        prompt = build_story_prompt(description, self.config)
        if prompt_path is not None:
            write_text(prompt_path, prompt)
            print(f"[story] Prompt saved: {prompt_path}")
        print(f"[story] Prompt separator token: {self.config.story_separator_token}")
        print(f"[story] Drawing description preview: {_preview_text(description.text, limit=240)}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You follow output formats exactly. Use the PART_BREAK token exactly as requested "
                    "and do not emit JSON or extra commentary."
                ),
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
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        raw_text = self.tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        if response_path is not None:
            write_text(response_path, raw_text)
            print(f"[story] Response saved: {response_path}")

        print(f"[story] Raw preview: {_preview_text(raw_text)}")
        print(f"[story] Raw separator count: {raw_text.count(self.config.story_separator_token)}")
        story = parse_story_response(raw_text, self.config)
        print(f"[story] Parsed title: {story.title}")
        print(f"[story] Parsed part word counts: {[len(part.story_text.split()) for part in story.parts]}")
        return enrich_story_with_image_prompts(story, description, self.config)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        _clear_torch_memory()


class SDTurboSceneGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipeline: Any | None = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.pipeline is not None:
            return

        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(),
        }
        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"
            load_kwargs["use_safetensors"] = True

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self.config.models.scene_generator,
            **load_kwargs,
        )
        self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "vae") and hasattr(self.pipeline.vae, "enable_slicing"):
            self.pipeline.vae.enable_slicing()
        if self.device == "cuda":
            self.pipeline = self.pipeline.to(self.device)

    def generate(self, story: StoryPackage, images_dir: Path) -> StoryPackage:
        self._load()
        assert self.pipeline is not None

        import torch

        if len(story.parts) != 3:
            raise RuntimeError("Scene generation requires exactly 3 story parts.")

        updated_parts: list[StoryPart] = []
        for index, part in enumerate(story.parts, start=1):
            generator = torch.Generator(device=self.device).manual_seed(
                self.config.random_seed + index
            )

            result = self.pipeline(
                prompt=part.image_prompt,
                width=self.config.image_width,
                height=self.config.image_height,
                num_inference_steps=self.config.diffusion_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )
            print(f"[images] Scene {index} prompt preview: {_preview_text(part.image_prompt, limit=240)}")
            image_path = images_dir / f"scene_{index}.png"
            result.images[0].save(image_path)
            print(f"[images] Scene {index} saved: {image_path}")
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

        self.pipeline = KPipeline(lang_code="a")

    def narrate(self, story: StoryPackage, audio_dir: Path, merged_audio_path: Path) -> StoryPackage:
        self._load()
        assert self.pipeline is not None

        import numpy as np
        import soundfile as sf

        if len(story.parts) != 3:
            raise RuntimeError("Narration requires exactly 3 story parts.")

        sample_rate = 24000
        merged_segments: list[Any] = []
        updated_parts: list[StoryPart] = []

        for index, part in enumerate(story.parts, start=1):
            print(f"[narration] Part {index} text word count: {len(part.story_text.split())}")
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
            print(f"[narration] Part {index} saved: {audio_path} ({duration_sec:.2f}s)")
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
            merged_segments.append(audio)

        merged_audio = np.concatenate(merged_segments).astype("float32")
        sf.write(merged_audio_path, merged_audio, sample_rate)
        print(f"[narration] Merged audio saved: {merged_audio_path}")
        return StoryPackage(title=story.title, age_range=story.age_range, parts=updated_parts)
