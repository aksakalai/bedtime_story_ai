from __future__ import annotations

import builtins
import gc
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from .config import GenerationConfig
from .schemas import ValidationError

_MODEL_CACHE_ATTR = "_bedtime_story_ai_model_cache"


def _clear_torch_memory() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _get_model_cache() -> dict[tuple[str, str, str], dict[str, Any]]:
    cache = getattr(builtins, _MODEL_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(builtins, _MODEL_CACHE_ATTR, cache)
    return cache


def clear_cached_models() -> None:
    cache = _get_model_cache()
    cache.clear()
    _clear_torch_memory()


def count_image_placeholders(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
    return count


def _build_model_kwargs() -> dict[str, Any]:
    import torch

    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
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
    return model_kwargs


class Qwen25VLMultimodalEngine:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.processor: Any | None = None
        self.model: Any | None = None

    def _resolve_model_id(self) -> str:
        return self.config.models.image_describer

    def _cache_key(self) -> tuple[str, str, str]:
        return ("multimodal_engine", self._resolve_model_id(), self.device)

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = self._cache_key()
        cached = _get_model_cache().get(cache_key)
        if cached is not None:
            print(f"[multimodal] Reusing cached model: {self._resolve_model_id()} on {self.device}")
            self.processor = cached["processor"]
            self.model = cached["model"]
            return

        print(f"[multimodal] Loading model: {self._resolve_model_id()} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(self._resolve_model_id())
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._resolve_model_id(),
            **_build_model_kwargs(),
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        _get_model_cache()[cache_key] = {
            "processor": self.processor,
            "model": self.model,
        }

    def _generate_from_messages(
        self,
        *,
        image_path: str | Path | None,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        log_prefix: str,
        error_message: str,
    ) -> str:
        self._load()
        assert self.processor is not None
        assert self.model is not None

        image_slots = count_image_placeholders(messages)
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images: list[Any] = []
        if image_slots:
            if image_path is None:
                raise ValidationError("Image-backed generation requires an input image path.")
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                images = [image.copy() for _ in range(image_slots)]

        if image_slots:
            model_inputs = self.processor(
                text=[rendered_prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        else:
            model_inputs = self.processor(
                text=[rendered_prompt],
                padding=True,
                return_tensors="pt",
            )

        prepared_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        prompt_token_count = int(prepared_inputs["input_ids"].shape[1])
        print(f"[{log_prefix}] Prompt token count: {prompt_token_count}")

        eos_token_id = None
        if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "eos_token_id", None) is not None:
            eos_token_id = int(self.processor.tokenizer.eos_token_id)

        generated_ids = self.model.generate(
            **prepared_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
        prompt_length = prepared_inputs["input_ids"].shape[1]
        completion = generated_ids[:, prompt_length:]
        generated_token_count = int(completion.shape[1])
        hit_token_cap = generated_token_count >= max_new_tokens
        ended_with_eos = bool(
            generated_token_count
            and eos_token_id is not None
            and int(completion[0, -1].item()) == eos_token_id
        )
        decoded = self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        print(
            f"[{log_prefix}] Generation stats: "
            f"generated_tokens={generated_token_count}, "
            f"max_new_tokens={max_new_tokens}, "
            f"hit_token_cap={hit_token_cap}, "
            f"ended_with_eos={ended_with_eos}"
        )
        print(f"[{log_prefix}] Output word count: {len(decoded.split())}")
        preview = decoded[:240] if log_prefix == "describe" else decoded[-120:]
        label = "Output preview" if log_prefix == "describe" else "Output tail preview"
        print(f"[{log_prefix}] {label}: {preview}")
        if not ended_with_eos:
            raise ValidationError(error_message)
        return decoded

    def describe(self, image_path: str | Path, messages: list[dict[str, Any]]) -> str:
        return self._generate_from_messages(
            image_path=image_path,
            messages=messages,
            max_new_tokens=self.config.description_max_tokens,
            log_prefix="describe",
            error_message=(
                "Description generation did not finish naturally before the safety limit. "
                "Increase the description token ceiling or tighten the prompt."
            ),
        )

    def generate_part(self, image_path: str | Path, messages: list[dict[str, Any]]) -> str:
        return self._generate_from_messages(
            image_path=image_path,
            messages=messages,
            max_new_tokens=self.config.story_part_max_tokens,
            log_prefix="story",
            error_message=(
                "Story generation did not finish naturally before the safety limit. "
                "Increase the story token ceiling or tighten the prompt."
            ),
        )

    def generate_continuity_brief(self, messages: list[dict[str, Any]]) -> str:
        return self._generate_from_messages(
            image_path=None,
            messages=messages,
            max_new_tokens=self.config.story_part_max_tokens,
            log_prefix="continuity",
            error_message=(
                "Continuity brief generation did not finish naturally before the safety limit. "
                "Increase the continuity token ceiling or tighten the prompt."
            ),
        )

    def generate_image_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        return self._generate_from_messages(
            image_path=None,
            messages=messages,
            max_new_tokens=max_new_tokens or self.config.image_prompt_summary_max_tokens,
            log_prefix="image_prompt",
            error_message=(
                "Image prompt summary generation did not finish naturally before the safety limit. "
                "Tighten the summary prompt or lower the requested detail."
            ),
        )

    def unload(self, clear_cache: bool = False) -> None:
        if clear_cache:
            _get_model_cache().pop(self._cache_key(), None)
            self.processor = None
            self.model = None
            _clear_torch_memory()
            return

        self.processor = None
        self.model = None


Qwen2VLImageDescriber = Qwen25VLMultimodalEngine
QwenStoryWriter = Qwen25VLMultimodalEngine


class SSD1BTextToImageGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.pipeline: Any | None = None

    def _resolve_model_id(self) -> str:
        return self.config.models.part_image_generator

    def _cache_key(self) -> tuple[str, str, str]:
        return ("image_generator", self._resolve_model_id(), self.device)

    def _load(self) -> None:
        if self.pipeline is not None:
            return

        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = self._cache_key()
        cached = _get_model_cache().get(cache_key)
        if cached is not None:
            print(f"[image] Reusing cached model: {self._resolve_model_id()} on {self.device}")
            self.pipeline = cached["pipeline"]
            return

        print(f"[image] Loading model: {self._resolve_model_id()} on {self.device}")
        if self.device == "cuda":
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self._resolve_model_id(),
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self._resolve_model_id(),
                use_safetensors=True,
            )

        self.pipeline.to(self.device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
        _get_model_cache()[cache_key] = {"pipeline": self.pipeline}

    def generate(
        self,
        *,
        prompt_text: str,
        negative_prompt_text: str,
        seed: int,
        output_path: str | Path,
    ) -> Path:
        self._load()
        assert self.pipeline is not None

        import torch

        if self.device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        result = self.pipeline(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            width=self.config.image_width,
            height=self.config.image_height,
            num_inference_steps=self.config.image_num_inference_steps,
            guidance_scale=self.config.image_guidance_scale,
            generator=generator,
        )
        generated_image = result.images[0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_image.save(output_path)
        print(f"[image] Saved image to {output_path}")
        return output_path

    def _count_tokens(self, tokenizer: Any, text: str) -> int:
        encoded = tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)

    def get_prompt_token_limit(self, *, buffer_tokens: int = 0) -> int:
        self._load()
        assert self.pipeline is not None

        max_lengths: list[int] = []
        for tokenizer_name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self.pipeline, tokenizer_name, None)
            if tokenizer is None:
                continue
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if isinstance(model_max_length, int) and model_max_length > 0:
                max_lengths.append(model_max_length)

        if not max_lengths:
            raise ValidationError("Could not determine the current image model token limit.")

        return min(max_lengths) - buffer_tokens

    def validate_prompt_token_budget(
        self,
        prompt_text: str,
        *,
        buffer_tokens: int = 0,
        strict: bool = True,
    ) -> dict[str, int]:
        self._load()
        assert self.pipeline is not None

        counts: dict[str, int] = {}
        violations: list[str] = []
        for tokenizer_name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self.pipeline, tokenizer_name, None)
            if tokenizer is None:
                continue
            token_count = self._count_tokens(tokenizer, prompt_text)
            counts[tokenizer_name] = token_count
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if isinstance(model_max_length, int) and model_max_length > 0:
                effective_limit = model_max_length - buffer_tokens
                if token_count > effective_limit:
                    violations.append(f"{tokenizer_name}={token_count}>{effective_limit}")

        if violations and strict:
            raise ValidationError(
                "Image prompt exceeds the current image model token budget: "
                + ", ".join(violations)
            )

        return counts

    def unload(self, clear_cache: bool = False) -> None:
        if clear_cache:
            _get_model_cache().pop(self._cache_key(), None)
            self.pipeline = None
            _clear_torch_memory()
            return

        self.pipeline = None


class KokoroNarrationEngine:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipeline: Any | None = None

    def _resolve_model_id(self) -> str:
        return self.config.models.part_narrator

    def _cache_key(self) -> tuple[str, str, str]:
        return ("narration_engine", self._resolve_model_id(), self.config.narration_lang_code)

    def _ensure_system_dependency(self) -> None:
        if shutil.which("espeak-ng") is not None:
            return
        raise ValidationError(
            "Kokoro narration requires espeak-ng. In Colab, run "
            "`!apt-get -qq -y install espeak-ng` once, then restart the session and relaunch."
        )

    def _load(self) -> None:
        if self.pipeline is not None:
            return

        self._ensure_system_dependency()
        from kokoro import KPipeline

        cache_key = self._cache_key()
        cached = _get_model_cache().get(cache_key)
        if cached is not None:
            print(f"[narration] Reusing cached model: {self._resolve_model_id()}")
            self.pipeline = cached["pipeline"]
            return

        print(f"[narration] Loading model: {self._resolve_model_id()}")
        self.pipeline = KPipeline(lang_code=self.config.narration_lang_code)
        _get_model_cache()[cache_key] = {"pipeline": self.pipeline}

    def narrate(
        self,
        *,
        text: str,
        output_path: str | Path,
    ) -> tuple[Path, float]:
        self._load()
        assert self.pipeline is not None

        import numpy as np
        import soundfile as sf

        audio_chunks: list[Any] = []
        for _, _, audio in self.pipeline(
            text,
            voice=self.config.narration_voice,
            speed=self.config.narration_speed,
        ):
            if audio is None:
                continue
            audio_chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not audio_chunks:
            raise ValidationError("Narration model produced no audio.")

        full_audio = np.concatenate(audio_chunks)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, full_audio, self.config.narration_sample_rate)
        duration_seconds = float(len(full_audio) / self.config.narration_sample_rate)
        print(
            f"[narration] Saved audio to {output_path} "
            f"(duration={duration_seconds:.2f}s)"
        )
        return output_path, duration_seconds

    def unload(self, clear_cache: bool = False) -> None:
        if clear_cache:
            _get_model_cache().pop(self._cache_key(), None)
        self.pipeline = None


class WhisperWordTimingEngine:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model: Any | None = None

    def _resolve_model_id(self) -> str:
        return self.config.models.word_aligner

    def _cache_key(self) -> tuple[str, str, str]:
        return ("word_aligner", self._resolve_model_id(), "cpu")

    def _load(self) -> None:
        if self.model is not None:
            return

        cache_key = self._cache_key()
        cached = _get_model_cache().get(cache_key)
        if cached is not None:
            print(f"[timing] Reusing cached model: {self._resolve_model_id()}")
            self.model = cached["model"]
            return

        import whisper

        print(f"[timing] Loading model: {self._resolve_model_id()} on cpu")
        self.model = whisper.load_model(self._resolve_model_id(), device="cpu")
        _get_model_cache()[cache_key] = {"model": self.model}

    def transcribe_words(self, audio_path: str | Path) -> list[dict[str, float | str]]:
        self._load()
        assert self.model is not None

        result = self.model.transcribe(
            str(Path(audio_path).resolve()),
            language="en",
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        words: list[dict[str, float | str]] = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append(
                    {
                        "word": str(word.get("word", "")).strip(),
                        "start": float(word.get("start", 0.0) or 0.0),
                        "end": float(word.get("end", 0.0) or 0.0),
                    }
                )
        if not words:
            raise ValidationError("Whisper word alignment produced no words.")
        return words

    def unload(self, clear_cache: bool = False) -> None:
        if clear_cache:
            _get_model_cache().pop(self._cache_key(), None)
        self.model = None
