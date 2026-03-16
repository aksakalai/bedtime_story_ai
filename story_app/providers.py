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
        image_path: str | Path,
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


class PixArtSigmaTextToImageGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.prompt_encoder_pipeline: Any | None = None
        self.generation_pipeline: Any | None = None
        self.uses_split_pipeline = False

    def _resolve_model_id(self) -> str:
        return self.config.models.part_image_generator

    def _cache_key(self) -> tuple[str, str, str]:
        return ("image_generator", self._resolve_model_id(), self.device)

    def _load(self) -> None:
        if self.generation_pipeline is not None:
            return

        import torch
        from diffusers import PixArtSigmaPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = self._cache_key()
        cached = _get_model_cache().get(cache_key)
        if cached is not None:
            print(f"[image] Reusing cached model: {self._resolve_model_id()} on {self.device}")
            self.prompt_encoder_pipeline = cached["prompt_encoder_pipeline"]
            self.generation_pipeline = cached["generation_pipeline"]
            self.uses_split_pipeline = bool(cached.get("uses_split_pipeline", False))
            return

        print(f"[image] Loading model: {self._resolve_model_id()} on {self.device}")
        if self.device == "cuda":
            self._try_load_split_pixart_pipeline()
        else:
            self.prompt_encoder_pipeline = None
            self.generation_pipeline = PixArtSigmaPipeline.from_pretrained(
                self._resolve_model_id(),
                use_safetensors=True,
            )
            self.generation_pipeline.to(self.device)
            self.uses_split_pipeline = False

        for pipe in (self.prompt_encoder_pipeline, self.generation_pipeline):
            if pipe is None:
                continue
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        _get_model_cache()[cache_key] = {
            "prompt_encoder_pipeline": self.prompt_encoder_pipeline,
            "generation_pipeline": self.generation_pipeline,
            "uses_split_pipeline": self.uses_split_pipeline,
        }

    def _try_load_split_pixart_pipeline(self) -> None:
        import torch
        from diffusers import PixArtSigmaPipeline
        from transformers import BitsAndBytesConfig, T5EncoderModel

        load_attempts: list[tuple[str, dict[str, Any]]] = [
            (
                "4-bit",
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                },
            ),
            (
                "8-bit",
                {
                    "load_in_8bit": True,
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                },
            ),
        ]

        last_error: Exception | None = None
        for label, text_encoder_kwargs in load_attempts:
            try:
                text_encoder = T5EncoderModel.from_pretrained(
                    self._resolve_model_id(),
                    subfolder="text_encoder",
                    **text_encoder_kwargs,
                )
                self.prompt_encoder_pipeline = PixArtSigmaPipeline.from_pretrained(
                    self._resolve_model_id(),
                    text_encoder=text_encoder,
                    transformer=None,
                    device_map="balanced",
                    use_safetensors=True,
                )
                self.generation_pipeline = PixArtSigmaPipeline.from_pretrained(
                    self._resolve_model_id(),
                    text_encoder=None,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
                self.generation_pipeline.to(self.device)
                self.uses_split_pipeline = True
                print(f"[image] Loaded split PixArt-Sigma pipeline with {label} T5 encoder")
                return
            except Exception as exc:
                last_error = exc
                print(f"[image] Split PixArt-Sigma load failed with {label} T5 encoder: {exc}")
                self.prompt_encoder_pipeline = None
                self.generation_pipeline = None
                self.uses_split_pipeline = False

        print(f"[image] Falling back to single PixArt-Sigma pipeline: {last_error}")
        self.prompt_encoder_pipeline = None
        self.generation_pipeline = PixArtSigmaPipeline.from_pretrained(
            self._resolve_model_id(),
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.generation_pipeline.to(self.device)
        self.uses_split_pipeline = False

    def generate(
        self,
        *,
        prompt_text: str,
        negative_prompt_text: str,
        seed: int,
        output_path: str | Path,
    ) -> Path:
        self._load()
        assert self.generation_pipeline is not None

        import torch

        if self.device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        pipeline_kwargs: dict[str, Any] = {
            "width": self.config.image_width,
            "height": self.config.image_height,
            "num_inference_steps": self.config.image_num_inference_steps,
            "guidance_scale": self.config.image_guidance_scale,
            "generator": generator,
            "clean_caption": False,
            "max_sequence_length": self.config.image_max_sequence_length,
            "use_resolution_binning": True,
        }
        if self.uses_split_pipeline:
            assert self.prompt_encoder_pipeline is not None
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                self.prompt_encoder_pipeline.encode_prompt(
                    prompt=prompt_text,
                    negative_prompt=negative_prompt_text,
                    clean_caption=False,
                    max_sequence_length=self.config.image_max_sequence_length,
                )
            )
            result = self.generation_pipeline(
                prompt_embeds=prompt_embeds.to(self.device),
                prompt_attention_mask=prompt_attention_mask.to(self.device),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device),
                negative_prompt_attention_mask=negative_prompt_attention_mask.to(self.device),
                **pipeline_kwargs,
            )
        else:
            result = self.generation_pipeline(
                prompt=prompt_text,
                negative_prompt=negative_prompt_text,
                **pipeline_kwargs,
            )
        generated_image = result.images[0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_image.save(output_path)
        print(f"[image] Saved image to {output_path}")
        return output_path

    def unload(self, clear_cache: bool = False) -> None:
        if clear_cache:
            _get_model_cache().pop(self._cache_key(), None)
            self.prompt_encoder_pipeline = None
            self.generation_pipeline = None
            self.uses_split_pipeline = False
            _clear_torch_memory()
            return

        self.prompt_encoder_pipeline = None
        self.generation_pipeline = None
        self.uses_split_pipeline = False


SSD1BTextToImageGenerator = PixArtSigmaTextToImageGenerator


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
