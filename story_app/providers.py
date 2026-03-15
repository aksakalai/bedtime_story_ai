from __future__ import annotations

import builtins
import gc
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
