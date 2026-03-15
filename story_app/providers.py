from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from PIL import Image

from .config import GenerationConfig
from .schemas import ValidationError


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


class Qwen2VLImageDescriber:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.processor: Any | None = None
        self.model: Any | None = None

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[describe] Loading model: {self.config.models.image_describer} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(self.config.models.image_describer)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.models.image_describer,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)

    def describe(self, image_path: str | Path, prompt_text: str) -> str:
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
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            rendered_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.processor(
                text=[rendered_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        prepared_inputs: dict[str, Any] = {}
        for key, value in model_inputs.items():
            prepared_inputs[key] = value.to(self.device)

        prompt_token_count = int(prepared_inputs["input_ids"].shape[1])
        print(f"[describe] Prompt token count: {prompt_token_count}")

        generated_ids = self.model.generate(
            **prepared_inputs,
            max_new_tokens=self.config.description_max_tokens,
            do_sample=False,
        )
        prompt_length = prepared_inputs["input_ids"].shape[1]
        completion = generated_ids[:, prompt_length:]
        generated_token_count = int(completion.shape[1])
        hit_token_cap = generated_token_count >= self.config.description_max_tokens
        eos_token_id = None
        if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "eos_token_id", None) is not None:
            eos_token_id = int(self.processor.tokenizer.eos_token_id)
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
            "[describe] Generation stats: "
            f"generated_tokens={generated_token_count}, "
            f"max_new_tokens={self.config.description_max_tokens}, "
            f"hit_token_cap={hit_token_cap}, "
            f"ended_with_eos={ended_with_eos}"
        )
        print(f"[describe] Output word count: {len(decoded.split())}")
        print(f"[describe] Output preview: {decoded[:240]}")
        if not ended_with_eos:
            raise ValidationError(
                "Description generation did not finish naturally before the safety limit. "
                "Increase the description token ceiling or tighten the prompt."
            )
        return decoded

    def unload(self) -> None:
        self.processor = None
        self.model = None
        _clear_torch_memory()


class QwenStoryWriter:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def _load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
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
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        try:
            self.device = str(next(self.model.parameters()).device)
        except StopIteration:
            self.device = "cpu"

    def generate_part(self, prompt_text: str) -> str:
        self._load()
        assert self.tokenizer is not None
        assert self.model is not None

        messages = [
            {
                "role": "system",
                "content": "You write only clean bedtime-story prose. Follow the user's formatting and length instructions exactly.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[1])
        print(f"[story] Prompt token count: {prompt_token_count}")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.story_part_max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0][prompt_length:]
        generated_token_count = int(completion_ids.shape[0])
        hit_token_cap = generated_token_count >= self.config.story_part_max_tokens
        ended_with_eos = bool(
            generated_token_count
            and self.tokenizer.eos_token_id is not None
            and int(completion_ids[-1].item()) == int(self.tokenizer.eos_token_id)
        )
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        tail_preview = decoded[-120:] if decoded else ""
        print(
            "[story] Generation stats: "
            f"generated_tokens={generated_token_count}, "
            f"max_new_tokens={self.config.story_part_max_tokens}, "
            f"hit_token_cap={hit_token_cap}, "
            f"ended_with_eos={ended_with_eos}"
        )
        print(f"[story] Output word count: {len(decoded.split())}")
        print(f"[story] Output tail preview: {tail_preview}")
        if not ended_with_eos:
            raise ValidationError(
                "Story generation did not finish naturally before the safety limit. "
                "Increase the story token ceiling or tighten the prompt."
            )
        return decoded

    def unload(self) -> None:
        self.tokenizer = None
        self.model = None
        _clear_torch_memory()
