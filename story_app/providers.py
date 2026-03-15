from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .config import GenerationConfig


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


class SmolVLMImageDescriber:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cpu"
        self.processor: Any | None = None
        self.model: Any | None = None

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_class = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_class = AutoModelForVision2Seq

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[describe] Loading model: {self.config.models.image_describer} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(self.config.models.image_describer)
        self.model = model_class.from_pretrained(
            self.config.models.image_describer,
            torch_dtype=_torch_dtype(),
        )
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
                add_generation_prompt=True,
            )
            model_inputs = self.processor(
                text=rendered_prompt,
                images=[image],
                return_tensors="pt",
            )

        prepared_inputs: dict[str, Any] = {}
        for key, value in model_inputs.items():
            if getattr(value, "dtype", None) is not None and value.dtype.is_floating_point:
                prepared_inputs[key] = value.to(self.device, dtype=_torch_dtype())
            else:
                prepared_inputs[key] = value.to(self.device)

        generated_ids = self.model.generate(
            **prepared_inputs,
            max_new_tokens=self.config.description_max_tokens,
            do_sample=False,
        )
        prompt_length = prepared_inputs["input_ids"].shape[1]
        completion = generated_ids[:, prompt_length:]
        return self.processor.batch_decode(completion, skip_special_tokens=True)[0].strip()

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

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.story_part_max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self.tokenizer = None
        self.model = None
        _clear_torch_memory()
