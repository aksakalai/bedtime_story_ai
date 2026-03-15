from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from .assets import prepare_run_paths, write_text
from .config import DEFAULT_CONFIG, GenerationConfig
from .prompts import (
    build_description_prompt,
    build_story_messages,
    format_story_messages,
    validate_description_text,
    validate_story_part_text,
)
from .providers import Qwen2VLImageDescriber, QwenStoryWriter, clear_cached_models
from .schemas import DescriptionResult, PipelineResult, StoryDraft

ProgressCallback = Callable[[float, str], None]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class KidStoryPipeline:
    def __init__(
        self,
        config: GenerationConfig = DEFAULT_CONFIG,
        describer_factory=Qwen2VLImageDescriber,
        writer_factory=QwenStoryWriter,
    ):
        self.config = config
        self.describer_factory = describer_factory
        self.writer_factory = writer_factory
        self._describer = None
        self._writer = None

    def _notify(self, progress_callback: ProgressCallback | None, value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def _get_describer(self):
        if self._describer is None:
            self._describer = self.describer_factory(self.config)
        return self._describer

    def _get_writer(self):
        if self._writer is None:
            self._writer = self.writer_factory(self.config)
        return self._writer

    def preload_models(self) -> None:
        self._get_describer()._load()
        self._get_writer()._load()

    def clear_loaded_models(self) -> None:
        if self._describer is not None:
            try:
                self._describer.unload(clear_cache=True)
            except TypeError:
                self._describer.unload()
            self._describer = None
        if self._writer is not None:
            try:
                self._writer.unload(clear_cache=True)
            except TypeError:
                self._writer.unload()
            self._writer = None
        clear_cached_models()

    def create_story_draft(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        print("[pipeline] Starting phase-1 story drafting")
        _seed_everything(self.config.random_seed)
        run_paths = prepare_run_paths(image_path, self.config.outputs_root)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        self._notify(progress_callback, 0.1, "Describing the drawing")
        description_prompt = build_description_prompt(self.config)
        write_text(run_paths.description_prompt_path, description_prompt)

        describer = self._get_describer()
        raw_description = describer.describe(run_paths.input_image_path, description_prompt)
        description_text = validate_description_text(raw_description, self.config)

        write_text(run_paths.description_path, description_text)
        description = DescriptionResult(
            image_path=str(run_paths.input_image_path.resolve()),
            prompt_text=description_prompt,
            description_text=description_text,
        )
        print(f"[pipeline] Description: {description_text}")

        self._notify(progress_callback, 0.35, "Writing part 1")
        writer = self._get_writer()
        story_parts: list[str] = []
        for index, step_name in enumerate(("part_1", "part_2", "part_3"), start=1):
            messages = build_story_messages(
                description_text=description_text,
                previous_parts=story_parts,
            )
            raw_output = writer.generate_part(messages)
            output_text = validate_story_part_text(raw_output, self.config)
            story_parts.append(output_text)
            if step_name == "part_1":
                story_path = run_paths.story_part_1_path
            elif step_name == "part_2":
                story_path = run_paths.story_part_2_path
            else:
                story_path = run_paths.story_part_3_path
            write_text(story_path, output_text)
            print(
                f"[pipeline] {step_name} stats: "
                f"words={len(output_text.split())}, "
                f"chars={len(output_text)}"
            )
            print(f"[pipeline] {step_name} output: {output_text}")
            self._notify(progress_callback, 0.35 + (index * 0.18), f"Generated {step_name}")

        final_messages = build_story_messages(
            description_text=description_text,
            previous_parts=story_parts[:2],
        )
        full_conversation_messages = [
            *final_messages,
            {"role": "assistant", "content": story_parts[2]},
        ]
        full_conversation_text = format_story_messages(full_conversation_messages)
        write_text(run_paths.story_conversation_path, full_conversation_text)

        draft = StoryDraft(
            full_conversation_text=full_conversation_text,
            part_1_text=story_parts[0],
            part_2_text=story_parts[1],
            part_3_text=story_parts[2],
        )
        result = PipelineResult(
            run_id=run_paths.run_id,
            run_dir=run_paths.run_dir.resolve(),
            input_image_path=str(run_paths.input_image_path.resolve()),
            description=description,
            draft=draft,
            full_conversation_text=full_conversation_text,
            part_1_text=story_parts[0],
            part_2_text=story_parts[1],
            part_3_text=story_parts[2],
        )
        self._notify(progress_callback, 1.0, "Story draft ready")
        print("[pipeline] Story drafting complete")
        return result
