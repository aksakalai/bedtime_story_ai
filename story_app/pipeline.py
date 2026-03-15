from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from .assets import prepare_run_paths, write_text
from .config import DEFAULT_CONFIG, GenerationConfig
from .prompts import (
    build_description_prompt,
    build_story_part_prompt,
    normalize_text,
    validate_description_text,
    validate_story_part_text,
)
from .providers import QwenStoryWriter, QwenVLImageDescriber
from .schemas import DescriptionResult, PipelineResult, StoryDraft, StoryStep

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
        describer_factory=QwenVLImageDescriber,
        writer_factory=QwenStoryWriter,
    ):
        self.config = config
        self.describer_factory = describer_factory
        self.writer_factory = writer_factory

    def _notify(self, progress_callback: ProgressCallback | None, value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def create_story_draft(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        print("[pipeline] Starting phase-1 story drafting")
        _seed_everything(self.config.random_seed)
        run_paths = prepare_run_paths(image_path, self.config.outputs_root)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        self._notify(progress_callback, 0.08, "Describing the drawing")
        description_prompt = build_description_prompt(self.config)
        write_text(run_paths.description_prompt_path, description_prompt)

        describer = self.describer_factory(self.config)
        try:
            raw_description = describer.describe(run_paths.input_image_path, description_prompt)
            description_text = validate_description_text(raw_description, self.config)
        finally:
            describer.unload()
        write_text(run_paths.description_path, description_text)
        description = DescriptionResult(
            image_path=str(run_paths.input_image_path.resolve()),
            prompt_text=description_prompt,
            description_text=description_text,
        )
        print(f"[pipeline] Description: {description_text}")

        self._notify(progress_callback, 0.22, "Generating the three story parts")
        writer = self.writer_factory(self.config)
        steps: list[StoryStep] = []
        try:
            prompt_paths = {
                "part_1": run_paths.story_part_1_prompt_path,
                "part_2": run_paths.story_part_2_prompt_path,
                "part_3": run_paths.story_part_3_prompt_path,
            }
            output_paths = {
                "part_1": run_paths.story_part_1_path,
                "part_2": run_paths.story_part_2_path,
                "part_3": run_paths.story_part_3_path,
            }
            for index, step_name in enumerate(("part_1", "part_2", "part_3"), start=1):
                previous_parts = [step.output_text for step in steps]
                prompt_text = build_story_part_prompt(
                    description_text=description_text,
                    step_name=step_name,
                    previous_parts=previous_parts,
                )
                prompt_path = prompt_paths[step_name]
                output_path = output_paths[step_name]
                write_text(prompt_path, prompt_text)
                print(f"[pipeline] {step_name} prompt saved: {prompt_path}")

                raw_output = writer.generate_part(prompt_text)
                output_text = validate_story_part_text(raw_output, self.config)
                output_text = normalize_text(output_text)
                write_text(output_path, output_text)
                print(f"[pipeline] {step_name} output: {output_text}")
                steps.append(
                    StoryStep(
                        step_name=step_name,
                        prompt_text=prompt_text,
                        output_text=output_text,
                    )
                )
                self._notify(progress_callback, 0.22 + (index * 0.22), f"Generated {step_name}")
        finally:
            writer.unload()

        draft = StoryDraft(description=description, steps=steps)
        result = PipelineResult(
            run_id=run_paths.run_id,
            run_dir=run_paths.run_dir.resolve(),
            input_image_path=str(run_paths.input_image_path.resolve()),
            description_text=description_text,
            part_1_text=steps[0].output_text,
            part_2_text=steps[1].output_text,
            part_3_text=steps[2].output_text,
            draft=draft,
        )
        self._notify(progress_callback, 1.0, "Story draft ready")
        print("[pipeline] Story drafting complete")
        return result
