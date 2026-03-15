from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .assets import prepare_run_paths, write_json, write_text
from .config import DEFAULT_CONFIG, GenerationConfig
from .prompts import (
    build_description_prompt,
    build_description_messages,
    build_story_part_image_prompt,
    build_story_messages,
    format_story_messages,
    validate_description_text,
    validate_story_part_text,
)
from .providers import (
    KokoroNarrationEngine,
    Qwen25VLMultimodalEngine,
    SSD1BTextToImageGenerator,
    clear_cached_models,
)
from .schemas import (
    DescriptionResult,
    PipelineResult,
    RunPaths,
    StoryDraft,
    StoryboardManifest,
    StoryboardManifestPart,
)

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
        describer_factory=Qwen25VLMultimodalEngine,
        writer_factory=Qwen25VLMultimodalEngine,
        image_generator_factory=SSD1BTextToImageGenerator,
        narrator_factory=KokoroNarrationEngine,
    ):
        self.config = config
        self.describer_factory = describer_factory
        self.writer_factory = writer_factory
        self.image_generator_factory = image_generator_factory
        self.narrator_factory = narrator_factory
        self._describer = None
        self._writer = None
        self._image_generator = None
        self._narrator = None

    def _notify(self, progress_callback: ProgressCallback | None, value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def _get_describer(self):
        if self._describer is None:
            self._describer = self.describer_factory(self.config)
        return self._describer

    def _get_writer(self):
        if self.writer_factory is self.describer_factory:
            writer = self._get_describer()
            self._writer = writer
            return writer
        if self._writer is None:
            self._writer = self.writer_factory(self.config)
        return self._writer

    def _get_image_generator(self):
        if self._image_generator is None:
            self._image_generator = self.image_generator_factory(self.config)
        return self._image_generator

    def _get_narrator(self):
        if self._narrator is None:
            self._narrator = self.narrator_factory(self.config)
        return self._narrator

    def preload_models(self) -> None:
        self._get_describer()._load()
        self._get_writer()._load()
        self._get_image_generator()._load()
        self._get_narrator()._load()

    def clear_loaded_models(self) -> None:
        describer = self._describer
        writer = self._writer
        image_generator = self._image_generator
        narrator = self._narrator
        if describer is not None:
            try:
                describer.unload(clear_cache=True)
            except TypeError:
                describer.unload()
        if writer is not None and writer is not describer:
            try:
                writer.unload(clear_cache=True)
            except TypeError:
                writer.unload()
        if image_generator is not None:
            try:
                image_generator.unload(clear_cache=True)
            except TypeError:
                image_generator.unload()
        if narrator is not None:
            try:
                narrator.unload(clear_cache=True)
            except TypeError:
                narrator.unload()
        self._describer = None
        self._writer = None
        self._image_generator = None
        self._narrator = None
        clear_cached_models()

    def _create_story_draft_internal(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[RunPaths, PipelineResult]:
        print("[pipeline] Starting phase-1 story drafting")
        _seed_everything(self.config.random_seed)
        run_paths = prepare_run_paths(image_path, self.config.outputs_root)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        self._notify(progress_callback, 0.1, "Describing the drawing")
        description_prompt = build_description_prompt(self.config)
        write_text(run_paths.description_prompt_path, description_prompt)
        description_messages = build_description_messages(description_prompt)

        describer = self._get_describer()
        raw_description = describer.describe(run_paths.input_image_path, description_messages)
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
            raw_output = writer.generate_part(run_paths.input_image_path, messages)
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
        print("[pipeline] Story drafting complete")
        return run_paths, result

    def create_story_draft(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        _, result = self._create_story_draft_internal(
            image_path,
            progress_callback=progress_callback,
        )
        self._notify(progress_callback, 1.0, "Story draft ready")
        return result

    def create_story_package(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        print("[pipeline] Starting phase-2 storyboard package")
        run_paths, draft_result = self._create_story_draft_internal(
            image_path,
            progress_callback=progress_callback,
        )
        image_generator = self._get_image_generator()
        narrator = self._get_narrator()
        self._notify(progress_callback, 0.9, "Generating storyboard images")
        story_parts = [
            draft_result.part_1_text,
            draft_result.part_2_text,
            draft_result.part_3_text,
        ]
        prompt_paths = [
            run_paths.image_prompt_part_1_path,
            run_paths.image_prompt_part_2_path,
            run_paths.image_prompt_part_3_path,
        ]
        image_paths = [
            run_paths.story_part_1_image_path,
            run_paths.story_part_2_image_path,
            run_paths.story_part_3_image_path,
        ]
        audio_paths = [
            run_paths.story_part_1_audio_path,
            run_paths.story_part_2_audio_path,
            run_paths.story_part_3_audio_path,
        ]
        generated_prompt_paths: list[str] = []
        generated_image_paths: list[str] = []
        manifest_parts: list[StoryboardManifestPart] = []
        generated_audio_paths: list[str] = []
        progress_points = [0.92, 0.95, 0.98]

        for index, part_text in enumerate(story_parts, start=1):
            prompt_text = build_story_part_image_prompt(
                self.config,
                description_text=draft_result.description.description_text,
                part_text=part_text,
            )
            prompt_path = prompt_paths[index - 1]
            image_path_for_part = image_paths[index - 1]
            seed = self.config.random_seed + self.config.image_seed_stride + index

            write_text(prompt_path, prompt_text)
            output_path = image_generator.generate(
                prompt_text=prompt_text,
                negative_prompt_text=self.config.image_negative_prompt,
                seed=seed,
                output_path=image_path_for_part,
            )
            generated_prompt_paths.append(str(prompt_path.resolve()))
            generated_image_paths.append(str(output_path.resolve()))
            manifest_parts.append(
                StoryboardManifestPart(
                    index=index,
                    text=part_text,
                    image_prompt=prompt_text,
                    seed=seed,
                    image_path=str(output_path.resolve()),
                )
            )
            self._notify(
                progress_callback,
                progress_points[index - 1],
                f"Generated image for part {index}",
            )

        self._notify(progress_callback, 0.985, "Generating narration")
        narration_progress_points = [0.99, 0.995, 1.0]
        for index, part_text in enumerate(story_parts, start=1):
            audio_output_path, duration_seconds = narrator.narrate(
                text=part_text,
                output_path=audio_paths[index - 1],
            )
            resolved_audio_path = str(audio_output_path.resolve())
            generated_audio_paths.append(resolved_audio_path)
            manifest_parts[index - 1] = StoryboardManifestPart(
                index=manifest_parts[index - 1].index,
                text=manifest_parts[index - 1].text,
                image_prompt=manifest_parts[index - 1].image_prompt,
                seed=manifest_parts[index - 1].seed,
                image_path=manifest_parts[index - 1].image_path,
                audio_path=resolved_audio_path,
                audio_duration_seconds=duration_seconds,
            )
            self._notify(
                progress_callback,
                narration_progress_points[index - 1],
                f"Generated narration for part {index}",
            )

        manifest = StoryboardManifest(
            run_id=draft_result.run_id,
            input_image_path=draft_result.input_image_path,
            description_text=draft_result.description.description_text,
            parts=manifest_parts,
        )
        write_json(run_paths.storyboard_manifest_path, manifest.to_dict())
        print("[pipeline] Storyboard package complete")

        return replace(
            draft_result,
            image_prompt_part_1_path=generated_prompt_paths[0],
            image_prompt_part_2_path=generated_prompt_paths[1],
            image_prompt_part_3_path=generated_prompt_paths[2],
            story_part_1_image_path=generated_image_paths[0],
            story_part_2_image_path=generated_image_paths[1],
            story_part_3_image_path=generated_image_paths[2],
            story_part_1_audio_path=generated_audio_paths[0],
            story_part_2_audio_path=generated_audio_paths[1],
            story_part_3_audio_path=generated_audio_paths[2],
            storyboard_manifest_path=str(run_paths.storyboard_manifest_path.resolve()),
        )
