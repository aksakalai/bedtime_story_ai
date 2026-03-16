from __future__ import annotations

import random
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from PIL import Image

from .assets import prepare_run_paths, write_json, write_text
from .config import DEFAULT_CONFIG, GenerationConfig
from .prompts import (
    build_description_messages,
    build_description_prompt,
    build_next_part_anchor_messages,
    build_story_messages,
    build_story_part_image_summary_messages,
    finalize_image_prompt,
    format_story_messages,
    validate_anchor_text,
    validate_description_text,
    validate_image_prompt_text,
    validate_story_part_text,
)
from .providers import (
    KokoroNarrationEngine,
    Qwen25VLMultimodalEngine,
    SSD1BTextToImageGenerator,
    WhisperWordTimingEngine,
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
from .video import FFmpegVideoAssembler, align_story_text_to_timestamps, build_story_part_ass

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
        word_aligner_factory=WhisperWordTimingEngine,
        video_assembler_factory=FFmpegVideoAssembler,
    ):
        self.config = config
        self.describer_factory = describer_factory
        self.writer_factory = writer_factory
        self.image_generator_factory = image_generator_factory
        self.narrator_factory = narrator_factory
        self.word_aligner_factory = word_aligner_factory
        self.video_assembler_factory = video_assembler_factory
        self._describer = None
        self._writer = None
        self._image_generator = None
        self._narrator = None
        self._word_aligner = None
        self._video_assembler = None
        self._runtime_is_warm = False

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

    def _get_image_generator(self):
        if self._image_generator is None:
            self._image_generator = self.image_generator_factory(self.config)
        return self._image_generator

    def _get_narrator(self):
        if self._narrator is None:
            self._narrator = self.narrator_factory(self.config)
        return self._narrator

    def _get_word_aligner(self):
        if self._word_aligner is None:
            self._word_aligner = self.word_aligner_factory(self.config)
        return self._word_aligner

    def _get_video_assembler(self):
        if self._video_assembler is None:
            self._video_assembler = self.video_assembler_factory(self.config)
        return self._video_assembler

    def preload_models(self) -> None:
        self._get_describer()._load()
        self._get_writer()._load()
        self._get_image_generator()._load()
        self._get_narrator()._load()
        self._get_word_aligner()._load()

    def warm_up_runtime(self) -> None:
        if self._runtime_is_warm:
            print("[warmup] Runtime already warmed up")
            return

        print("[warmup] Starting hidden runtime warm-up")
        self.preload_models()
        describer = self._get_describer()
        writer = self._get_writer()
        image_generator = self._get_image_generator()
        narrator = self._get_narrator()
        word_aligner = self._get_word_aligner()
        video_assembler = self._get_video_assembler()
        image_prompt_token_limit = image_generator.get_prompt_token_limit(
            buffer_tokens=self.config.image_prompt_token_buffer,
        )

        warm_anchor_1 = (
            "actor: curious little blue fish\n"
            "actor_colors: blue and purple\n"
            "actor_traits: bright eyes, long flowing fins\n"
            "scene: underwater garden\n"
            "scene_colors: teal water, green sea plants, sandy floor\n"
            "object_1: sea plants\n"
            "object_1_colors: green\n"
            "object_2: coral arch\n"
            "object_2_colors: orange\n"
            "object_3: sandy floor\n"
            "object_3_colors: pale beige\n"
            "object_4: bubbles\n"
            "object_4_colors: silvery white\n"
            "object_5: none\n"
            "object_5_colors: none\n"
            "secondary_actor: none\n"
            "secondary_actor_colors: none\n"
            "page_event: calm exploration\n"
            "mood: peaceful and curious"
        )
        warm_part_1 = (
            "A curious little blue fish with purple shimmer swam through the underwater garden beside the green sea "
            "plants. The orange coral arch glowed softly nearby while the water stayed calm and bright. The little "
            "fish slowed down as if it had just noticed something new ahead."
        )

        with tempfile.TemporaryDirectory(prefix="bedtime_story_ai_warmup_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            warm_image_path = temp_dir / "warmup_input.png"
            warm_generated_image_path = temp_dir / "warmup_story_image.png"
            warm_audio_path = temp_dir / "warmup_audio.wav"
            warm_subtitle_path = temp_dir / "warmup_subtitles.ass"
            warm_clip_path = temp_dir / "warmup_clip.mp4"
            warm_final_video_path = temp_dir / "warmup_final.mp4"

            Image.new(
                "RGB",
                (self.config.video_width, self.config.video_height),
                color=(245, 241, 231),
            ).save(warm_image_path)

            initial_anchor_messages = build_description_messages(build_description_prompt(self.config))
            describer.extract_initial_anchor(warm_image_path, initial_anchor_messages)
            story_messages = build_story_messages(
                part_index=1,
                current_anchor_text=warm_anchor_1,
                previous_anchors=[],
                previous_parts=[],
            )
            writer.generate_part(warm_image_path, story_messages)
            next_anchor_messages = build_next_part_anchor_messages(
                current_anchor_text=warm_anchor_1,
                latest_part_text=warm_part_1,
                next_part_index=2,
            )
            writer.generate_next_anchor(next_anchor_messages)

            prompt_messages = build_story_part_image_summary_messages(
                self.config,
                page_anchor_text=warm_anchor_1,
                part_index=1,
                max_image_prompt_tokens=image_prompt_token_limit,
            )
            scene_prompt_text = writer.generate_image_prompt(
                prompt_messages,
                max_new_tokens=image_prompt_token_limit,
            )
            scene_prompt_text = validate_image_prompt_text(scene_prompt_text, self.config)
            prompt_text = finalize_image_prompt(self.config, scene_prompt_text)
            prompt_token_counts = image_generator.validate_prompt_token_budget(
                prompt_text,
                buffer_tokens=self.config.image_prompt_token_buffer,
                strict=True,
            )
            print(f"[warmup] Image prompt token limit: {image_prompt_token_limit}")
            print(
                "[warmup] Image prompt token counts: "
                + ", ".join(f"{name}={count}" for name, count in prompt_token_counts.items())
            )
            image_generator.generate(
                prompt_text=prompt_text,
                negative_prompt_text=self.config.image_negative_prompt,
                seed=self.config.random_seed,
                output_path=warm_generated_image_path,
            )
            audio_output_path, duration_seconds = narrator.narrate(
                text=warm_part_1,
                output_path=warm_audio_path,
            )
            whisper_words = word_aligner.transcribe_words(audio_output_path)
            timed_tokens = align_story_text_to_timestamps(
                display_text=warm_part_1,
                whisper_words=whisper_words,
                fallback_total_duration=duration_seconds,
            )
            ass_text, overlay_layout = build_story_part_ass(
                timed_tokens=timed_tokens,
                total_duration_seconds=duration_seconds,
                config=self.config,
            )
            write_text(warm_subtitle_path, ass_text)
            clip_output_path = video_assembler.render_story_part_clip(
                image_path=warm_generated_image_path,
                audio_path=audio_output_path,
                subtitle_path=warm_subtitle_path,
                overlay_layout=overlay_layout,
                total_duration_seconds=duration_seconds,
                output_path=warm_clip_path,
            )
            video_assembler.concatenate_story_clips(
                clip_paths=[clip_output_path],
                output_path=warm_final_video_path,
            )

        self._runtime_is_warm = True
        print("[warmup] Runtime warm-up complete")

    def clear_loaded_models(self) -> None:
        describer = self._describer
        writer = self._writer
        image_generator = self._image_generator
        narrator = self._narrator
        word_aligner = self._word_aligner
        if describer is not None:
            try:
                describer.unload(clear_cache=True)
            except TypeError:
                describer.unload()
        if writer is not None:
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
        if word_aligner is not None:
            try:
                word_aligner.unload(clear_cache=True)
            except TypeError:
                word_aligner.unload()
        self._describer = None
        self._writer = None
        self._image_generator = None
        self._narrator = None
        self._word_aligner = None
        self._video_assembler = None
        self._runtime_is_warm = False
        clear_cached_models()

    def _create_story_draft_internal(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[RunPaths, PipelineResult, list[str]]:
        print("[pipeline] Starting phase-1 story drafting")
        _seed_everything(self.config.random_seed)
        run_paths = prepare_run_paths(image_path, self.config.outputs_root)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        self._notify(progress_callback, 0.08, "Extracting page 1 anchors")
        prompt_text = build_description_prompt(self.config)
        write_text(run_paths.description_prompt_path, prompt_text)
        initial_anchor_messages = build_description_messages(prompt_text)

        describer = self._get_describer()
        writer = self._get_writer()
        raw_initial_anchor = describer.extract_initial_anchor(run_paths.input_image_path, initial_anchor_messages)
        initial_anchor_text = validate_description_text(raw_initial_anchor, self.config)
        write_text(run_paths.description_path, initial_anchor_text)
        write_text(run_paths.story_part_1_anchor_path, initial_anchor_text)

        description = DescriptionResult(
            image_path=str(run_paths.input_image_path.resolve()),
            prompt_text=prompt_text,
            description_text=initial_anchor_text,
        )
        print(f"[pipeline] part_1 anchor:\n{initial_anchor_text}")

        page_anchors = [initial_anchor_text]
        story_parts: list[str] = []
        transcript_messages: list[dict[str, Any]] = []

        for part_index in range(1, 4):
            if part_index == 1:
                self._notify(progress_callback, 0.25, "Writing part 1")
            elif part_index == 2:
                self._notify(progress_callback, 0.45, "Writing part 2")
            else:
                self._notify(progress_callback, 0.65, "Writing part 3")

            story_messages = build_story_messages(
                part_index=part_index,
                current_anchor_text=page_anchors[-1],
                previous_anchors=page_anchors[:-1],
                previous_parts=story_parts,
            )
            raw_story_part = writer.generate_part(run_paths.input_image_path, story_messages)
            story_part_text = validate_story_part_text(raw_story_part, self.config)
            story_parts.append(story_part_text)

            if not transcript_messages:
                transcript_messages.extend(story_messages)
            else:
                transcript_messages.append(story_messages[1])
            transcript_messages.append({"role": "assistant", "content": story_part_text})

            if part_index == 1:
                story_path = run_paths.story_part_1_path
            elif part_index == 2:
                story_path = run_paths.story_part_2_path
            else:
                story_path = run_paths.story_part_3_path
            write_text(story_path, story_part_text)
            print(
                f"[pipeline] part_{part_index} stats: "
                f"words={len(story_part_text.split())}, "
                f"chars={len(story_part_text)}"
            )
            print(f"[pipeline] part_{part_index} output: {story_part_text}")

            if part_index < 3:
                next_anchor_messages = build_next_part_anchor_messages(
                    current_anchor_text=page_anchors[-1],
                    latest_part_text=story_part_text,
                    next_part_index=part_index + 1,
                )
                raw_next_anchor = writer.generate_next_anchor(next_anchor_messages)
                next_anchor_text = validate_anchor_text(raw_next_anchor, self.config)
                page_anchors.append(next_anchor_text)
                next_anchor_path = (
                    run_paths.story_part_2_anchor_path
                    if part_index == 1
                    else run_paths.story_part_3_anchor_path
                )
                write_text(next_anchor_path, next_anchor_text)
                print(f"[pipeline] part_{part_index + 1} anchor:\n{next_anchor_text}")

        full_conversation_text = format_story_messages(transcript_messages)
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
            story_part_1_anchor_path=str(run_paths.story_part_1_anchor_path.resolve()),
            story_part_2_anchor_path=str(run_paths.story_part_2_anchor_path.resolve()),
            story_part_3_anchor_path=str(run_paths.story_part_3_anchor_path.resolve()),
        )
        print("[pipeline] Story drafting complete")
        return run_paths, result, page_anchors

    def create_story_draft(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        _, result, _ = self._create_story_draft_internal(
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
        run_paths, draft_result, page_anchors = self._create_story_draft_internal(
            image_path,
            progress_callback=progress_callback,
        )
        image_generator = self._get_image_generator()
        writer = self._get_writer()
        narrator = self._get_narrator()
        word_aligner = self._get_word_aligner()
        video_assembler = self._get_video_assembler()
        image_prompt_token_limit = image_generator.get_prompt_token_limit(
            buffer_tokens=self.config.image_prompt_token_buffer,
        )
        self._notify(progress_callback, 0.84, "Generating storyboard images")

        story_parts = [
            draft_result.part_1_text,
            draft_result.part_2_text,
            draft_result.part_3_text,
        ]
        anchor_paths = [
            run_paths.story_part_1_anchor_path,
            run_paths.story_part_2_anchor_path,
            run_paths.story_part_3_anchor_path,
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
        subtitle_paths = [
            run_paths.story_part_1_subtitle_path,
            run_paths.story_part_2_subtitle_path,
            run_paths.story_part_3_subtitle_path,
        ]
        clip_paths = [
            run_paths.story_part_1_clip_path,
            run_paths.story_part_2_clip_path,
            run_paths.story_part_3_clip_path,
        ]

        generated_prompt_paths: list[str] = []
        generated_image_paths: list[str] = []
        manifest_parts: list[StoryboardManifestPart] = []
        generated_audio_paths: list[str] = []
        generated_subtitle_paths: list[str] = []
        generated_clip_paths: list[str] = []
        progress_points = [0.88, 0.91, 0.94]

        for index, page_anchor_text in enumerate(page_anchors, start=1):
            prompt_messages = build_story_part_image_summary_messages(
                self.config,
                page_anchor_text=page_anchor_text,
                part_index=index,
                max_image_prompt_tokens=image_prompt_token_limit,
            )
            scene_prompt_text = writer.generate_image_prompt(
                prompt_messages,
                max_new_tokens=image_prompt_token_limit,
            )
            scene_prompt_text = validate_image_prompt_text(scene_prompt_text, self.config)
            prompt_text = finalize_image_prompt(self.config, scene_prompt_text)
            prompt_token_counts = image_generator.validate_prompt_token_budget(
                prompt_text,
                buffer_tokens=self.config.image_prompt_token_buffer,
                strict=True,
            )
            print(f"[pipeline] part_{index} image prompt token limit: {image_prompt_token_limit}")
            print(
                f"[pipeline] part_{index} image prompt token counts: "
                + ", ".join(f"{name}={count}" for name, count in prompt_token_counts.items())
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
                    text=story_parts[index - 1],
                    anchor_text=page_anchor_text,
                    image_prompt=prompt_text,
                    seed=seed,
                    image_path=str(output_path.resolve()),
                    anchor_path=str(anchor_paths[index - 1].resolve()),
                )
            )
            self._notify(
                progress_callback,
                progress_points[index - 1],
                f"Generated image for part {index}",
            )

        self._notify(progress_callback, 0.955, "Generating narration")
        narration_progress_points = [0.965, 0.972, 0.979]
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
                anchor_text=manifest_parts[index - 1].anchor_text,
                image_prompt=manifest_parts[index - 1].image_prompt,
                seed=manifest_parts[index - 1].seed,
                image_path=manifest_parts[index - 1].image_path,
                anchor_path=manifest_parts[index - 1].anchor_path,
                audio_path=resolved_audio_path,
                audio_duration_seconds=duration_seconds,
            )
            self._notify(
                progress_callback,
                narration_progress_points[index - 1],
                f"Generated narration for part {index}",
            )

        self._notify(progress_callback, 0.982, "Aligning narration and building overlays")
        overlay_progress_points = [0.988, 0.993, 0.997]
        for index, part_text in enumerate(story_parts, start=1):
            resolved_audio_path = generated_audio_paths[index - 1]
            whisper_words = word_aligner.transcribe_words(resolved_audio_path)
            timed_tokens = align_story_text_to_timestamps(
                display_text=part_text,
                whisper_words=whisper_words,
                fallback_total_duration=manifest_parts[index - 1].audio_duration_seconds or 0.0,
            )
            ass_text, overlay_layout = build_story_part_ass(
                timed_tokens=timed_tokens,
                total_duration_seconds=manifest_parts[index - 1].audio_duration_seconds or 0.0,
                config=self.config,
            )
            subtitle_path = subtitle_paths[index - 1]
            write_text(subtitle_path, ass_text)
            generated_subtitle_paths.append(str(subtitle_path.resolve()))
            clip_output_path = video_assembler.render_story_part_clip(
                image_path=generated_image_paths[index - 1],
                audio_path=resolved_audio_path,
                subtitle_path=subtitle_path,
                overlay_layout=overlay_layout,
                total_duration_seconds=manifest_parts[index - 1].audio_duration_seconds or 0.0,
                output_path=clip_paths[index - 1],
            )
            generated_clip_paths.append(str(clip_output_path.resolve()))
            manifest_parts[index - 1] = StoryboardManifestPart(
                index=manifest_parts[index - 1].index,
                text=manifest_parts[index - 1].text,
                anchor_text=manifest_parts[index - 1].anchor_text,
                image_prompt=manifest_parts[index - 1].image_prompt,
                seed=manifest_parts[index - 1].seed,
                image_path=manifest_parts[index - 1].image_path,
                anchor_path=manifest_parts[index - 1].anchor_path,
                audio_path=manifest_parts[index - 1].audio_path,
                audio_duration_seconds=manifest_parts[index - 1].audio_duration_seconds,
                subtitle_path=str(subtitle_path.resolve()),
                clip_path=str(clip_output_path.resolve()),
            )
            self._notify(
                progress_callback,
                overlay_progress_points[index - 1],
                f"Rendered clip for part {index}",
            )

        self._notify(progress_callback, 0.998, "Combining final story video")
        final_story_video_path = video_assembler.concatenate_story_clips(
            clip_paths=generated_clip_paths,
            output_path=run_paths.final_story_video_path,
        )

        manifest = StoryboardManifest(
            run_id=draft_result.run_id,
            input_image_path=draft_result.input_image_path,
            description_text=draft_result.description.description_text,
            parts=manifest_parts,
            final_story_video_path=str(final_story_video_path.resolve()),
        )
        write_json(run_paths.storyboard_manifest_path, manifest.to_dict())
        print("[pipeline] Storyboard package complete")
        self._notify(progress_callback, 1.0, "Final story video ready")

        return replace(
            draft_result,
            story_part_1_anchor_path=str(run_paths.story_part_1_anchor_path.resolve()),
            story_part_2_anchor_path=str(run_paths.story_part_2_anchor_path.resolve()),
            story_part_3_anchor_path=str(run_paths.story_part_3_anchor_path.resolve()),
            image_prompt_part_1_path=generated_prompt_paths[0],
            image_prompt_part_2_path=generated_prompt_paths[1],
            image_prompt_part_3_path=generated_prompt_paths[2],
            story_part_1_image_path=generated_image_paths[0],
            story_part_2_image_path=generated_image_paths[1],
            story_part_3_image_path=generated_image_paths[2],
            story_part_1_audio_path=generated_audio_paths[0],
            story_part_2_audio_path=generated_audio_paths[1],
            story_part_3_audio_path=generated_audio_paths[2],
            story_part_1_subtitle_path=generated_subtitle_paths[0],
            story_part_2_subtitle_path=generated_subtitle_paths[1],
            story_part_3_subtitle_path=generated_subtitle_paths[2],
            story_part_1_clip_path=generated_clip_paths[0],
            story_part_2_clip_path=generated_clip_paths[1],
            story_part_3_clip_path=generated_clip_paths[2],
            final_story_video_path=str(final_story_video_path.resolve()),
            storyboard_manifest_path=str(run_paths.storyboard_manifest_path.resolve()),
        )
