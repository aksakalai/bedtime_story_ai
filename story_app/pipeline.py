from __future__ import annotations

from pathlib import Path
from typing import Callable

from .assets import RunPaths, build_run_manifest, prepare_run_paths, write_json
from .config import DEFAULT_CONFIG, GenerationConfig
from .playback import build_playback_panel_html, build_timeline
from .prompts import build_story_markdown
from .providers import KokoroNarrator, QwenStoryWriter, SSD1BSceneGenerator, SmolVLMDescriber
from .schemas import DrawingDescription, PipelineResult, SchemaError, StoryPackage

ProgressCallback = Callable[[float, str], None]


class KidStoryPipeline:
    def __init__(self, config: GenerationConfig = DEFAULT_CONFIG):
        self.config = config
        self._narrator = KokoroNarrator(config)

    def _notify(self, progress_callback: ProgressCallback | None, value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def _with_schema_retries(
        self,
        label: str,
        action: Callable[[int], DrawingDescription | StoryPackage],
        max_attempts: int = 3,
    ) -> DrawingDescription | StoryPackage:
        last_error: Exception | None = None
        for attempt_index in range(max_attempts):
            try:
                print(f"[pipeline] {label}: attempt {attempt_index + 1}/{max_attempts}")
                return action(attempt_index)
            except SchemaError as exc:
                last_error = exc
                print(f"[pipeline] {label}: attempt {attempt_index + 1} failed with schema error: {exc}")
        raise RuntimeError(f"Failed to produce valid {label} after {max_attempts} attempts: {last_error}")

    def _save_story(self, run_paths: RunPaths, story: StoryPackage) -> None:
        write_json(run_paths.story_path, story.to_dict())

    def create_story(
        self,
        image_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        print("[pipeline] Starting story generation")
        self._notify(progress_callback, 0.02, "Preparing output folders")
        run_paths = prepare_run_paths(image_path, self.config.outputs_root)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        describer = SmolVLMDescriber(self.config)
        self._notify(progress_callback, 0.12, "Describing the drawing")
        description = self._with_schema_retries(
            "drawing description",
            lambda attempt_index: describer.describe(run_paths.input_image_path, attempt_index),
        )
        assert isinstance(description, DrawingDescription)
        write_json(run_paths.description_path, description.to_dict())
        print(f"[pipeline] Drawing description saved: {run_paths.description_path}")
        describer.unload()

        writer = QwenStoryWriter(self.config)
        self._notify(progress_callback, 0.34, "Writing the bedtime story")
        story = self._with_schema_retries(
            "story package",
            lambda attempt_index: writer.write_story(description, attempt_index),
        )
        assert isinstance(story, StoryPackage)
        self._save_story(run_paths, story)
        print(f"[pipeline] Story saved with {len(story.parts)} parts: {run_paths.story_path}")
        writer.unload()

        image_generator = SSD1BSceneGenerator(self.config)
        self._notify(progress_callback, 0.56, "Generating storybook scenes")
        story = image_generator.generate(story, run_paths.images_dir)
        self._save_story(run_paths, story)
        print("[pipeline] Scene images generated")
        image_generator.unload()

        self._notify(progress_callback, 0.78, "Narrating the story")
        story = self._narrator.narrate(story, run_paths.audio_dir, run_paths.narration_audio_path)
        self._save_story(run_paths, story)
        print(f"[pipeline] Narration audio saved: {run_paths.narration_audio_path}")

        self._notify(progress_callback, 0.9, "Syncing playback timeline")
        timeline = build_timeline(story, str(run_paths.narration_audio_path.resolve()))
        write_json(run_paths.timeline_path, timeline.to_dict())
        print(f"[pipeline] Timeline saved: {run_paths.timeline_path}")

        manifest = build_run_manifest(run_paths, story)
        write_json(run_paths.manifest_path, manifest.to_dict())

        story_markdown = build_story_markdown(story)
        playback_html = build_playback_panel_html(story, timeline)

        self._notify(progress_callback, 1.0, "Story ready")
        print("[pipeline] Story generation complete")
        return PipelineResult(
            run_id=run_paths.run_id,
            run_dir=run_paths.run_dir.resolve(),
            description=description,
            story=story,
            timeline=timeline,
            manifest=manifest,
            story_markdown=story_markdown,
            playback_html=playback_html,
            image_gallery=[part.image_path for part in story.parts],
            narration_audio_path=str(run_paths.narration_audio_path.resolve()),
            manifest_path=str(run_paths.manifest_path.resolve()),
        )
