from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable

from .assets import RunPaths, build_run_manifest, prepare_run_paths, write_json
from .config import DEFAULT_CONFIG, GenerationConfig
from .playback import build_playback_panel_html, build_timeline
from .prompts import build_story_markdown
from .providers import KokoroNarrator, QwenStoryWriter, SSD1BSceneGenerator, SmolVLMDescriber
from .schemas import EXPECTED_SCENE_GOALS, DrawingDescription, PipelineResult, StoryPackage
from .video import render_story_video

ProgressCallback = Callable[[float, str], None]


def _seed_everything(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class KidStoryPipeline:
    def __init__(self, config: GenerationConfig = DEFAULT_CONFIG):
        self.config = config
        self._narrator = KokoroNarrator(config)

    def _notify(self, progress_callback: ProgressCallback | None, value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def _fail_stage(self, stage: str, error: Exception, artifacts: Iterable[Path]) -> RuntimeError:
        artifact_text = ", ".join(str(path.resolve()) for path in artifacts)
        return RuntimeError(
            f"{stage} failed: {error}. Check these artifacts: {artifact_text}"
        )

    def _validate_description(self, description: DrawingDescription) -> None:
        if not description.summary.strip():
            raise RuntimeError("Drawing description summary must not be empty.")
        if not description.characters:
            raise RuntimeError("Drawing description must include at least one character or object.")
        if not description.color_palette:
            raise RuntimeError("Drawing description must include at least one color.")
        if not description.safety_notes:
            raise RuntimeError("Drawing description must include at least one safety note.")

    def _validate_story(
        self,
        story: StoryPackage,
        *,
        require_image_prompts: bool = False,
        require_images: bool = False,
        require_audio: bool = False,
    ) -> None:
        if len(story.parts) != 3:
            raise RuntimeError("Story must contain exactly 3 story parts.")

        for expected_goal, part in zip(EXPECTED_SCENE_GOALS, story.parts):
            if part.scene_goal != expected_goal:
                raise RuntimeError(
                    f"Story parts must be ordered as {', '.join(EXPECTED_SCENE_GOALS)}."
                )
            if not part.story_text.strip():
                raise RuntimeError(f"Story part '{expected_goal}' must not be empty.")
            if require_image_prompts and not part.image_prompt.strip():
                raise RuntimeError(f"Story part '{expected_goal}' is missing an image prompt.")
            if require_images:
                if not part.image_path.strip():
                    raise RuntimeError(f"Story part '{expected_goal}' is missing an image path.")
                if not Path(part.image_path).exists():
                    raise RuntimeError(
                        f"Scene image for '{expected_goal}' was not created at {part.image_path}."
                    )
            if require_audio:
                if not part.audio_path.strip():
                    raise RuntimeError(f"Story part '{expected_goal}' is missing an audio path.")
                if not Path(part.audio_path).exists():
                    raise RuntimeError(
                        f"Narration audio for '{expected_goal}' was not created at {part.audio_path}."
                    )
                if part.duration_sec <= 0:
                    raise RuntimeError(
                        f"Narration audio for '{expected_goal}' must have a positive duration."
                    )

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
        _seed_everything(self.config.random_seed)
        print(f"[pipeline] Run directory: {run_paths.run_dir}")

        describer = SmolVLMDescriber(self.config)
        try:
            self._notify(progress_callback, 0.12, "Describing the drawing")
            description = describer.describe(
                run_paths.input_image_path,
                prompt_path=run_paths.description_prompt_path,
                response_path=run_paths.description_response_path,
            )
            self._validate_description(description)
            write_json(run_paths.description_path, description.to_dict())
            print(f"[pipeline] Drawing description saved: {run_paths.description_path}")
        except Exception as exc:
            raise self._fail_stage(
                "Drawing description",
                exc,
                [run_paths.description_prompt_path, run_paths.description_response_path],
            ) from exc
        finally:
            describer.unload()

        writer = QwenStoryWriter(self.config)
        try:
            self._notify(progress_callback, 0.34, "Writing the bedtime story")
            story = writer.write_story(
                description,
                prompt_path=run_paths.story_prompt_path,
                response_path=run_paths.story_response_path,
            )
            self._validate_story(story, require_image_prompts=True)
            self._save_story(run_paths, story)
            print(f"[pipeline] Story saved with {len(story.parts)} parts: {run_paths.story_path}")
        except Exception as exc:
            raise self._fail_stage(
                "Story generation",
                exc,
                [
                    run_paths.story_prompt_path,
                    run_paths.story_response_path,
                    run_paths.story_path,
                ],
            ) from exc
        finally:
            writer.unload()

        image_generator = SSD1BSceneGenerator(self.config)
        try:
            self._notify(progress_callback, 0.56, "Generating storybook scenes")
            story = image_generator.generate(story, run_paths.images_dir)
            self._validate_story(story, require_image_prompts=True, require_images=True)
            self._save_story(run_paths, story)
            print("[pipeline] Scene images generated")
        except Exception as exc:
            raise self._fail_stage("Scene generation", exc, [run_paths.story_path]) from exc
        finally:
            image_generator.unload()

        try:
            self._notify(progress_callback, 0.78, "Narrating the story")
            story = self._narrator.narrate(story, run_paths.audio_dir, run_paths.narration_audio_path)
            self._validate_story(
                story,
                require_image_prompts=True,
                require_images=True,
                require_audio=True,
            )
            if not run_paths.narration_audio_path.exists():
                raise RuntimeError(
                    f"Merged narration audio was not created at {run_paths.narration_audio_path}."
                )
            self._save_story(run_paths, story)
            print(f"[pipeline] Narration audio saved: {run_paths.narration_audio_path}")
        except Exception as exc:
            raise self._fail_stage(
                "Narration",
                exc,
                [run_paths.story_path, run_paths.narration_audio_path],
            ) from exc

        try:
            self._notify(progress_callback, 0.9, "Syncing playback timeline")
            timeline = build_timeline(story, str(run_paths.narration_audio_path.resolve()))
            write_json(run_paths.timeline_path, timeline.to_dict())
            print(f"[pipeline] Timeline saved: {run_paths.timeline_path}")
        except Exception as exc:
            raise self._fail_stage(
                "Timeline build",
                exc,
                [run_paths.story_path, run_paths.narration_audio_path, run_paths.timeline_path],
            ) from exc

        try:
            self._notify(progress_callback, 0.96, "Rendering story video")
            render_story_video(
                story=story,
                timeline=timeline,
                output_path=run_paths.video_path,
                work_dir=run_paths.video_dir,
            )
            print(f"[pipeline] Story video saved: {run_paths.video_path}")
        except Exception as exc:
            raise self._fail_stage(
                "Video render",
                exc,
                [run_paths.timeline_path, run_paths.narration_audio_path, run_paths.video_path],
            ) from exc

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
            video_path=str(run_paths.video_path.resolve()),
            image_gallery=[part.image_path for part in story.parts],
            narration_audio_path=str(run_paths.narration_audio_path.resolve()),
            manifest_path=str(run_paths.manifest_path.resolve()),
        )
