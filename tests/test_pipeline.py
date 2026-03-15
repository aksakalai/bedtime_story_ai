import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import DrawingDescription, StoryPackage, StoryPart


VALID_DESCRIPTION = DrawingDescription(
    text="A rabbit stands beside a moonlit pond and listens to the quiet water under silver stars.",
)


def make_valid_story() -> StoryPackage:
    return StoryPackage(
        title="Rabbit and the Quiet Pond",
        age_range="5-10",
        parts=[
            StoryPart(
                scene_goal="entrance",
                story_text="The rabbit padded to the moonlit pond and felt the quiet night settle into a soft beginning.",
                image_prompt="entrance prompt",
            ),
            StoryPart(
                scene_goal="buildup",
                story_text="The moon shimmered on the water while the rabbit noticed reeds, ripples, and the gentle sounds of bedtime.",
                image_prompt="buildup prompt",
            ),
            StoryPart(
                scene_goal="ending",
                story_text="The rabbit curled near the pond, watched the silver glow turn sleepy, and rested in a calm dreamy hush.",
                image_prompt="ending prompt",
            ),
        ],
    )


class _BaseFakeDescriber:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, prompt_path=None, response_path=None):
        if prompt_path is not None:
            prompt_path.write_text("description prompt", encoding="utf-8")
        if response_path is not None:
            response_path.write_text("description response", encoding="utf-8")
        return VALID_DESCRIPTION

    def unload(self):
        return None


class _BaseFakeWriter:
    def __init__(self, config):
        self.config = config

    def write_story(self, description, prompt_path=None, response_path=None):
        if prompt_path is not None:
            prompt_path.write_text("story prompt", encoding="utf-8")
        if response_path is not None:
            response_path.write_text("story response", encoding="utf-8")
        return make_valid_story()

    def unload(self):
        return None


class _BaseFakeImageGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, story, images_dir):
        updated_parts = []
        for index, part in enumerate(story.parts, start=1):
            image_path = images_dir / f"scene_{index}.png"
            image_path.write_bytes(f"image-{index}".encode("utf-8"))
            updated_parts.append(
                StoryPart(
                    scene_goal=part.scene_goal,
                    story_text=part.story_text,
                    image_prompt=part.image_prompt,
                    image_path=str(image_path.resolve()),
                    audio_path=part.audio_path,
                    duration_sec=part.duration_sec,
                )
            )
        return StoryPackage(title=story.title, age_range=story.age_range, parts=updated_parts)

    def unload(self):
        return None


class _BaseFakeNarrator:
    def __init__(self, config):
        self.config = config

    def narrate(self, story, audio_dir, merged_audio_path):
        updated_parts = []
        for index, part in enumerate(story.parts, start=1):
            audio_path = audio_dir / f"part_{index}.wav"
            audio_path.write_bytes(f"audio-{index}".encode("utf-8"))
            updated_parts.append(
                StoryPart(
                    scene_goal=part.scene_goal,
                    story_text=part.story_text,
                    image_prompt=part.image_prompt,
                    image_path=part.image_path,
                    audio_path=str(audio_path.resolve()),
                    duration_sec=1.5 + index,
                )
            )
        merged_audio_path.write_bytes(b"merged-audio")
        return StoryPackage(title=story.title, age_range=story.age_range, parts=updated_parts)


def _fake_render_story_video(*, output_path, **_kwargs):
    output_path.write_bytes(b"video")
    return output_path


class PipelineTests(unittest.TestCase):
    def _create_input_file(self, root: Path) -> Path:
        input_path = root / "input.png"
        input_path.write_bytes(b"not-a-real-image")
        return input_path

    def test_pipeline_stops_before_image_generation_without_three_story_parts(self):
        class FakeWriter(_BaseFakeWriter):
            def write_story(self, description, prompt_path=None, response_path=None):
                super().write_story(description, prompt_path=prompt_path, response_path=response_path)
                return StoryPackage(
                    title="Broken Story",
                    age_range="5-10",
                    parts=[
                        StoryPart("entrance", "Only one opening.", image_prompt="p1"),
                        StoryPart("buildup", "Only two parts.", image_prompt="p2"),
                    ],
                )

        class ShouldNotStartImageGenerator:
            def __init__(self, _config):
                raise AssertionError("image generation should not start")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = self._create_input_file(root)
            config = GenerationConfig(outputs_root=root / "outputs")
            with patch("story_app.pipeline.FlorenceDrawingDescriber", _BaseFakeDescriber), patch(
                "story_app.pipeline.QwenStoryWriter", FakeWriter
            ), patch("story_app.pipeline.SSD1BSceneGenerator", ShouldNotStartImageGenerator), patch(
                "story_app.pipeline.KokoroNarrator", _BaseFakeNarrator
            ):
                pipeline = KidStoryPipeline(config)
                with self.assertRaisesRegex(RuntimeError, "exactly 3 story parts"):
                    pipeline.create_story(input_path)

    def test_pipeline_stops_before_narration_when_scene_images_missing(self):
        class FakeImageGenerator(_BaseFakeImageGenerator):
            def generate(self, story, images_dir):
                image_path = images_dir / "scene_1.png"
                image_path.write_bytes(b"image-1")
                return StoryPackage(
                    title=story.title,
                    age_range=story.age_range,
                    parts=[
                        StoryPart("entrance", story.parts[0].story_text, image_prompt=story.parts[0].image_prompt, image_path=str(image_path.resolve())),
                        StoryPart("buildup", story.parts[1].story_text, image_prompt=story.parts[1].image_prompt, image_path=""),
                        StoryPart("ending", story.parts[2].story_text, image_prompt=story.parts[2].image_prompt, image_path=""),
                    ],
                )

        class ShouldNotNarrate(_BaseFakeNarrator):
            def narrate(self, story, audio_dir, merged_audio_path):
                raise AssertionError("narration should not start")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = self._create_input_file(root)
            config = GenerationConfig(outputs_root=root / "outputs")
            with patch("story_app.pipeline.FlorenceDrawingDescriber", _BaseFakeDescriber), patch(
                "story_app.pipeline.QwenStoryWriter", _BaseFakeWriter
            ), patch("story_app.pipeline.SSD1BSceneGenerator", FakeImageGenerator), patch(
                "story_app.pipeline.KokoroNarrator", ShouldNotNarrate
            ):
                pipeline = KidStoryPipeline(config)
                with self.assertRaisesRegex(RuntimeError, "Scene generation failed"):
                    pipeline.create_story(input_path)

    def test_pipeline_stops_before_video_when_audio_missing(self):
        class FakeNarrator(_BaseFakeNarrator):
            def narrate(self, story, audio_dir, merged_audio_path):
                updated_story = super().narrate(story, audio_dir, merged_audio_path)
                broken_parts = updated_story.parts[:]
                broken_parts[1] = StoryPart(
                    scene_goal=broken_parts[1].scene_goal,
                    story_text=broken_parts[1].story_text,
                    image_prompt=broken_parts[1].image_prompt,
                    image_path=broken_parts[1].image_path,
                    audio_path="",
                    duration_sec=0.0,
                )
                return StoryPackage(
                    title=updated_story.title,
                    age_range=updated_story.age_range,
                    parts=broken_parts,
                )

        def should_not_render(**_kwargs):
            raise AssertionError("video rendering should not start")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = self._create_input_file(root)
            config = GenerationConfig(outputs_root=root / "outputs")
            with patch("story_app.pipeline.FlorenceDrawingDescriber", _BaseFakeDescriber), patch(
                "story_app.pipeline.QwenStoryWriter", _BaseFakeWriter
            ), patch("story_app.pipeline.SSD1BSceneGenerator", _BaseFakeImageGenerator), patch(
                "story_app.pipeline.KokoroNarrator", FakeNarrator
            ), patch("story_app.pipeline.render_story_video", side_effect=should_not_render):
                pipeline = KidStoryPipeline(config)
                with self.assertRaisesRegex(RuntimeError, "Narration failed"):
                    pipeline.create_story(input_path)

    def test_pipeline_smoke_path_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = self._create_input_file(root)
            config = GenerationConfig(outputs_root=root / "outputs")
            with patch("story_app.pipeline.FlorenceDrawingDescriber", _BaseFakeDescriber), patch(
                "story_app.pipeline.QwenStoryWriter", _BaseFakeWriter
            ), patch("story_app.pipeline.SSD1BSceneGenerator", _BaseFakeImageGenerator), patch(
                "story_app.pipeline.KokoroNarrator", _BaseFakeNarrator
            ), patch("story_app.pipeline.render_story_video", side_effect=_fake_render_story_video):
                pipeline = KidStoryPipeline(config)
                result = pipeline.create_story(input_path)

            self.assertEqual(len(result.story.parts), 3)
            self.assertEqual(len(result.manifest.scene_image_paths), 3)
            self.assertEqual(len(result.manifest.part_audio_paths), 3)
            self.assertTrue(Path(result.narration_audio_path).exists())
            self.assertTrue(Path(result.video_path).exists())
            self.assertTrue(Path(result.manifest_path).exists())

            manifest_payload = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertTrue(Path(manifest_payload["description_prompt_path"]).exists())
            self.assertTrue(Path(manifest_payload["description_response_path"]).exists())
            self.assertTrue(Path(manifest_payload["story_prompt_path"]).exists())
            self.assertTrue(Path(manifest_payload["story_response_path"]).exists())
            self.assertEqual(len(manifest_payload["scene_image_paths"]), 3)
            self.assertEqual(len(manifest_payload["part_audio_paths"]), 3)


if __name__ == "__main__":
    unittest.main()
