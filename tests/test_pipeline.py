import tempfile
import unittest
from pathlib import Path

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import ValidationError


VALID_DESCRIPTION = (
    "A small blue house with a red roof stands between two green trees beside a little blue car under a yellow sun."
)
VALID_PART_1 = (
    "The small blue house glowed softly under the yellow sun while the little blue car rested between the two green "
    "trees. Everything felt quiet and warm, and the stillness held one tiny question about what gentle moment might "
    "begin next."
)
VALID_PART_2 = (
    "A light breeze stirred the two green trees, and the little blue car seemed to wait patiently beside the house. "
    "The quiet scene felt full of promise, as if the warm sunlight were guiding the whole place toward a tender new "
    "moment."
)
VALID_PART_3 = (
    "By evening, the blue house, the two green trees, and the little blue car all rested beneath the fading yellow "
    "sun. The gentle scene settled into peace, and the day ended with a calm feeling that made everything seem safe "
    "and still."
)


class SharedFakeProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.description_calls = 0
        self.story_calls = 0
        self.unload_calls = 0
        SharedFakeProvider.instances.append(self)

    def describe(self, image_path, messages):
        self.description_calls += 1
        return VALID_DESCRIPTION

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        step_index = ((self.story_calls - 1) % 3) + 1
        if step_index == 1:
            return VALID_PART_1
        if step_index == 2:
            return VALID_PART_2
        return VALID_PART_3

    def unload(self, clear_cache=False):
        self.unload_calls += 1
        return None


class NonEosDescriptionProvider:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, messages):
        raise ValidationError("Description generation did not finish naturally before the safety limit.")

    def generate_part(self, image_path, messages):
        return VALID_PART_1

    def unload(self, clear_cache=False):
        return None


class NonEosStoryProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.story_calls = 0
        NonEosStoryProvider.instances.append(self)

    def describe(self, image_path, messages):
        return VALID_DESCRIPTION

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        if self.story_calls == 1:
            return VALID_PART_1
        raise ValidationError("Story generation did not finish naturally before the safety limit.")

    def unload(self, clear_cache=False):
        return None


class PipelineTests(unittest.TestCase):
    def _create_input_file(self, root: Path) -> Path:
        path = root / "input.png"
        path.write_bytes(b"fake-image")
        return path

    def test_pipeline_stops_when_description_does_not_finish_with_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosDescriptionProvider,
                writer_factory=NonEosDescriptionProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_stops_before_part_3_when_part_2_does_not_finish_with_eos(self):
        NonEosStoryProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosStoryProvider,
                writer_factory=NonEosStoryProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))
            self.assertEqual(NonEosStoryProvider.instances[0].story_calls, 2)

    def test_pipeline_smoke_path_saves_minimal_story_artifacts(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            result = pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(result.description.description_text, VALID_DESCRIPTION)
            self.assertEqual(result.part_1_text, VALID_PART_1)
            self.assertEqual(result.part_2_text, VALID_PART_2)
            self.assertEqual(result.part_3_text, VALID_PART_3)
            self.assertIn("SYSTEM:", result.full_conversation_text)
            self.assertIn("[IMAGE]", result.full_conversation_text)
            self.assertIn(VALID_DESCRIPTION, result.full_conversation_text)
            self.assertIn(VALID_PART_1, result.full_conversation_text)
            self.assertIn(VALID_PART_2, result.full_conversation_text)
            self.assertIn(VALID_PART_3, result.full_conversation_text)

            input_images = list(result.run_dir.glob("input_image*"))
            self.assertEqual(len(input_images), 1)

            expected_files = [
                "description_prompt.txt",
                "description.txt",
                "story_conversation.txt",
                "story_part_1.txt",
                "story_part_2.txt",
                "story_part_3.txt",
            ]
            for filename in expected_files:
                self.assertTrue((result.run_dir / filename).exists(), filename)

    def test_pipeline_reuses_single_provider_instance_across_runs(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            pipeline.create_story_draft(self._create_input_file(root))
            pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(len(SharedFakeProvider.instances), 1)
            self.assertEqual(SharedFakeProvider.instances[0].description_calls, 2)
            self.assertEqual(SharedFakeProvider.instances[0].story_calls, 6)
            self.assertEqual(SharedFakeProvider.instances[0].unload_calls, 0)

    def test_pipeline_can_clear_loaded_models(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            pipeline.create_story_draft(self._create_input_file(root))
            pipeline.clear_loaded_models()

            self.assertEqual(SharedFakeProvider.instances[0].unload_calls, 1)


if __name__ == "__main__":
    unittest.main()
