import tempfile
import unittest
from pathlib import Path

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import ValidationError


VALID_DESCRIPTION = (
    "A gentle drawing of a small blue house with a red roof beside two green trees and a little blue car under a bright yellow sun."
)
VALID_PART_1 = (
    "Mina stood beside the small blue house with the red roof and watched the bright yellow sun glow above the two green trees. A little blue car rested nearby, and the whole yard felt quiet and welcoming as she wondered what gentle adventure the day might bring."
)
VALID_PART_2 = (
    "A soft breeze shook the two green trees, and Mina noticed a folded paper tucked beneath the little blue car. She picked it up beside the blue house and found a kind note inviting her to follow the warm sunlight to a small surprise waiting in the garden."
)
VALID_PART_3 = (
    "Mina followed the sunlight to the shady spot between the trees, where a basket of sweet berries waited beside the blue house. She smiled at the little blue car, thanked the bright yellow sun for guiding her, and ended the day feeling peaceful and glad."
)


class FakeDescriber:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, prompt_text):
        return VALID_DESCRIPTION

    def unload(self):
        return None


class FakeWriter:
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def generate_part(self, messages):
        self.calls += 1
        if self.calls == 1:
            return VALID_PART_1
        if self.calls == 2:
            return VALID_PART_2
        return VALID_PART_3

    def unload(self):
        return None


class NonEosDescription:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, prompt_text):
        raise ValidationError("Description generation did not finish naturally before the safety limit.")

    def unload(self):
        return None


class NonEosWriter:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = 0
        NonEosWriter.instances.append(self)

    def generate_part(self, messages):
        self.calls += 1
        if self.calls == 1:
            return VALID_PART_1
        if self.calls == 2:
            raise ValidationError("Story generation did not finish naturally before the safety limit.")
        return VALID_PART_3

    def unload(self):
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
                describer_factory=NonEosDescription,
                writer_factory=FakeWriter,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_stops_before_part_3_when_part_2_does_not_finish_with_eos(self):
        NonEosWriter.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=FakeDescriber,
                writer_factory=NonEosWriter,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))
            self.assertEqual(NonEosWriter.instances[0].calls, 2)

    def test_pipeline_smoke_path_saves_minimal_story_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=FakeDescriber,
                writer_factory=FakeWriter,
            )
            result = pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(result.description.description_text, VALID_DESCRIPTION)
            self.assertEqual(result.part_1_text, VALID_PART_1)
            self.assertEqual(result.part_2_text, VALID_PART_2)
            self.assertEqual(result.part_3_text, VALID_PART_3)
            self.assertIn("SYSTEM:", result.full_conversation_text)
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


if __name__ == "__main__":
    unittest.main()
