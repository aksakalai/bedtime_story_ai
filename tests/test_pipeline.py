import tempfile
import unittest
from pathlib import Path

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import ValidationError


VALID_DESCRIPTION = (
    "A gentle drawing of a small house beside two trees and a little car under a calm evening sky."
)
VALID_PART_1 = (
    "A little fox walked past the small house at dusk and noticed the two trees swaying softly nearby while a tiny car rested by the path. The evening felt warm and calm, and the fox slowed down to listen to the quiet sounds of home before night settled in."
)
VALID_PART_2 = (
    "The fox followed the path between the trees and found a patch of glowing fireflies circling the house in patient loops. He watched them drift over the parked car and across the windows, and the whole place felt like a gentle secret waiting to be shared."
)
VALID_PART_3 = (
    "When the stars brightened, the fox curled beside the house and let the fireflies fade into the dark blue sky. The trees stood still, the little car gleamed softly, and the fox closed his eyes, happy to fall asleep in such a peaceful place."
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

    def generate_part(self, prompt_text):
        self.calls += 1
        if self.calls == 1:
            return VALID_PART_1
        if self.calls == 2:
            return VALID_PART_2
        return VALID_PART_3

    def unload(self):
        return None


class FailingDescription:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, prompt_text):
        return "```bad```"

    def unload(self):
        return None


class CountingWriter:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = 0
        CountingWriter.instances.append(self)

    def generate_part(self, prompt_text):
        self.calls += 1
        if self.calls == 1:
            return VALID_PART_1
        if self.calls == 2:
            return "Sure, here is the second part of the story."
        return VALID_PART_3

    def unload(self):
        return None


class PipelineTests(unittest.TestCase):
    def _create_input_file(self, root: Path) -> Path:
        path = root / "input.png"
        path.write_bytes(b"fake-image")
        return path

    def test_pipeline_stops_before_part_1_when_description_is_invalid(self):
        class ShouldNotWrite:
            def __init__(self, config):
                raise AssertionError("writer should not be created")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=FailingDescription,
                writer_factory=ShouldNotWrite,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_stops_before_part_3_when_part_2_is_invalid(self):
        CountingWriter.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=FakeDescriber,
                writer_factory=CountingWriter,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))
            self.assertEqual(CountingWriter.instances[0].calls, 2)

    def test_pipeline_smoke_path_saves_all_phase_1_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=FakeDescriber,
                writer_factory=FakeWriter,
            )
            result = pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(result.description_text, VALID_DESCRIPTION)
            self.assertEqual(result.part_1_text, VALID_PART_1)
            self.assertEqual(result.part_2_text, VALID_PART_2)
            self.assertEqual(result.part_3_text, VALID_PART_3)

            expected_files = [
                "description_prompt.txt",
                "description.txt",
                "story_part_1_prompt.txt",
                "story_part_1.txt",
                "story_part_2_prompt.txt",
                "story_part_2.txt",
                "story_part_3_prompt.txt",
                "story_part_3.txt",
            ]
            for filename in expected_files:
                self.assertTrue((result.run_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
