import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from story_app.schemas import DescriptionResult, PipelineResult, StoryDraft

if find_spec("gradio") is not None:
    import story_app.app as app_module
else:
    app_module = None


class FakePipeline:
    def __init__(self, result):
        self.result = result

    def create_story_draft(self, image_path, progress_callback=None):
        return self.result


@unittest.skipIf(app_module is None, "gradio is not installed in the local test environment")
class AppTests(unittest.TestCase):
    def test_generate_story_returns_outputs_in_expected_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "input.png"
            image_path.write_bytes(b"image")
            result = PipelineResult(
                run_id="run_1",
                run_dir=root,
                input_image_path=str(image_path),
                description=DescriptionResult(
                    image_path=str(image_path),
                    prompt_text="Describe this child's drawing.",
                    description_text="A calm blue house with a red roof stands between two green trees beside a blue car.",
                ),
                draft=StoryDraft(
                    full_conversation_text="SYSTEM:\nStory system\n\nUSER:\nWrite the first part.\n\nASSISTANT:\nPart 1.",
                    part_1_text="Part 1 text.",
                    part_2_text="Part 2 text.",
                    part_3_text="Part 3 text.",
                ),
                full_conversation_text="SYSTEM:\nStory system\n\nUSER:\nWrite the first part.\n\nASSISTANT:\nPart 1.",
                part_1_text="Part 1 text.",
                part_2_text="Part 2 text.",
                part_3_text="Part 3 text.",
            )
            with patch.object(app_module, "_PIPELINE", FakePipeline(result)):
                outputs = app_module.generate_story(str(image_path))

            self.assertEqual(len(outputs), 8)
            self.assertEqual(outputs[1], str(image_path))
            self.assertIn("A calm blue house", outputs[2])
            self.assertIn("SYSTEM:", outputs[3])
            self.assertEqual(outputs[4], "Part 1 text.")
            self.assertEqual(outputs[5], "Part 2 text.")
            self.assertEqual(outputs[6], "Part 3 text.")


if __name__ == "__main__":
    unittest.main()
