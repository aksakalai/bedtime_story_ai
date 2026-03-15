import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from story_app.schemas import DescriptionResult, PipelineResult, StoryDraft, StoryStep

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
    def test_generate_story_returns_debug_outputs_in_expected_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "input.png"
            image_path.write_bytes(b"image")
            result = PipelineResult(
                run_id="run_1",
                run_dir=root,
                input_image_path=str(image_path),
                description_text="A calm house beside two trees and a little car.",
                part_1_text="Part 1 text with enough words to look like a real story opening in the debug UI.",
                part_2_text="Part 2 text with enough words to look like a real story middle in the debug UI.",
                part_3_text="Part 3 text with enough words to look like a real story ending in the debug UI.",
                draft=StoryDraft(
                    description=DescriptionResult(
                        image_path=str(image_path),
                        prompt_text="a child's drawing of",
                        description_text="A calm house beside two trees and a little car.",
                    ),
                    steps=[
                        StoryStep("part_1", "Input Slice 1", "Part 1 text with enough words to look like a real story opening in the debug UI."),
                        StoryStep("part_2", "Input Slice 2", "Part 2 text with enough words to look like a real story middle in the debug UI."),
                        StoryStep("part_3", "Input Slice 3", "Part 3 text with enough words to look like a real story ending in the debug UI."),
                    ],
                    full_conversation_text="SYSTEM:\nStory system\n\nUSER:\nPart 1 input\n\nASSISTANT:\nPart 1 output",
                ),
            )
            with patch.object(app_module, "_PIPELINE", FakePipeline(result)):
                outputs = app_module.generate_story(str(image_path))

            self.assertEqual(len(outputs), 11)
            self.assertEqual(outputs[1], str(image_path))
            self.assertIn("A calm house", outputs[2])
            self.assertIn("SYSTEM:", outputs[3])
            self.assertEqual(outputs[4], "Input Slice 1")
            self.assertEqual(outputs[6], "Input Slice 2")
            self.assertEqual(outputs[8], "Input Slice 3")


if __name__ == "__main__":
    unittest.main()
