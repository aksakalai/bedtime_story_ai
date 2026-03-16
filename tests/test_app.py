import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from story_app.schemas import DescriptionResult, PipelineResult, StoryDraft

if find_spec("gradio") is not None:
    import story_app.app as app_module
else:
    app_module = None


class FakePipeline:
    def __init__(self, result):
        self.result = result

    def create_story_package(self, image_path, progress_callback=None):
        return self.result


@unittest.skipIf(app_module is None, "gradio is not installed in the local test environment")
class AppTests(unittest.TestCase):
    def test_generate_story_returns_outputs_in_expected_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "input.png"
            Image.new("RGB", (8, 8), color=(200, 210, 220)).save(image_path)

            part_1_prompt_path = root / "image_prompt_part_1.txt"
            part_2_prompt_path = root / "image_prompt_part_2.txt"
            part_3_prompt_path = root / "image_prompt_part_3.txt"
            part_1_prompt_path.write_text("part 1 prompt", encoding="utf-8")
            part_2_prompt_path.write_text("part 2 prompt", encoding="utf-8")
            part_3_prompt_path.write_text("part 3 prompt", encoding="utf-8")

            part_1_image_path = root / "story_part_1_image.png"
            part_2_image_path = root / "story_part_2_image.png"
            part_3_image_path = root / "story_part_3_image.png"
            Image.new("RGB", (8, 8), color=(240, 220, 220)).save(part_1_image_path)
            Image.new("RGB", (8, 8), color=(220, 240, 220)).save(part_2_image_path)
            Image.new("RGB", (8, 8), color=(220, 220, 240)).save(part_3_image_path)

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
                image_prompt_part_1_path=str(part_1_prompt_path),
                image_prompt_part_2_path=str(part_2_prompt_path),
                image_prompt_part_3_path=str(part_3_prompt_path),
                story_part_1_image_path=str(part_1_image_path),
                story_part_2_image_path=str(part_2_image_path),
                story_part_3_image_path=str(part_3_image_path),
                story_part_1_audio_path=str(root / "story_part_1_audio.wav"),
                story_part_2_audio_path=str(root / "story_part_2_audio.wav"),
                story_part_3_audio_path=str(root / "story_part_3_audio.wav"),
                final_story_video_path=str(root / "final_story_video.mp4"),
            )
            with patch.object(app_module, "_PIPELINE", FakePipeline(result)):
                outputs = app_module.generate_story(str(image_path))

            self.assertEqual(len(outputs), 18)
            self.assertIn("Run `run_1` completed.", outputs[0])
            self.assertIsNotNone(outputs[1])
            self.assertIn("A calm blue house", outputs[2])
            self.assertIn("SYSTEM:", outputs[3])
            self.assertEqual(outputs[4], "Part 1 text.")
            self.assertEqual(outputs[5], "part 1 prompt")
            self.assertIsNotNone(outputs[6])
            self.assertEqual(outputs[7], str(root / "story_part_1_audio.wav"))
            self.assertEqual(outputs[8], "Part 2 text.")
            self.assertEqual(outputs[9], "part 2 prompt")
            self.assertIsNotNone(outputs[10])
            self.assertEqual(outputs[11], str(root / "story_part_2_audio.wav"))
            self.assertEqual(outputs[12], "Part 3 text.")
            self.assertEqual(outputs[13], "part 3 prompt")
            self.assertIsNotNone(outputs[14])
            self.assertEqual(outputs[15], str(root / "story_part_3_audio.wav"))
            self.assertEqual(outputs[16], str(root / "final_story_video.mp4"))
            self.assertEqual(outputs[17], str(root.resolve()))


if __name__ == "__main__":
    unittest.main()
