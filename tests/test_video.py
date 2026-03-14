import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image, ImageChops
    from story_app.video import _render_story_frame
except ModuleNotFoundError:  # pragma: no cover - local fallback for bare Python installs
    Image = None
    ImageChops = None
    _render_story_frame = None


@unittest.skipIf(Image is None, "Pillow is not installed in the local test environment.")
class VideoFrameTests(unittest.TestCase):
    def test_render_story_frame_changes_when_highlight_progress_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "scene.png"
            frame_a_path = tmp / "frame_a.png"
            frame_b_path = tmp / "frame_b.png"

            Image.new("RGB", (900, 700), color=(70, 120, 170)).save(image_path)

            _render_story_frame(
                image_path=str(image_path),
                title="Dream Castle",
                story_text="Tarun drew a sleepy castle by the water while the moon watched quietly.",
                part_label="Part 1 of 3",
                highlighted_words=2,
                progress_ratio=0.15,
                output_path=frame_a_path,
            )
            _render_story_frame(
                image_path=str(image_path),
                title="Dream Castle",
                story_text="Tarun drew a sleepy castle by the water while the moon watched quietly.",
                part_label="Part 1 of 3",
                highlighted_words=10,
                progress_ratio=0.85,
                output_path=frame_b_path,
            )

            with Image.open(frame_a_path) as frame_a, Image.open(frame_b_path) as frame_b:
                difference = ImageChops.difference(frame_a, frame_b)
                self.assertIsNotNone(difference.getbbox())


if __name__ == "__main__":
    unittest.main()
