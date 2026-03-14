import tempfile
import unittest
from pathlib import Path

from story_app.assets import RunPaths, build_run_manifest
from story_app.schemas import StoryPackage, StoryPart


class AssetsTests(unittest.TestCase):
    def test_build_run_manifest_collects_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_paths = RunPaths(
                run_id="demo123",
                run_dir=root,
                input_image_path=root / "input.png",
                description_path=root / "description.json",
                story_path=root / "story.json",
                timeline_path=root / "timeline.json",
                narration_audio_path=root / "audio" / "story.wav",
                manifest_path=root / "manifest.json",
                images_dir=root / "images",
                audio_dir=root / "audio",
            )
            story = StoryPackage(
                title="Soft Stars",
                age_range="5-10",
                parts=[
                    StoryPart("intro", "text", image_path=str(root / "images" / "scene_1.png"), audio_path=str(root / "audio" / "part_1.wav")),
                    StoryPart("middle", "text", image_path=str(root / "images" / "scene_2.png"), audio_path=str(root / "audio" / "part_2.wav")),
                    StoryPart("ending", "text", image_path=str(root / "images" / "scene_3.png"), audio_path=str(root / "audio" / "part_3.wav")),
                ],
            )
            manifest = build_run_manifest(run_paths, story)
            self.assertEqual(manifest.run_id, "demo123")
            self.assertEqual(len(manifest.scene_image_paths), 3)
            self.assertTrue(manifest.narration_audio_path.endswith("story.wav"))


if __name__ == "__main__":
    unittest.main()
