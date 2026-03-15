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
                description_prompt_path=root / "description_prompt.txt",
                description_response_path=root / "description_response.txt",
                story_path=root / "story.json",
                story_prompt_path=root / "story_prompt.txt",
                story_response_path=root / "story_response.txt",
                timeline_path=root / "timeline.json",
                narration_audio_path=root / "audio" / "story.wav",
                video_path=root / "video" / "story.mp4",
                manifest_path=root / "manifest.json",
                images_dir=root / "images",
                audio_dir=root / "audio",
                video_dir=root / "video",
            )
            story = StoryPackage(
                title="Soft Stars",
                age_range="5-10",
                parts=[
                    StoryPart("entrance", "text", image_path=str(root / "images" / "scene_1.png"), audio_path=str(root / "audio" / "part_1.wav")),
                    StoryPart("buildup", "text", image_path=str(root / "images" / "scene_2.png"), audio_path=str(root / "audio" / "part_2.wav")),
                    StoryPart("ending", "text", image_path=str(root / "images" / "scene_3.png"), audio_path=str(root / "audio" / "part_3.wav")),
                ],
            )
            manifest = build_run_manifest(run_paths, story)
            self.assertEqual(manifest.run_id, "demo123")
            self.assertEqual(len(manifest.scene_image_paths), 3)
            self.assertTrue(manifest.description_prompt_path.endswith("description_prompt.txt"))
            self.assertTrue(manifest.story_response_path.endswith("story_response.txt"))
            self.assertTrue(manifest.narration_audio_path.endswith("story.wav"))
            self.assertTrue(manifest.video_path.endswith("story.mp4"))


if __name__ == "__main__":
    unittest.main()
