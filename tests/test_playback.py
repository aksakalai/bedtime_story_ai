import tempfile
import unittest
from pathlib import Path

from story_app.playback import build_playback_panel_html, build_timeline
from story_app.schemas import StoryPackage, StoryPart


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.story = StoryPackage(
            title="Bedtime on the Hill",
            age_range="5-10",
            parts=[
                StoryPart("intro", "The rabbit watched the sky turn peach.", image_path="/tmp/scene1.png", duration_sec=1.2),
                StoryPart("middle", "A sleepy owl hummed from a branch.", image_path="/tmp/scene2.png", duration_sec=2.3),
                StoryPart("ending", "The hill tucked everyone into moonlight.", image_path="/tmp/scene3.png", duration_sec=3.4),
            ],
        )

    def test_build_timeline_accumulates_segment_boundaries(self):
        timeline = build_timeline(self.story, "/tmp/story.wav")
        self.assertEqual(timeline.total_duration_sec, 6.9)
        self.assertEqual(timeline.segments[0].start_sec, 0.0)
        self.assertEqual(timeline.segments[1].start_sec, 1.2)
        self.assertEqual(timeline.segments[2].end_sec, 6.9)

    def test_build_playback_panel_embeds_audio_and_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_paths = []
            for index in range(1, 4):
                image_path = tmp / f"scene_{index}.png"
                image_path.write_bytes(b"fake-image")
                image_paths.append(str(image_path))
            audio_path = tmp / "story.wav"
            audio_path.write_bytes(b"fake-audio")

            story = StoryPackage(
                title=self.story.title,
                age_range=self.story.age_range,
                parts=[
                    StoryPart(part.scene_goal, part.story_text, image_path=image_paths[idx], duration_sec=part.duration_sec)
                    for idx, part in enumerate(self.story.parts)
                ],
            )
            timeline = build_timeline(story, str(audio_path))
            html = build_playback_panel_html(story, timeline)
            self.assertIn("data:image/png;base64", html)
            self.assertIn("data:audio/wav;base64", html)
            self.assertIn("Bedtime on the Hill", html)


if __name__ == "__main__":
    unittest.main()
