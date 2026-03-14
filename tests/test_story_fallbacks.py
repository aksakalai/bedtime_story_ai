import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.providers import QwenStoryWriter
from story_app.schemas import DrawingDescription


class StoryFallbackTests(unittest.TestCase):
    def setUp(self):
        self.description = DrawingDescription(
            summary="A child drew a tall castle by the water.",
            characters=["Tarun", "castle", "seagull"],
            setting="a bright seaside castle at sunset",
            visual_style="crayon drawing",
            color_palette=["red", "blue", "gold"],
            safety_notes=["keep it gentle", "no scary moments", "quiet bedtime ending"],
        )
        self.writer = QwenStoryWriter(DEFAULT_CONFIG)

    def test_coerce_story_package_accepts_part_keys_without_parts_list(self):
        raw_text = """
        {
          "title": "Draw Your Dream Castle",
          "age_range": "5-10",
          "part_1": "Tarun began drawing his castle beside the sea.",
          "part_2": "A soft breeze and a seagull turned the picture into a dream.",
          "part_3": "At the end, the whole castle glowed quietly for bedtime."
        }
        """
        story = self.writer._coerce_story_package(raw_text, self.description)
        self.assertEqual(len(story.parts), 3)
        self.assertEqual(story.title, "Draw Your Dream Castle")

    def test_build_story_fallback_always_returns_three_parts(self):
        story = self.writer._build_story_fallback(self.description)
        self.assertEqual(len(story.parts), 3)
        self.assertTrue(all(part.story_text for part in story.parts))
        self.assertIn("Dreamy Night", story.title)


if __name__ == "__main__":
    unittest.main()
