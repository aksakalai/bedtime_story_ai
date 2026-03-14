import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import build_character_bible, build_story_prompt, enrich_story_with_image_prompts
from story_app.schemas import DrawingDescription, StoryPackage, StoryPart


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.description = DrawingDescription(
            summary="A rabbit sailing under stars",
            characters=["rabbit", "paper boat"],
            setting="a sparkling lake at night",
            visual_style="colored pencil storybook",
            color_palette=["navy", "silver", "peach"],
            safety_notes=["keep it calm", "no danger", "soft bedtime tone"],
        )
        self.story = StoryPackage(
            title="Luna and the Quiet Lake",
            age_range="5-10",
            parts=[
                StoryPart(scene_goal="beginning", story_text="Part one text", image_prompt="rabbit in boat"),
                StoryPart(scene_goal="middle", story_text="Part two text", image_prompt="boat under stars"),
                StoryPart(scene_goal="ending", story_text="Part three text", image_prompt="rabbit going to sleep"),
            ],
        )

    def test_build_story_prompt_includes_schema_and_safety_rules(self):
        prompt = build_story_prompt(self.description, DEFAULT_CONFIG)
        self.assertIn("exactly 3 objects", prompt)
        self.assertIn("Calm, cozy, and bedtime-friendly", prompt)
        self.assertIn("A rabbit sailing under stars", prompt)

    def test_build_character_bible_uses_shared_visual_details(self):
        bible = build_character_bible(self.description)
        self.assertIn("rabbit, paper boat", bible)
        self.assertIn("sparkling lake at night", bible)
        self.assertIn("navy, silver, peach", bible)

    def test_enrich_story_with_image_prompts_adds_shared_consistency(self):
        enriched = enrich_story_with_image_prompts(self.story, self.description, DEFAULT_CONFIG)
        self.assertEqual(len(enriched.parts), 3)
        self.assertTrue(all("consistent recurring characters" in part.image_prompt for part in enriched.parts))
        self.assertTrue(all(f"scene {index} of 3" in part.image_prompt for index, part in enumerate(enriched.parts, start=1)))


if __name__ == "__main__":
    unittest.main()
