import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    PART_2_USER_PROMPT,
    PART_3_USER_PROMPT,
    STORY_SYSTEM_PROMPT,
    build_description_prompt,
    build_story_messages,
    build_story_part_1_prompt,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_targets_rich_scene_detail_not_meta_commentary(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("Describe the visible scene in one rich, precise paragraph", prompt)
        self.assertIn("Include as many directly visible, uniquely identifiable details as possible", prompt)
        self.assertIn("Prefer exact scene details over broad summaries", prompt)
        self.assertIn("Do not mention the image itself, the medium, the artist, style", prompt)
        self.assertIn("If a detail is not clearly visible, leave it out", prompt)

    def test_build_story_part_1_prompt_uses_scene_as_only_source_of_facts(self):
        prompt = build_story_part_1_prompt(
            "A blue house with a red roof stands beside two green trees and a blue car."
        )
        self.assertIn("part 1 of a three-part bedtime story", prompt)
        self.assertIn("Scene description:", prompt)
        self.assertIn("blue house with a red roof", prompt)
        self.assertIn("only source of story facts", prompt)
        self.assertIn("Begin inside the exact same scene", prompt)
        self.assertIn("actively and specifically", prompt)
        self.assertIn("Do not introduce any new character, object, scenery element, location", prompt)
        self.assertIn("small gentle point of curiosity", prompt)
        self.assertIn("Do not mention AI, prompts, instructions", prompt)
        self.assertIn("Now write only part 1.", prompt)

    def test_build_story_messages_for_part_2_uses_grounded_follow_up_turn(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            previous_parts=["The rabbit watched the still water shimmer under the moon."],
        )
        self.assertEqual(messages[0]["content"], STORY_SYSTEM_PROMPT)
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "The rabbit watched the still water shimmer under the moon.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], PART_2_USER_PROMPT)
        self.assertIn("Stay in the same scene", messages[3]["content"])
        self.assertIn("Keep reusing the specific scene details", messages[3]["content"])
        self.assertIn("Do not introduce any new character, object, scenery element, location", messages[3]["content"])
        self.assertIn("time jump", messages[3]["content"])

    def test_build_story_messages_for_part_3_uses_grounded_final_turn(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            previous_parts=[
                "The rabbit watched the still water shimmer under the moon.",
                "A silver fish surfaced once and left tiny rings drifting outward.",
            ],
        )
        self.assertEqual(messages[4]["role"], "assistant")
        self.assertEqual(messages[4]["content"], "A silver fish surfaced once and left tiny rings drifting outward.")
        self.assertEqual(messages[5]["role"], "user")
        self.assertEqual(messages[5]["content"], PART_3_USER_PROMPT)
        self.assertIn("Stay in the same scene", messages[5]["content"])
        self.assertIn("Keep reusing the specific scene details", messages[5]["content"])
        self.assertIn("End with a calm, hopeful, bedtime-safe feeling", messages[5]["content"])
        self.assertIn("time jump", messages[5]["content"])

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
