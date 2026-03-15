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
    def test_build_description_prompt_uses_scene_focused_caption_prefix(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("plain visual scene", prompt)
        self.assertIn("scenery and layout", prompt)
        self.assertIn("concrete spatial wording", prompt)
        self.assertIn("Do not add opinions", prompt)

    def test_build_story_part_1_prompt_contains_grounded_opening_rules(self):
        prompt = build_story_part_1_prompt(
            "A blue house with a red roof stands beside two green trees and a blue car."
        )
        self.assertIn("part 1 of a three-part bedtime story", prompt)
        self.assertIn("Description of the image:", prompt)
        self.assertIn("blue house with a red roof", prompt)
        self.assertIn("opening scene of a story, not a caption or checklist", prompt)
        self.assertIn("Do not introduce major new characters, locations, or props", prompt)
        self.assertIn("small point of curiosity", prompt)
        self.assertIn("Do not mention AI, prompts, instructions", prompt)
        self.assertIn("Now write only part 1 of the story.", prompt)

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
        self.assertIn("same setting", messages[3]["content"])
        self.assertIn("Do not introduce major new characters", messages[3]["content"])
        self.assertIn("Avoid sudden time jumps", messages[3]["content"])

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
        self.assertIn("same setting", messages[5]["content"])
        self.assertIn("End with a calm, hopeful, bedtime-safe feeling", messages[5]["content"])
        self.assertIn("Avoid sudden time jumps", messages[5]["content"])

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
