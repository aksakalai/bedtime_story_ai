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
    def test_build_description_prompt_uses_clean_caption_prefix(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("exact visible details", prompt)
        self.assertIn("colors", prompt)
        self.assertIn("Do not invent hidden actions", prompt)

    def test_build_story_part_1_prompt_contains_master_instruction_and_description(self):
        prompt = build_story_part_1_prompt(
            "A blue house with a red roof stands beside two green trees and a blue car."
        )
        self.assertIn("three-part bedtime story", prompt)
        self.assertIn("Description of the image:", prompt)
        self.assertIn("blue house with a red roof", prompt)
        self.assertIn("Part 1 introduces", prompt)
        self.assertIn("Do not mention AI, prompts, instructions", prompt)
        self.assertIn("Now write the first part of the story.", prompt)

    def test_build_story_messages_for_part_2_uses_short_follow_up_turn(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            previous_parts=["The rabbit watched the still water shimmer under the moon."],
        )
        self.assertEqual(messages[0]["content"], STORY_SYSTEM_PROMPT)
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "The rabbit watched the still water shimmer under the moon.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], PART_2_USER_PROMPT)
        self.assertNotIn("Description of the image:", messages[3]["content"])

    def test_build_story_messages_for_part_3_uses_short_final_turn(self):
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
        self.assertNotIn("Description of the image:", messages[5]["content"])

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
