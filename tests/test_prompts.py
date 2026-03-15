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
    def test_build_description_prompt_targets_concise_visible_scene_facts(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("Describe only the visible scene in one concise paragraph", prompt)
        self.assertIn("Include uniquely identifiable objects", prompt)
        self.assertIn("Use only directly visible facts", prompt)
        self.assertIn("Do not mention the image, medium, artist, style", prompt)
        self.assertIn("Leave out anything not clearly visible", prompt)

    def test_build_story_part_1_prompt_uses_scene_as_only_source_of_facts(self):
        prompt = build_story_part_1_prompt(
            "A blue house with a red roof stands beside two green trees and a blue car."
        )
        self.assertIn("part 1 of a three-part bedtime story", prompt)
        self.assertIn("Scene description:", prompt)
        self.assertIn("blue house with a red roof", prompt)
        self.assertIn("whole story world", prompt)
        self.assertIn("Begin in the exact same scene", prompt)
        self.assertIn("specific described details actively", prompt)
        self.assertIn("Do not add any new detail not explicit in the description", prompt)
        self.assertIn("small gentle point of curiosity", prompt)
        self.assertIn("Do not mention AI, prompts, instructions", prompt)
        self.assertIn("Write about 50 words", prompt)
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
        self.assertIn("Stay in the exact same scene", messages[3]["content"])
        self.assertIn("details explicit in the description and part 1", messages[3]["content"])
        self.assertIn("Do not add any new detail not explicit in the description", messages[3]["content"])
        self.assertIn("Write about 50 words", messages[3]["content"])

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
        self.assertIn("Stay in the exact same scene", messages[5]["content"])
        self.assertIn("details explicit in the description and earlier parts", messages[5]["content"])
        self.assertIn("Do not add any new detail not explicit in the description", messages[5]["content"])
        self.assertIn("End with a calm, hopeful, bedtime-safe feeling", messages[5]["content"])
        self.assertIn("Write about 50 words", messages[5]["content"])

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
