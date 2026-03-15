import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    DESCRIPTION_USER_PROMPT_SUFFIX,
    MULTIMODAL_SYSTEM_PROMPT,
    PART_1_USER_PROMPT,
    PART_2_USER_PROMPT,
    PART_3_USER_PROMPT,
    build_description_messages,
    build_description_prompt,
    build_image_text_content,
    build_story_messages,
    format_story_messages,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_targets_visible_scene_only(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("Describe the visible scene in one concise paragraph", prompt)
        self.assertIn("Include distinctive objects, colors, counts, positions", prompt)
        self.assertIn("Use only visible facts", prompt)
        self.assertIn("Reply only with the description text", prompt)

    def test_build_description_messages_uses_system_and_image_turn(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        messages = build_description_messages(prompt)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], MULTIMODAL_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(
            messages[1]["content"],
            build_image_text_content(prompt),
        )
        self.assertTrue(prompt.endswith(DESCRIPTION_USER_PROMPT_SUFFIX))

    def test_build_story_messages_for_part_1_preserves_description_turn(self):
        messages = build_story_messages(
            description_prompt="Describe the visible scene in one concise paragraph.",
            description_text="A blue house stands beside two green trees and a blue car.",
            previous_parts=[],
        )

        self.assertEqual(messages[0]["content"], MULTIMODAL_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "A blue house stands beside two green trees and a blue car.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], build_image_text_content(PART_1_USER_PROMPT))

    def test_build_story_messages_for_part_2_keeps_description_and_part_1(self):
        messages = build_story_messages(
            description_prompt="Describe the visible scene in one concise paragraph.",
            description_text="A rabbit stands beside a pond.",
            previous_parts=["The rabbit watched the still pond shine softly."],
        )

        self.assertEqual(messages[4]["role"], "assistant")
        self.assertEqual(messages[4]["content"], "The rabbit watched the still pond shine softly.")
        self.assertEqual(messages[5]["role"], "user")
        self.assertEqual(messages[5]["content"], build_image_text_content(PART_2_USER_PROMPT))

    def test_build_story_messages_for_part_3_keeps_full_multimodal_history(self):
        messages = build_story_messages(
            description_prompt="Describe the visible scene in one concise paragraph.",
            description_text="A rabbit stands beside a pond.",
            previous_parts=[
                "The rabbit watched the still pond shine softly.",
                "A small ripple widened once and then grew still again.",
            ],
        )

        self.assertEqual(messages[6]["role"], "assistant")
        self.assertEqual(messages[6]["content"], "A small ripple widened once and then grew still again.")
        self.assertEqual(messages[7]["role"], "user")
        self.assertEqual(messages[7]["content"], build_image_text_content(PART_3_USER_PROMPT))

    def test_format_story_messages_serializes_image_turns(self):
        formatted = format_story_messages(
            [
                {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
                {"role": "user", "content": build_image_text_content(PART_1_USER_PROMPT)},
                {"role": "assistant", "content": "A calm story part."},
            ]
        )

        self.assertIn("SYSTEM:", formatted)
        self.assertIn("[IMAGE]", formatted)
        self.assertIn(PART_1_USER_PROMPT, formatted)
        self.assertIn("ASSISTANT:", formatted)
        self.assertIn("A calm story part.", formatted)

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
