import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    DESCRIPTION_SYSTEM_PROMPT,
    DESCRIPTION_USER_PROMPT_SUFFIX,
    PART_2_USER_PROMPT,
    PART_3_USER_PROMPT,
    build_description_messages,
    build_description_prompt,
    build_story_part_1_prompt,
    build_story_messages,
    format_story_messages,
    STORY_SYSTEM_PROMPT,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_targets_visible_scene_only(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("Describe the depicted scene itself in one concise paragraph", prompt)
        self.assertIn("Include distinctive objects, colors, counts, positions", prompt)
        self.assertIn("Use only visible scene facts", prompt)
        self.assertIn("Do not mention the image, drawing, painting", prompt)
        self.assertIn("Reply only with the description text", prompt)

    def test_build_description_messages_uses_system_and_image_turn(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        messages = build_description_messages(prompt)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DESCRIPTION_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertEqual(messages[1]["content"][1]["type"], "text")
        self.assertEqual(messages[1]["content"][1]["text"], prompt)
        self.assertTrue(prompt.endswith(DESCRIPTION_USER_PROMPT_SUFFIX))

    def test_build_story_part_1_prompt_injects_description_text(self):
        prompt = build_story_part_1_prompt("A blue house stands beside two green trees and a blue car.")
        self.assertIn("part 1 of a gentle three-part bedtime story", prompt)
        self.assertIn("Scene description:", prompt)
        self.assertIn("A blue house stands beside two green trees and a blue car.", prompt)
        self.assertIn("grounded in those visible details", prompt)
        self.assertIn("write about 50 words", prompt)

    def test_build_story_messages_for_part_1_starts_new_text_only_conversation(self):
        messages = build_story_messages(
            description_text="A blue house stands beside two green trees and a blue car.",
            previous_parts=[],
        )

        self.assertEqual(messages[0]["content"], STORY_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(messages[1]["content"], str)
        self.assertIn("Scene description:", messages[1]["content"])
        self.assertIn("blue house stands beside two green trees", messages[1]["content"])

    def test_build_story_messages_for_part_2_keeps_text_history_only(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            previous_parts=["The rabbit watched the still pond shine softly."],
        )

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "The rabbit watched the still pond shine softly.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], PART_2_USER_PROMPT)

    def test_build_story_messages_for_part_3_keeps_full_text_history(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            previous_parts=[
                "The rabbit watched the still pond shine softly.",
                "A small ripple widened once and then grew still again.",
            ],
        )

        self.assertEqual(messages[4]["role"], "assistant")
        self.assertEqual(messages[4]["content"], "A small ripple widened once and then grew still again.")
        self.assertEqual(messages[5]["role"], "user")
        self.assertEqual(messages[5]["content"], PART_3_USER_PROMPT)

    def test_format_story_messages_serializes_text_story_turns(self):
        formatted = format_story_messages(
            [
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {"role": "user", "content": build_story_part_1_prompt("A calm blue house beside two trees.")},
                {"role": "assistant", "content": "A calm story part."},
            ]
        )

        self.assertIn("SYSTEM:", formatted)
        self.assertIn("Scene description:", formatted)
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
