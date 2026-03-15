import unittest

from story_app.prompts import build_description_messages, build_story_messages
from story_app.providers import count_image_placeholders


class ProviderHelperTests(unittest.TestCase):
    def test_count_image_placeholders_matches_description_turn(self):
        messages = build_description_messages("Describe the visible scene.")
        self.assertEqual(count_image_placeholders(messages), 1)

    def test_count_image_placeholders_matches_text_only_story_turn_history(self):
        messages = build_story_messages(
            description_text="A blue house stands beside two green trees and a blue car.",
            previous_parts=[
                "Part 1 text.",
                "Part 2 text.",
            ],
        )
        self.assertEqual(count_image_placeholders(messages), 0)


if __name__ == "__main__":
    unittest.main()
