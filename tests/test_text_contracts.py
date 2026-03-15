import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import parse_description_response, parse_story_response
from story_app.schemas import SchemaError


class TextContractTests(unittest.TestCase):
    def test_parse_description_response_reads_labeled_lines(self):
        raw_text = "\n".join(
            [
                "SUMMARY: A calm rabbit stands beside a moonlit pond.",
                "CHARACTERS: rabbit, moon, pond",
                "SETTING: a moonlit pond in a quiet meadow",
                "STYLE: colored pencil storybook art",
                "COLORS: navy, silver, cream",
                "SAFETY: gentle mood, bedtime calm, no danger",
            ]
        )

        description = parse_description_response(raw_text)
        self.assertEqual(description.characters, ["rabbit", "moon", "pond"])
        self.assertEqual(description.color_palette, ["navy", "silver", "cream"])

    def test_parse_description_response_rejects_missing_labels(self):
        raw_text = "\n".join(
            [
                "SUMMARY: A calm rabbit stands beside a moonlit pond.",
                "CHARACTERS: rabbit, moon, pond",
                "SETTING: a moonlit pond in a quiet meadow",
                "STYLE: colored pencil storybook art",
                "COLORS: navy, silver, cream",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_description_response(raw_text)

    def test_parse_description_response_rejects_malformed_character_line(self):
        raw_text = "\n".join(
            [
                "SUMMARY: A calm rabbit stands beside a moonlit pond.",
                "CHARACTERS: rabbit; moon; pond",
                "SETTING: a moonlit pond in a quiet meadow",
                "STYLE: colored pencil storybook art",
                "COLORS: navy, silver, cream",
                "SAFETY: gentle mood, bedtime calm, no danger",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_description_response(raw_text)

    def test_parse_story_response_reads_three_parts(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PART1_ENTRANCE: The rabbit wandered to the pond and listened to the night settle softly around the water.",
                "PART2_BUILDUP: The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime.",
                "PART3_ENDING: The rabbit curled beside the pond, watched the silver light fade into sleep, and rested in a calm dreamy hush.",
            ]
        )

        story = parse_story_response(raw_text, DEFAULT_CONFIG)
        self.assertEqual([part.scene_goal for part in story.parts], ["entrance", "buildup", "ending"])
        self.assertEqual(story.title, "Rabbit and the Quiet Pond")

    def test_parse_story_response_rejects_empty_or_duplicate_label_lines(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PART1_ENTRANCE: The rabbit wandered to the pond and listened to the night settle softly around the water.",
                "PART1_ENTRANCE: The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime.",
                "PART3_ENDING: ",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_story_response(raw_text, DEFAULT_CONFIG)

    def test_parse_story_response_rejects_extra_story_part(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PART1_ENTRANCE: The rabbit wandered to the pond and listened to the night settle softly around the water.",
                "PART2_BUILDUP: The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime.",
                "PART3_ENDING: The rabbit curled beside the pond, watched the silver light fade into sleep, and rested in a calm dreamy hush.",
                "PART4_CODA: Extra ending text.",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_story_response(raw_text, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
