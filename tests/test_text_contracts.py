import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import parse_description_response, parse_story_response
from story_app.schemas import SchemaError


class TextContractTests(unittest.TestCase):
    def test_parse_description_response_keeps_single_description_string(self):
        raw_text = (
            "A calm rabbit stands beside a moonlit pond while silver stars reflect in the water "
            "and soft reeds sway in the quiet night."
        )

        description = parse_description_response(raw_text, DEFAULT_CONFIG)
        self.assertIn("moonlit pond", description.text)

    def test_parse_description_response_rejects_too_short_output(self):
        with self.assertRaises(SchemaError):
            parse_description_response("House. Trees. Sun. Car.", DEFAULT_CONFIG)

    def test_parse_description_response_rejects_corrupted_special_tokens(self):
        with self.assertRaises(SchemaError):
            parse_description_response(
                "madeupword0002 DepthInterface<loc_471>quet negotiations Shir prayers diagnostic 670Chel",
                DEFAULT_CONFIG,
            )

    def test_parse_story_response_reads_title_and_three_parts(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PARTS:",
                "The rabbit wandered to the pond and listened to the night settle softly around the water while a calm adventure began there.",
                DEFAULT_CONFIG.story_separator_token,
                "The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime all around.",
                DEFAULT_CONFIG.story_separator_token,
                "The rabbit curled beside the pond, watched the silver light fade into sleep, and rested in a calm dreamy hush until morning.",
            ]
        )

        story = parse_story_response(raw_text, DEFAULT_CONFIG)
        self.assertEqual([part.scene_goal for part in story.parts], ["entrance", "buildup", "ending"])
        self.assertEqual(story.title, "Rabbit and the Quiet Pond")

    def test_parse_story_response_accepts_title_and_parts_on_same_line(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond PARTS:",
                "The rabbit wandered to the pond and listened to the night settle softly around the water while a calm adventure began there.",
                DEFAULT_CONFIG.story_separator_token,
                "The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime all around.",
                DEFAULT_CONFIG.story_separator_token,
                "The rabbit curled beside the pond, watched the silver light fade into sleep, and rested in a calm dreamy hush until morning.",
            ]
        )

        story = parse_story_response(raw_text, DEFAULT_CONFIG)
        self.assertEqual(story.title, "Rabbit and the Quiet Pond")

    def test_parse_story_response_rejects_missing_parts_header(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "The rabbit wandered to the pond and listened to the night settle softly around the water while a calm adventure began there.",
                DEFAULT_CONFIG.story_separator_token,
                "The moon reflected in the pond while the rabbit noticed gentle ripples, friendly reeds, and the quiet hush of bedtime all around.",
                DEFAULT_CONFIG.story_separator_token,
                "The rabbit curled beside the pond, watched the silver light fade into sleep, and rested in a calm dreamy hush until morning.",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_story_response(raw_text, DEFAULT_CONFIG)

    def test_parse_story_response_rejects_wrong_separator_count(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PARTS:",
                "Part one text with enough words to satisfy the parser and keep the story calm and gentle.",
                "Part two text without the required separator token in between the story sections.",
                DEFAULT_CONFIG.story_separator_token,
                "Part three text with enough words to satisfy the parser and keep the story calm and gentle.",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_story_response(raw_text, DEFAULT_CONFIG)

    def test_parse_story_response_rejects_short_part(self):
        raw_text = "\n".join(
            [
                "TITLE: Rabbit and the Quiet Pond",
                "PARTS:",
                "Too short for this parser.",
                DEFAULT_CONFIG.story_separator_token,
                "This second part has enough words to satisfy the minimum length requirement for the deterministic parser to accept it.",
                DEFAULT_CONFIG.story_separator_token,
                "This third part also has enough words to satisfy the minimum length requirement for the deterministic parser to accept it.",
            ]
        )

        with self.assertRaises(SchemaError):
            parse_story_response(raw_text, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
