import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    build_description_prompt,
    build_story_part_prompt,
    extract_visual_anchor_words,
    find_anchor_overlap,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_uses_clean_caption_prefix(self):
        self.assertIn("exact visible detail", build_description_prompt(DEFAULT_CONFIG))
        self.assertIn("colors", build_description_prompt(DEFAULT_CONFIG))

    def test_build_story_part_1_prompt_includes_description_and_beginning_instruction(self):
        prompt = build_story_part_prompt(
            description_text="A rabbit stands by a moonlit pond.",
            step_name="part_1",
            previous_parts=[],
        )
        self.assertIn("Drawing description:", prompt)
        self.assertIn("A rabbit stands by a moonlit pond.", prompt)
        self.assertIn("beginning of a three-part bedtime story", prompt)
        self.assertIn("calm starting situation", prompt)
        self.assertIn("invent only one gentle main character", prompt)
        self.assertNotIn("Accepted story so far:", prompt)

    def test_build_story_part_2_prompt_includes_part_1_exactly(self):
        prompt = build_story_part_prompt(
            description_text="A rabbit stands by a moonlit pond.",
            step_name="part_2",
            previous_parts=["The rabbit padded softly toward the quiet water under the moon."],
        )
        self.assertIn("Accepted story so far:", prompt)
        self.assertIn("1. The rabbit padded softly toward the quiet water under the moon.", prompt)
        self.assertIn("middle of the same story", prompt)
        self.assertIn("specific gentle event", prompt)
        self.assertIn("directly involve something clearly visible in the drawing", prompt)

    def test_build_story_part_3_prompt_includes_part_1_and_part_2(self):
        prompt = build_story_part_prompt(
            description_text="A rabbit stands by a moonlit pond.",
            step_name="part_3",
            previous_parts=[
                "The rabbit padded softly toward the quiet water under the moon.",
                "He watched silver ripples drift across the pond and listened to the reeds.",
            ],
        )
        self.assertIn("1. The rabbit padded softly toward the quiet water under the moon.", prompt)
        self.assertIn("2. He watched silver ripples drift across the pond and listened to the reeds.", prompt)
        self.assertIn("ending of the same story", prompt)
        self.assertIn("resolve the gentle event", prompt.lower())
        self.assertIn("Do not start a new event.", prompt)
        self.assertIn("same setting unless the earlier parts already changed it", prompt)
        self.assertIn("End with a complete sentence.", prompt)
        self.assertIn("Stop immediately after the paragraph.", prompt)
        self.assertIn("Use at least three concrete details", prompt)

    def test_extract_visual_anchor_words_prefers_visual_nouns(self):
        anchors = extract_visual_anchor_words(
            "A child's drawing of a blue house with a red roof and two brown trees with green tops and two blue cars."
        )
        self.assertIn("house", anchors)
        self.assertIn("roof", anchors)
        self.assertIn("trees", anchors)
        self.assertIn("cars", anchors)

    def test_find_anchor_overlap_reports_grounded_words(self):
        overlap = find_anchor_overlap(
            "A blue house with two trees and two cars under a yellow sun.",
            "A child walked past the house and trees while the sun warmed the cars.",
        )
        self.assertIn("house", overlap)
        self.assertIn("trees", overlap)
        self.assertIn("cars", overlap)

    def test_validate_description_text_rejects_structured_markers(self):
        with self.assertRaises(ValidationError):
            validate_description_text("```json {\"caption\": \"bad\"}```", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_meta_wrapper(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text(
                "Here is the first part of the story: A rabbit walked to the pond in the moonlight and listened quietly.",
                DEFAULT_CONFIG,
            )

    def test_validate_story_part_text_rejects_incomplete_ending(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text(
                "A rabbit walked beside the pond under the moon and listened to the reeds while the quiet forest wrapped around him and he knew he belonged in this",
                DEFAULT_CONFIG,
            )


if __name__ == "__main__":
    unittest.main()
