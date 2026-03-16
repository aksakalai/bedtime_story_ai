import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    ANCHOR_FIELDS,
    ANCHOR_UPDATE_SYSTEM_PROMPT,
    DESCRIPTION_SYSTEM_PROMPT,
    DESCRIPTION_USER_PROMPT_SUFFIX,
    IMAGE_PROMPT_SYSTEM_PROMPT,
    STORY_SYSTEM_PROMPT,
    build_description_messages,
    build_description_prompt,
    build_next_part_anchor_messages,
    build_story_messages,
    build_story_part_1_prompt,
    build_story_part_image_summary_messages,
    format_story_messages,
    validate_anchor_text,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_targets_exact_anchor_sheet(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("fill one compact anchor sheet for page 1", prompt)
        self.assertIn("Treat the depicted content as a real scene", prompt)
        self.assertIn("If no clear actor exists, create one fitting general actor", prompt)
        for field in ANCHOR_FIELDS:
            self.assertIn(f"{field}:", prompt)
        self.assertTrue(prompt.endswith("Reply only with the anchor sheet text."))

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
        self.assertIn("never as a drawing, picture, sketch, illustration", DESCRIPTION_SYSTEM_PROMPT)

    def test_build_next_part_anchor_messages_requests_full_next_page_sheet(self):
        messages = build_next_part_anchor_messages(
            current_anchor_text="actor: little blue fish\nscene: underwater garden",
            latest_part_text="The fish noticed a pale glowing creature ahead.",
            next_part_index=2,
        )

        self.assertEqual(messages[0]["content"], ANCHOR_UPDATE_SYSTEM_PROMPT)
        self.assertIn("Current page anchor:", messages[1]["content"])
        self.assertIn("Latest story part:", messages[1]["content"])
        self.assertIn("Write the full anchor sheet for part 2", messages[1]["content"])
        self.assertIn("Carry forward the same actor identity and the same colors", messages[1]["content"])
        for field in ANCHOR_FIELDS:
            self.assertIn(f"{field}:", messages[1]["content"])

    def test_story_system_prompt_requires_anchor_driven_arc(self):
        self.assertIn("current page anchor is the source of truth", STORY_SYSTEM_PROMPT)
        self.assertIn("Earlier anchors and story parts are only for continuity", STORY_SYSTEM_PROMPT)
        self.assertIn("Part 1 sets up the actor and scene", STORY_SYSTEM_PROMPT)
        self.assertIn("Part 2 introduces one visible event", STORY_SYSTEM_PROMPT)
        self.assertIn("Part 3 resolves that event", STORY_SYSTEM_PROMPT)
        self.assertIn("never mention an image, picture, drawing, illustration", STORY_SYSTEM_PROMPT)

    def test_build_story_part_1_prompt_uses_current_page_anchor(self):
        prompt = build_story_part_1_prompt(
            "actor: curious little blue fish\nactor_colors: blue and purple\nscene: underwater garden"
        )
        self.assertIn("Current page anchor for part 1:", prompt)
        self.assertIn("Introduce the actor, scene, and named objects", prompt)
        self.assertIn("Do not start the main event yet", prompt)
        self.assertIn("Use exactly 3 short sentences", prompt)

    def test_build_story_messages_for_part_2_includes_anchor_and_story_history(self):
        messages = build_story_messages(
            part_index=2,
            current_anchor_text="actor: curious little blue fish\npage_event: glowing creature appears",
            previous_anchors=["actor: curious little blue fish\npage_event: calm exploration"],
            previous_parts=["The little blue fish swam calmly through the underwater garden."],
        )

        self.assertEqual(messages[0]["content"], STORY_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Earlier page 1 anchor:", messages[1]["content"])
        self.assertIn("Earlier story part 1:", messages[1]["content"])
        self.assertIn("Current page anchor for part 2:", messages[1]["content"])
        self.assertIn("Write only part 2 of the bedtime story", messages[1]["content"])

    def test_build_story_messages_for_part_3_includes_full_history(self):
        messages = build_story_messages(
            part_index=3,
            current_anchor_text="actor: curious little blue fish\npage_event: calm return home",
            previous_anchors=[
                "actor: curious little blue fish\npage_event: calm exploration",
                "actor: curious little blue fish\npage_event: glowing creature appears",
            ],
            previous_parts=[
                "Part 1 text.",
                "Part 2 text.",
            ],
        )

        self.assertIn("Earlier page 1 anchor:", messages[1]["content"])
        self.assertIn("Earlier page 2 anchor:", messages[1]["content"])
        self.assertIn("Earlier story part 1:", messages[1]["content"])
        self.assertIn("Earlier story part 2:", messages[1]["content"])
        self.assertIn("Current page anchor for part 3:", messages[1]["content"])
        self.assertIn("Resolve the same event with a calm ending", messages[1]["content"])

    def test_build_story_part_image_summary_messages_uses_only_current_anchor(self):
        messages = build_story_part_image_summary_messages(
            DEFAULT_CONFIG,
            page_anchor_text="actor: curious little blue fish\nactor_colors: blue and purple\npage_event: glowing creature appears",
            part_index=2,
            max_image_prompt_tokens=74,
        )

        self.assertEqual(messages[0]["content"], IMAGE_PROMPT_SYSTEM_PROMPT)
        self.assertIn("Current page anchor:", messages[1]["content"])
        self.assertIn("within 74 image-model tokens", messages[1]["content"])
        self.assertIn("Use only the current page anchor, not any previous story text", messages[1]["content"])
        self.assertNotIn("Previous page final image prompt:", messages[1]["content"])

    def test_format_story_messages_serializes_anchor_prompts(self):
        formatted = format_story_messages(
            [
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {"role": "user", "content": build_story_part_1_prompt("actor: little bunny\nscene: meadow")},
                {"role": "assistant", "content": "A little bunny looked across the meadow."},
            ]
        )

        self.assertIn("SYSTEM:", formatted)
        self.assertIn("Current page anchor for part 1:", formatted)
        self.assertIn("ASSISTANT:", formatted)

    def test_validate_description_text_preserves_anchor_lines(self):
        validated = validate_description_text("actor: fish\n\nscene: pond  ", DEFAULT_CONFIG)
        self.assertEqual(validated, "actor: fish\nscene: pond")

    def test_validate_anchor_text_preserves_anchor_lines(self):
        validated = validate_anchor_text("actor: fish\n\npage_event: splash  ", DEFAULT_CONFIG)
        self.assertEqual(validated, "actor: fish\npage_event: splash")

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_no_longer_enforces_minimum_word_count(self):
        self.assertEqual(validate_story_part_text("hush", DEFAULT_CONFIG), "hush")


if __name__ == "__main__":
    unittest.main()
