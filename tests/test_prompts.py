import unittest

from story_app.config import DEFAULT_CONFIG
from story_app.prompts import (
    CONTINUITY_BRIEF_SYSTEM_PROMPT,
    DESCRIPTION_SYSTEM_PROMPT,
    DESCRIPTION_USER_PROMPT_SUFFIX,
    IMAGE_PROMPT_SYSTEM_PROMPT,
    PART_2_USER_PROMPT,
    PART_3_USER_PROMPT,
    STORY_SYSTEM_PROMPT,
    build_continuity_brief_messages,
    build_description_messages,
    build_description_prompt,
    build_story_messages,
    build_story_part_1_prompt,
    build_story_part_image_summary_messages,
    format_story_messages,
    validate_description_text,
    validate_story_part_text,
)
from story_app.schemas import ValidationError


class PromptTests(unittest.TestCase):
    def test_build_description_prompt_targets_scene_and_actor_brief(self):
        prompt = build_description_prompt(DEFAULT_CONFIG)
        self.assertIn("about 5 short sentences", prompt)
        self.assertIn("one notable central actor or character", prompt)
        self.assertIn("If no notable character is clearly present, invent one fitting scene-related actor", prompt)
        self.assertIn("descriptive role instead of a proper name", prompt)
        self.assertIn("One sentence must explicitly say who the central actor is", prompt)
        self.assertIn("Keep all other details faithful to visible scene facts", prompt)
        self.assertIn("Treat the depicted content as a real scene", prompt)
        self.assertIn("Never mention the image, picture, drawing, illustration, sketch, painting, child art", prompt)
        self.assertIn("Keep the prose concise", prompt)
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
        self.assertIn("Treat the depicted content as a real scene", DESCRIPTION_SYSTEM_PROMPT)
        self.assertIn("Never mention the image, picture, drawing, illustration, sketch", DESCRIPTION_SYSTEM_PROMPT)

    def test_story_system_prompt_requires_same_actor_and_arc(self):
        self.assertIn("same central actor", STORY_SYSTEM_PROMPT)
        self.assertIn("continuity brief", STORY_SYSTEM_PROMPT)
        self.assertIn("required visual canon", STORY_SYSTEM_PROMPT)
        self.assertIn("naturally repeat the actor's defining appearance", STORY_SYSTEM_PROMPT)
        self.assertIn("Part 2 introduces one visible event, mystery, or noticeable change", STORY_SYSTEM_PROMPT)
        self.assertIn("Part 3 resolves that same event", STORY_SYSTEM_PROMPT)
        self.assertIn("directly related nearby place", STORY_SYSTEM_PROMPT)
        self.assertIn("keep the prose concise", STORY_SYSTEM_PROMPT)
        self.assertIn("never refer to an image, picture, drawing, illustration, sketch, painting, child art", STORY_SYSTEM_PROMPT)

    def test_build_continuity_brief_messages_targets_recurring_visual_canon(self):
        messages = build_continuity_brief_messages("A blue house with a red roof stands behind a child in a sunny yard.")

        self.assertEqual(messages[0]["content"], CONTINUITY_BRIEF_SYSTEM_PROMPT)
        self.assertIn("Scene and actor description:", messages[1]["content"])
        self.assertIn("Reply in exactly this compact format", messages[1]["content"])
        self.assertIn("actor: <full actor phrase>; anchors: <anchor phrase 1>, <anchor phrase 2>, <anchor phrase 3>", messages[1]["content"])
        self.assertIn("The actor phrase must include the actor's defining visual traits", messages[1]["content"])
        self.assertIn("Avoid vague anchors like fish, house, scales, plants", messages[1]["content"])
        self.assertIn("Use only real scene terms, never image-medium terms", messages[1]["content"])
        self.assertIn("Reply only with the continuity brief text", messages[1]["content"])

    def test_build_story_part_1_prompt_injects_scene_and_actor_description(self):
        prompt = build_story_part_1_prompt(
            "A blue house stands beside two green trees and a blue car.",
            "actor: cheerful yard child in a red coat; anchors: blue house with red roof, small blue car, two green trees",
        )
        self.assertIn("part 1 of a gentle three-part bedtime story", prompt)
        self.assertIn("Scene and actor description:", prompt)
        self.assertIn("Continuity brief:", prompt)
        self.assertIn("A blue house stands beside two green trees and a blue car.", prompt)
        self.assertIn("Open with the same central actor", prompt)
        self.assertIn("Naturally repeat the actor phrase and any visible canon anchors", prompt)
        self.assertIn("do not start the main event yet", prompt)
        self.assertIn("make each sentence short", prompt)
        self.assertIn("never mention drawings, pictures, illustrations, sketches, paintings, or child art", prompt)
        self.assertIn("Use exactly 3 short sentences", prompt)

    def test_part_2_prompt_targets_actor_affecting_event(self):
        self.assertIn("same central actor", PART_2_USER_PROMPT)
        self.assertIn("visible event, mystery, or noticeable change", PART_2_USER_PROMPT)
        self.assertIn("Naturally repeat the actor's defining appearance", PART_2_USER_PROMPT)
        self.assertIn("Do not resolve the event yet", PART_2_USER_PROMPT)
        self.assertIn("Use exactly 3 short sentences", PART_2_USER_PROMPT)

    def test_part_3_prompt_targets_resolution(self):
        self.assertIn("same central actor", PART_3_USER_PROMPT)
        self.assertIn("resolve the same event or change from part 2", PART_3_USER_PROMPT)
        self.assertIn("calm final state", PART_3_USER_PROMPT)
        self.assertIn("Naturally repeat the actor's defining appearance", PART_3_USER_PROMPT)
        self.assertIn("Use exactly 3 short sentences", PART_3_USER_PROMPT)

    def test_build_story_messages_for_part_1_starts_new_text_only_conversation(self):
        messages = build_story_messages(
            description_text="A blue house stands beside two green trees and a blue car.",
            continuity_brief="actor: cheerful yard child in a red coat; anchors: blue house with red roof, small blue car, two green trees",
            previous_parts=[],
        )

        self.assertEqual(messages[0]["content"], STORY_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(messages[1]["content"], str)
        self.assertIn("Scene and actor description:", messages[1]["content"])
        self.assertIn("Continuity brief:", messages[1]["content"])
        self.assertIn("blue house stands beside two green trees", messages[1]["content"])

    def test_build_story_messages_for_part_2_keeps_text_history_only(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            continuity_brief="actor: small gray rabbit with a lantern; anchors: still pond, glowing lantern",
            previous_parts=["The rabbit watched the still pond shine softly."],
        )

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "The rabbit watched the still pond shine softly.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("Continuity brief:", messages[3]["content"])
        self.assertIn(PART_2_USER_PROMPT, messages[3]["content"])

    def test_build_story_messages_for_part_3_keeps_full_text_history(self):
        messages = build_story_messages(
            description_text="A rabbit stands beside a pond.",
            continuity_brief="actor: small gray rabbit with a lantern; anchors: still pond, glowing lantern",
            previous_parts=[
                "The rabbit watched the still pond shine softly.",
                "A small ripple widened once and then grew still again.",
            ],
        )

        self.assertEqual(messages[4]["role"], "assistant")
        self.assertEqual(messages[4]["content"], "A small ripple widened once and then grew still again.")
        self.assertEqual(messages[5]["role"], "user")
        self.assertIn("Continuity brief:", messages[5]["content"])
        self.assertIn(PART_3_USER_PROMPT, messages[5]["content"])

    def test_build_story_part_image_summary_messages_mentions_previous_page(self):
        messages = build_story_part_image_summary_messages(
            DEFAULT_CONFIG,
            description_text="A gray rabbit stands beside a pond with a small lantern.",
            continuity_brief="The same gray rabbit, small lantern, and pond stay visually consistent when visible.",
            part_text="The rabbit sees a bright ripple in the pond and leans closer.",
            part_index=2,
            max_image_prompt_tokens=74,
            previous_image_prompt="gray rabbit beside the pond and lantern, children's picture-book illustration",
        )

        self.assertEqual(messages[0]["content"], IMAGE_PROMPT_SYSTEM_PROMPT)
        self.assertIn("Scene and actor description:", messages[1]["content"])
        self.assertIn("Continuity brief:", messages[1]["content"])
        self.assertIn("Previous page final image prompt:", messages[1]["content"])
        self.assertIn(
            "gray rabbit beside the pond and lantern, children's picture-book illustration",
            messages[1]["content"],
        )
        self.assertIn("Preserve continuity with that page", messages[1]["content"])
        self.assertIn("within 74 image-model tokens", messages[1]["content"])
        self.assertIn("Mention the same actor first or early", messages[1]["content"])
        self.assertIn("reuse the actor phrase from the continuity brief", messages[1]["content"])
        self.assertIn("at most two recurring anchor details", messages[1]["content"])
        self.assertIn("keep its same described color and identity", messages[1]["content"])
        self.assertIn("Do not describe the scene as a drawing, sketch, or picture", IMAGE_PROMPT_SYSTEM_PROMPT)

    def test_format_story_messages_serializes_text_story_turns(self):
        formatted = format_story_messages(
            [
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_story_part_1_prompt(
                        "A calm blue house beside two trees.",
                        "actor: calm blue house; anchors: two green trees, bright moon",
                    ),
                },
                {"role": "assistant", "content": "A calm story part."},
            ]
        )

        self.assertIn("SYSTEM:", formatted)
        self.assertIn("Scene and actor description:", formatted)
        self.assertIn("ASSISTANT:", formatted)
        self.assertIn("A calm story part.", formatted)

    def test_validate_description_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_description_text("   ", DEFAULT_CONFIG)

    def test_validate_story_part_text_rejects_empty_output(self):
        with self.assertRaises(ValidationError):
            validate_story_part_text("   ", DEFAULT_CONFIG)

    def test_validate_description_text_no_longer_enforces_minimum_word_count(self):
        self.assertEqual(validate_description_text("fox", DEFAULT_CONFIG), "fox")

    def test_validate_story_part_text_no_longer_enforces_minimum_word_count(self):
        self.assertEqual(validate_story_part_text("hush", DEFAULT_CONFIG), "hush")


if __name__ == "__main__":
    unittest.main()
