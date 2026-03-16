from __future__ import annotations

import re
from typing import Any

from .config import GenerationConfig
from .schemas import ValidationError

ANCHOR_FIELDS = [
    "actor",
    "actor_colors",
    "actor_traits",
    "scene",
    "scene_colors",
    "object_1",
    "object_1_colors",
    "object_2",
    "object_2_colors",
    "object_3",
    "object_3_colors",
    "secondary_actor",
    "secondary_actor_colors",
    "page_event",
    "mood",
]

DESCRIPTION_SYSTEM_PROMPT = (
    "You observe one uploaded scene and convert it into one compact page anchor sheet for a children's story. Treat "
    "the depicted content as a real scene, never as a drawing, picture, sketch, illustration, painting, or child "
    "art. Extract one central actor and explicit colors for the actor, the scene, and important objects whenever "
    "visible. If no clear actor exists, create one fitting simple actor for the scene, such as a child, bunny, "
    "fish, duckling, fox, or similar gentle character. Reply only with the exact anchor sheet."
)

DESCRIPTION_USER_PROMPT_SUFFIX = "Reply only with the anchor sheet text."

ANCHOR_UPDATE_SYSTEM_PROMPT = (
    "You update one compact page anchor sheet for the next page of the same bedtime story. Treat all anchors as real "
    "scene facts, never as image-medium descriptions. Keep the same actor identity unless the latest story part "
    "clearly changes it. Keep actor colors, object colors, and scene colors consistent unless the latest story part "
    "clearly changes them. Output one full next-page anchor sheet with the exact same keys, not a diff."
)

STORY_SYSTEM_PROMPT = (
    "You write gentle bedtime-story prose for three consecutive children's picture-book pages. The current page "
    "anchor is the source of truth for what should be visible on this page. Earlier anchors and story parts are only "
    "for continuity. Keep the same central actor across the story unless the anchors explicitly change that. Reuse "
    "the actor identity, actor colors, scene, and named objects naturally in the prose so the illustrations stay "
    "consistent. Part 1 sets up the actor and scene. Part 2 introduces one visible event, mystery, or clear change. "
    "Part 3 resolves that event with a calm ending. Treat the story world as real and never mention an image, "
    "picture, drawing, illustration, sketch, painting, child art, paper, artist, or style. Reply only with the "
    "requested story text."
)

IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You turn one page anchor into one very short image prompt sentence for a CLIP-limited image model. The anchor "
    "is the source of truth for what should be visible. Mention the actor early, preserve the actor colors and object "
    "colors exactly when they are present, and describe only the current page snapshot. Treat the content as a real "
    "scene. Never describe it as a drawing, picture, sketch, or illustration inside the prompt sentence. Reply only "
    "with one short sentence."
)


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValidationError("Model output was empty.")
    return text


def normalize_multiline_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValidationError("Model output was empty.")

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        raise ValidationError("Model output was empty.")
    return "\n".join(lines)


def build_description_prompt(config: GenerationConfig) -> str:
    prompt = config.description_prompt_prefix.rstrip()
    if not prompt.endswith(DESCRIPTION_USER_PROMPT_SUFFIX.strip()):
        prompt = f"{prompt} {DESCRIPTION_USER_PROMPT_SUFFIX}"
    return prompt


def build_image_text_content(prompt_text: str) -> list[dict[str, str]]:
    return [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]


def build_description_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_image_text_content(prompt_text)},
    ]


def build_next_part_anchor_messages(
    *,
    current_anchor_text: str,
    latest_part_text: str,
    next_part_index: int,
) -> list[dict[str, Any]]:
    if next_part_index == 2:
        update_goal = (
            "Write the full anchor sheet for part 2. The next page should show one visible event, mystery, or clear "
            "change affecting the same actor."
        )
    else:
        update_goal = (
            "Write the full anchor sheet for part 3. The next page should show the calm resolved state of the same "
            "event."
        )

    prompt_text = (
        f"Current page anchor:\n{current_anchor_text}\n\n"
        f"Latest story part:\n{latest_part_text}\n\n"
        f"{update_goal}\n"
        "Carry forward the same actor identity and the same colors unless the latest story part clearly changes them. "
        "If the scene shifts, move only to a directly related nearby place. If a new important object or secondary "
        "actor appears, include it. Output exactly these keys in this exact order, using `none` when needed:\n"
        + "\n".join(f"{field}:" for field in ANCHOR_FIELDS)
        + "\nReply only with the full next-page anchor sheet."
    )
    return [
        {"role": "system", "content": ANCHOR_UPDATE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def _format_anchor_history(previous_anchors: list[str]) -> str:
    if not previous_anchors:
        return ""
    sections = [
        f"Earlier page {index} anchor:\n{anchor_text}"
        for index, anchor_text in enumerate(previous_anchors, start=1)
    ]
    return "\n\n".join(sections) + "\n\n"


def _format_story_history(previous_parts: list[str]) -> str:
    if not previous_parts:
        return ""
    sections = [
        f"Earlier story part {index}:\n{part_text}"
        for index, part_text in enumerate(previous_parts, start=1)
    ]
    return "\n\n".join(sections) + "\n\n"


def build_story_part_1_prompt(current_anchor_text: str) -> str:
    return (
        "Current page anchor for part 1:\n"
        f"{current_anchor_text}\n\n"
        "Write only part 1 of the bedtime story. Introduce the actor, scene, and named objects from this anchor. "
        "Keep the mood calm and warm. Do not start the main event yet. Use exactly 3 short sentences. Reply only "
        "with the story text."
    )


def build_story_messages(
    *,
    part_index: int,
    current_anchor_text: str,
    previous_anchors: list[str],
    previous_parts: list[str],
) -> list[dict[str, Any]]:
    if part_index == 1:
        user_prompt = build_story_part_1_prompt(current_anchor_text)
    else:
        history_section = _format_anchor_history(previous_anchors) + _format_story_history(previous_parts)
        if part_index == 2:
            part_goal = (
                "Write only part 2 of the bedtime story. Continue directly from part 1. Use the current page anchor "
                "as the source of truth. Introduce one visible event, mystery, or clear change affecting the actor. "
                "Use exactly 3 short sentences."
            )
        else:
            part_goal = (
                "Write only part 3 of the bedtime story. Continue directly from the earlier parts. Use the current "
                "page anchor as the source of truth. Resolve the same event with a calm ending. Use exactly 3 short "
                "sentences."
            )
        user_prompt = (
            f"{history_section}"
            f"Current page anchor for part {part_index}:\n{current_anchor_text}\n\n"
            f"{part_goal} Reply only with the story text."
        )

    return [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_story_part_image_summary_messages(
    config: GenerationConfig,
    *,
    page_anchor_text: str,
    part_index: int,
    max_image_prompt_tokens: int,
) -> list[dict[str, Any]]:
    if part_index == 1:
        part_role = "This is page 1, so show the opening setup clearly."
    elif part_index == 2:
        part_role = "This is page 2, so show the event or change clearly."
    else:
        part_role = "This is page 3, so show the resolved ending clearly."

    prompt_text = (
        f"Current page anchor:\n{page_anchor_text}\n\n"
        f"{part_role}\n"
        f"Keep the final sentence within {max_image_prompt_tokens} image-model tokens. Mention the actor early. Keep "
        "the actor colors, scene colors, and object colors exactly when they are present. Use only the current page "
        "anchor, not any previous story text. Keep it extremely concise. Reply only with the final sentence."
    )
    return [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def finalize_image_prompt(config: GenerationConfig, scene_prompt_text: str) -> str:
    scene_prompt = normalize_text(scene_prompt_text)
    if config.image_prompt_style_suffix.strip():
        scene_prompt = scene_prompt.rstrip(".,;:")
        return f"{scene_prompt}, {config.image_prompt_style_suffix}"
    return scene_prompt


def _format_message_content(content: Any) -> str:
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                lines.append("[IMAGE]")
            elif item_type == "text":
                lines.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in lines if part).strip()
    return str(content).strip()


def format_story_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = _format_message_content(message.get("content", ""))
        lines.append(f"{role}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def validate_description_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_multiline_text(raw_text)


def validate_anchor_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_multiline_text(raw_text)


def validate_story_part_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_text(raw_text)


def validate_image_prompt_text(raw_text: str, config: GenerationConfig) -> str:
    return normalize_text(raw_text)
