from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .schemas import StoryPackage, TimelineManifest, TimelineSegment


def build_timeline(story: StoryPackage, audio_path: str) -> TimelineManifest:
    segments: list[TimelineSegment] = []
    elapsed = 0.0
    for index, part in enumerate(story.parts, start=1):
        duration = float(part.duration_sec)
        segment = TimelineSegment(
            index=index,
            start_sec=round(elapsed, 3),
            end_sec=round(elapsed + duration, 3),
            story_text=part.story_text,
            image_path=part.image_path,
        )
        segments.append(segment)
        elapsed += duration
    return TimelineManifest(
        audio_path=audio_path,
        total_duration_sec=round(elapsed, 3),
        segments=segments,
    )


def _to_data_url(path: str, mime_type: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_playback_panel_html(story: StoryPackage, timeline: TimelineManifest) -> str:
    images = [_to_data_url(segment.image_path, "image/png") for segment in timeline.segments]
    audio = _to_data_url(timeline.audio_path, "audio/wav")
    text_items = [segment.story_text for segment in timeline.segments]
    timeline_payload = json.dumps(
        [
            {
                "index": segment.index,
                "start": segment.start_sec,
                "end": segment.end_sec,
                "text": segment.story_text,
                "image": image_url,
            }
            for segment, image_url in zip(timeline.segments, images)
        ]
    )
    encoded_timeline = html.escape(timeline_payload, quote=True)
    title = html.escape(story.title)
    initial_text = html.escape(text_items[0] if text_items else "")
    initial_image = images[0] if images else ""
    return f"""
<div class="storybook-player" data-storybook-player="true" data-timeline="{encoded_timeline}">
  <div class="storybook-stage" data-storybook-stage="true" style="background-image:url('{initial_image}')">
    <div class="storybook-overlay"></div>
    <div class="storybook-copy">
      <div class="storybook-kicker">Now reading</div>
      <h2>{title}</h2>
      <p data-storybook-text="true">{initial_text}</p>
    </div>
  </div>
  <audio data-storybook-audio="true" controls preload="metadata" src="{audio}"></audio>
</div>
"""
