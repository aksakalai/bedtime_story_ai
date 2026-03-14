from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from .pipeline import KidStoryPipeline
from .schemas import StoryPackage

APP_BUILD = "video-v3-20260315"

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --paper: #fffaf0;
  --paper-soft: rgba(255, 250, 240, 0.92);
  --ink: #1f2933;
  --muted: #556271;
  --caramel: #8b5e3c;
  --gold: #f6cd7a;
  --sky: #a7d8de;
}

.gradio-container {
  font-family: "DM Sans", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(249, 214, 138, 0.72), transparent 30%),
    radial-gradient(circle at top right, rgba(167, 216, 222, 0.66), transparent 28%),
    linear-gradient(180deg, #fff9ee 0%, #f5ead9 100%);
}

#storybook-shell {
  max-width: 1180px;
  margin: 0 auto;
  padding-bottom: 36px;
}

.hero-card,
.surface-card {
  background: var(--paper-soft);
  border: 1px solid rgba(139, 94, 60, 0.14);
  border-radius: 28px;
  box-shadow: 0 18px 48px rgba(31, 41, 51, 0.09);
}

.hero-card {
  padding: 28px;
}

.hero-card h1,
.section-title,
.storyboard-card h3 {
  font-family: "Fraunces", serif;
}

.hero-card p,
.status-copy,
.storyboard-card p {
  color: var(--muted);
}

.section-title {
  margin: 0 0 10px;
  color: var(--ink);
}

.storyboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.storyboard-card {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(139, 94, 60, 0.1);
  border-radius: 22px;
  padding: 16px;
}

.storyboard-card .eyebrow {
  display: inline-block;
  margin-bottom: 8px;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(246, 205, 122, 0.22);
  color: var(--caramel);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.storyboard-card h3 {
  margin: 0 0 8px;
  color: var(--ink);
  font-size: 22px;
}

.storyboard-card p {
  margin: 0;
  line-height: 1.55;
}

.video-tip {
  margin-top: 8px;
  font-size: 14px;
  color: var(--muted);
}
"""

INTRO_HTML = """
<div class="hero-card">
  <h1>Kid Drawing to Bedtime Story</h1>
  <p>Upload one drawing and the app will turn it into a calm three-part bedtime story, three matching illustrations, spoken narration, and one final story video with built-in playback controls.</p>
  <p><strong>Build:</strong> video-v3-20260315</p>
</div>
"""

_PIPELINE = KidStoryPipeline()


def _progress_adapter(progress: gr.Progress):
    def callback(value: float, message: str) -> None:
        progress(value, desc=message)

    return callback


def _build_storyboard_html(story: StoryPackage) -> str:
    cards: list[str] = []
    for index, part in enumerate(story.parts, start=1):
        cards.append(
            """
            <div class="storyboard-card">
              <div class="eyebrow">Part {index}</div>
              <h3>{scene_goal}</h3>
              <p>{story_text}</p>
            </div>
            """.format(
                index=index,
                scene_goal=html.escape(part.scene_goal),
                story_text=html.escape(part.story_text),
            )
        )
    return '<div class="storyboard-grid">' + "".join(cards) + "</div>"


def generate_story(image_path: str | None, progress: gr.Progress = gr.Progress(track_tqdm=False)):
    if not image_path:
        raise gr.Error("Upload a drawing before starting the story pipeline.")

    result = _PIPELINE.create_story(image_path, progress_callback=_progress_adapter(progress))
    status = (
        f"Created run `{result.run_id}`.\n\n"
        f"Saved assets to `{result.run_dir}` and wrote the manifest to `{result.manifest_path}`."
    )
    gallery_items = [
        (part.image_path, f"Part {index}: {part.scene_goal}")
        for index, part in enumerate(result.story.parts, start=1)
    ]
    return (
        status,
        result.story.title,
        result.story_markdown,
        _build_storyboard_html(result.story),
        gallery_items,
        result.video_path,
        result.narration_audio_path,
        result.manifest_path,
        str(Path(result.run_dir).resolve()),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=APP_CSS, title="Kid Drawing Story App") as demo:
        with gr.Column(elem_id="storybook-shell"):
            gr.HTML(INTRO_HTML)

            with gr.Row():
                with gr.Column(scale=5, min_width=320):
                    drawing_input = gr.Image(
                        label="Upload a drawing",
                        type="filepath",
                        image_mode="RGB",
                    )
                    create_button = gr.Button("Create Story", variant="primary")
                with gr.Column(scale=4, min_width=320, elem_classes=["surface-card"]):
                    gr.Markdown("### Run status")
                    status_output = gr.Markdown(elem_classes=["status-copy"])
                    title_output = gr.Textbox(label="Story title", interactive=False)
                    gr.Markdown(
                        "The final output below is a real MP4 video. Use its scrub bar to jump between the three story scenes without waiting for narration to finish.",
                        elem_classes=["video-tip"],
                    )

            with gr.Row():
                with gr.Column(scale=7, min_width=360):
                    gr.Markdown("## Story Video", elem_classes=["section-title"])
                    video_output = gr.Video(label=f"Final story video ({APP_BUILD})")
                    audio_output = gr.File(label="Narration audio")
                with gr.Column(scale=5, min_width=320):
                    gr.Markdown("## Story Script", elem_classes=["section-title"])
                    story_output = gr.Markdown()
                    manifest_output = gr.File(label="Run manifest")
                    run_dir_output = gr.Textbox(label="Run directory", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Storyboard", elem_classes=["section-title"])
                    storyboard_output = gr.HTML()
                    gallery_output = gr.Gallery(
                        label="Scene gallery",
                        columns=3,
                        object_fit="cover",
                    )

            create_button.click(
                fn=generate_story,
                inputs=[drawing_input],
                outputs=[
                    status_output,
                    title_output,
                    story_output,
                    storyboard_output,
                    gallery_output,
                    video_output,
                    audio_output,
                    manifest_output,
                    run_dir_output,
                ],
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()
