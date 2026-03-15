from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from .config import APP_BUILD, DEFAULT_CONFIG
from .pipeline import KidStoryPipeline

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --cream: #f7f1e3;
  --card: rgba(255, 251, 245, 0.9);
  --ink: #24303d;
  --muted: #5d6875;
  --accent: #b26a3b;
  --border: rgba(36, 48, 61, 0.12);
}

.gradio-container {
  font-family: "IBM Plex Sans", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(255, 207, 153, 0.55), transparent 26%),
    radial-gradient(circle at top right, rgba(151, 201, 232, 0.45), transparent 24%),
    linear-gradient(180deg, #fbf4e7 0%, #efe6d7 100%);
}

#story-shell {
  max-width: 1320px;
  margin: 0 auto;
  padding-bottom: 32px;
}

.hero-card,
.surface-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: 0 16px 48px rgba(36, 48, 61, 0.08);
}

.hero-card {
  padding: 24px;
}

.hero-card h1,
.debug-title {
  font-family: "Fraunces", serif;
  color: var(--ink);
}

.hero-card p,
.debug-copy {
  color: var(--muted);
}

.story-part-card {
  padding: 22px;
  margin-top: 18px;
}

.story-part-image {
  min-height: 520px;
}

.story-part-image img {
  object-fit: contain;
  background: rgba(255, 248, 238, 0.9);
}
"""

INTRO_HTML = f"""
<div class="hero-card">
  <h1>Resident Storyboard Workbench</h1>
  <p class="debug-copy">This build turns one uploaded drawing into one grounded description, one three-part story, three separate text-to-image storyboard illustrations, three narration tracks, and one final highlighted story video.</p>
  <p class="debug-copy"><strong>Build:</strong> {APP_BUILD}</p>
  <p class="debug-copy"><strong>Description model:</strong> {DEFAULT_CONFIG.models.image_describer}</p>
  <p class="debug-copy"><strong>Story model:</strong> {DEFAULT_CONFIG.models.story_writer}</p>
  <p class="debug-copy"><strong>Part image model:</strong> {DEFAULT_CONFIG.models.part_image_generator}</p>
  <p class="debug-copy"><strong>Narration model:</strong> {DEFAULT_CONFIG.models.part_narrator}</p>
  <p class="debug-copy"><strong>Word timing model:</strong> {DEFAULT_CONFIG.models.word_aligner}</p>
</div>
"""

_PIPELINE = KidStoryPipeline()


def _progress_adapter(progress: gr.Progress):
    def callback(value: float, message: str) -> None:
        progress(value, desc=message)

    return callback


def _load_image_for_ui(image_path: str | None):
    if not image_path:
        return None
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def generate_story(image_path: str | None, progress: gr.Progress = gr.Progress(track_tqdm=False)):
    if not image_path:
        raise gr.Error("Upload a drawing before starting.")

    try:
        result = _PIPELINE.create_story_package(
            image_path,
            progress_callback=_progress_adapter(progress),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    status = (
        f"Run `{result.run_id}` completed.\n\n"
        f"Artifacts saved to `{result.run_dir}`."
    )
    return (
        status,
        _load_image_for_ui(result.input_image_path),
        result.description.description_text,
        result.full_conversation_text,
        result.part_1_text,
        _load_image_for_ui(result.story_part_1_image_path),
        result.story_part_1_audio_path,
        result.part_2_text,
        _load_image_for_ui(result.story_part_2_image_path),
        result.story_part_2_audio_path,
        result.part_3_text,
        _load_image_for_ui(result.story_part_3_image_path),
        result.story_part_3_audio_path,
        result.final_story_video_path,
        str(Path(result.run_dir).resolve()),
    )


def preload_models() -> str:
    _PIPELINE.preload_models()
    return "Models are loaded and ready in the current Python session."


def clear_loaded_models() -> str:
    _PIPELINE.clear_loaded_models()
    return "Loaded models have been cleared from the current Python session."


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=APP_CSS, title="Resident Storyboard Workbench") as demo:
        with gr.Column(elem_id="story-shell"):
            gr.HTML(INTRO_HTML)

            with gr.Row():
                with gr.Column(scale=4, min_width=320):
                    input_image = gr.Image(
                        label="Upload or capture a drawing",
                        type="filepath",
                        image_mode="RGB",
                        sources=["upload", "webcam"],
                    )
                    create_button = gr.Button("Create", variant="primary")
                with gr.Column(scale=5, min_width=320, elem_classes=["surface-card"]):
                    gr.Markdown("### Run status", elem_classes=["debug-title"])
                    status_output = gr.Markdown()
                    run_dir_output = gr.Textbox(label="Run directory", interactive=False)

            with gr.Column(elem_classes=["surface-card"]):
                preview_image = gr.Image(label="Uploaded image", interactive=False)
                description_output = gr.Textbox(
                    label="Generated description",
                    lines=8,
                    interactive=False,
                )
                full_conversation_output = gr.Textbox(
                    label="Full story conversation",
                    lines=18,
                    interactive=False,
                )
                final_video_output = gr.Video(
                    label="Final story video",
                    interactive=False,
                )

            with gr.Column():
                with gr.Column(elem_classes=["surface-card", "story-part-card"]):
                    gr.Markdown("### Story Part 1", elem_classes=["debug-title"])
                    part_1_output = gr.Textbox(lines=7, interactive=False)
                    part_1_image_output = gr.Image(
                        label="Part 1 image",
                        interactive=False,
                        height=560,
                        elem_classes=["story-part-image"],
                    )
                    part_1_audio_output = gr.Audio(
                        label="Part 1 narration",
                        interactive=False,
                    )
                with gr.Column(elem_classes=["surface-card", "story-part-card"]):
                    gr.Markdown("### Story Part 2", elem_classes=["debug-title"])
                    part_2_output = gr.Textbox(lines=7, interactive=False)
                    part_2_image_output = gr.Image(
                        label="Part 2 image",
                        interactive=False,
                        height=560,
                        elem_classes=["story-part-image"],
                    )
                    part_2_audio_output = gr.Audio(
                        label="Part 2 narration",
                        interactive=False,
                    )
                with gr.Column(elem_classes=["surface-card", "story-part-card"]):
                    gr.Markdown("### Story Part 3", elem_classes=["debug-title"])
                    part_3_output = gr.Textbox(lines=7, interactive=False)
                    part_3_image_output = gr.Image(
                        label="Part 3 image",
                        interactive=False,
                        height=560,
                        elem_classes=["story-part-image"],
                    )
                    part_3_audio_output = gr.Audio(
                        label="Part 3 narration",
                        interactive=False,
                    )

            create_button.click(
                fn=generate_story,
                inputs=[input_image],
                outputs=[
                    status_output,
                    preview_image,
                    description_output,
                    full_conversation_output,
                    part_1_output,
                    part_1_image_output,
                    part_1_audio_output,
                    part_2_output,
                    part_2_image_output,
                    part_2_audio_output,
                    part_3_output,
                    part_3_image_output,
                    part_3_audio_output,
                    final_video_output,
                    run_dir_output,
                ],
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(debug=True, share=True, inline=True)
