from __future__ import annotations

from pathlib import Path

import gradio as gr

from .pipeline import KidStoryPipeline

PLAYBACK_BOOTSTRAP_JS = """
<script>
(() => {
  const initialized = new WeakSet();

  const pickSegment = (timeline, currentTime) => {
    for (const segment of timeline) {
      if (currentTime >= segment.start && currentTime < segment.end) {
        return segment;
      }
    }
    return timeline[timeline.length - 1] || null;
  };

  const syncPlayer = (player) => {
    const timelineAttr = player.getAttribute("data-timeline");
    if (!timelineAttr) return;

    let timeline = [];
    try {
      timeline = JSON.parse(timelineAttr);
    } catch (error) {
      console.warn("Failed to parse story timeline.", error);
      return;
    }
    if (!timeline.length) return;

    const audio = player.querySelector("[data-storybook-audio='true']");
    const stage = player.querySelector("[data-storybook-stage='true']");
    const text = player.querySelector("[data-storybook-text='true']");
    if (!audio || !stage || !text) return;

    const applyFrame = () => {
      const segment = pickSegment(timeline, audio.currentTime || 0);
      if (!segment) return;
      stage.style.backgroundImage = `url('${segment.image}')`;
      text.textContent = segment.text;
    };

    if (!initialized.has(audio)) {
      audio.addEventListener("timeupdate", applyFrame);
      audio.addEventListener("seeked", applyFrame);
      audio.addEventListener("loadedmetadata", applyFrame);
      audio.addEventListener("play", applyFrame);
      initialized.add(audio);
    }

    applyFrame();
  };

  const initAllPlayers = (root = document) => {
    root.querySelectorAll("[data-storybook-player='true']").forEach(syncPlayer);
  };

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof HTMLElement)) return;
        if (node.matches?.("[data-storybook-player='true']")) {
          syncPlayer(node);
        } else {
          initAllPlayers(node);
        }
      });
    }
  });

  const start = () => {
    initAllPlayers(document);
    observer.observe(document.body, { childList: true, subtree: true });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
</script>
"""

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --paper: #fffaf0;
  --ink: #1f2933;
  --berry: #8b5e3c;
  --sky: #a7d8de;
  --sun: #f9d68a;
  --forest: #7a9d76;
}

.gradio-container {
  font-family: "DM Sans", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(249,214,138,0.8), transparent 32%),
    radial-gradient(circle at top right, rgba(167,216,222,0.7), transparent 28%),
    linear-gradient(180deg, #fff9ee 0%, #f8efe4 100%);
}

#storybook-shell {
  max-width: 1180px;
  margin: 0 auto;
}

.hero-card, .result-card {
  background: rgba(255, 250, 240, 0.92);
  border: 1px solid rgba(139, 94, 60, 0.14);
  border-radius: 28px;
  box-shadow: 0 18px 48px rgba(31, 41, 51, 0.09);
}

.hero-card {
  padding: 28px;
}

.hero-card h1, .storybook-copy h2 {
  font-family: "Fraunces", serif;
}

.hero-card p {
  color: #415164;
  font-size: 16px;
}

.storybook-player {
  display: grid;
  gap: 16px;
}

.storybook-stage {
  position: relative;
  min-height: 430px;
  background-size: cover;
  background-position: center;
  border-radius: 26px;
  overflow: hidden;
}

.storybook-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(16, 28, 36, 0.08), rgba(16, 28, 36, 0.62));
}

.storybook-copy {
  position: absolute;
  left: 24px;
  right: 24px;
  bottom: 24px;
  color: white;
  z-index: 1;
  padding: 20px 22px;
  border-radius: 22px;
  background: rgba(21, 26, 38, 0.4);
  backdrop-filter: blur(10px);
}

.storybook-kicker {
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 11px;
  opacity: 0.78;
  margin-bottom: 8px;
}
"""

INTRO_HTML = """
<div class="hero-card">
  <h1>Kid Drawing to Bedtime Story</h1>
  <p>Upload a child's drawing and create a calm three-part story with matching illustrations and synced narration. The app is designed for Colab GPU sessions and saves every run so nothing gets lost between stages.</p>
</div>
"""

_PIPELINE = KidStoryPipeline()


def _progress_adapter(progress: gr.Progress):
    def callback(value: float, message: str) -> None:
        progress(value, desc=message)

    return callback


def generate_story(image_path: str | None, progress: gr.Progress = gr.Progress(track_tqdm=False)):
    if not image_path:
        raise gr.Error("Upload a drawing before starting the story pipeline.")

    result = _PIPELINE.create_story(image_path, progress_callback=_progress_adapter(progress))
    status = (
        f"Created run `{result.run_id}`.\n\n"
        f"Saved assets to `{result.run_dir}` and wrote the manifest to `{result.manifest_path}`."
    )
    return (
        status,
        result.story.title,
        result.story_markdown,
        result.image_gallery,
        result.narration_audio_path,
        result.playback_html,
        result.manifest_path,
        str(Path(result.run_dir).resolve()),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=APP_CSS, head=PLAYBACK_BOOTSTRAP_JS, title="Kid Drawing Story App") as demo:
        with gr.Column(elem_id="storybook-shell"):
            gr.HTML(INTRO_HTML)

            with gr.Row():
                with gr.Column(scale=1):
                    drawing_input = gr.Image(
                        label="Upload a drawing",
                        type="filepath",
                        image_mode="RGB",
                    )
                    create_button = gr.Button("Create Story", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### Run status")
                    status_output = gr.Markdown()
                    title_output = gr.Textbox(label="Story title", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Story")
                    story_output = gr.Markdown()
                gallery_output = gr.Gallery(
                    label="Generated scenes",
                    columns=3,
                    object_fit="cover",
                )

            with gr.Row():
                audio_output = gr.File(label="Narration audio")
                manifest_output = gr.File(label="Run manifest")

            gr.Markdown("### Interactive playback")
            playback_output = gr.HTML()
            run_dir_output = gr.Textbox(label="Run directory", interactive=False)

            create_button.click(
                fn=generate_story,
                inputs=[drawing_input],
                outputs=[
                    status_output,
                    title_output,
                    story_output,
                    gallery_output,
                    audio_output,
                    playback_output,
                    manifest_output,
                    run_dir_output,
                ],
            )

    demo.queue()
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()
