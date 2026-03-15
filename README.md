# Bedtime Story AI

Resident storyboard workbench for turning one uploaded drawing into:

- one grounded image description
- one full story-writing conversation
- three extracted story parts
- three generated storyboard images
- three generated narration tracks
- one final highlighted story video

## Current Scope

This repository is intentionally a Phase 2B baseline, not the full end-state product.

Current implemented flow:

`uploaded image -> grounded description -> sequential story part 1/2/3 generation -> per-part text-to-image generation -> per-part narration generation -> word-timed text overlay clip rendering -> final story video -> artifact files under outputs/run_* -> Gradio debug UI`

Current architecture:

- `story_app/config.py` holds build and model settings
- `story_app/prompts.py` holds prompt and validation rules
- `story_app/providers.py` wraps the Hugging Face model calls
- `story_app/pipeline.py` orchestrates the run and writes artifacts
- `story_app/app.py` exposes the Gradio UI

Stable interfaces for the Colab workflow:

- `story_app.config.DEFAULT_CONFIG`
- `story_app.app.generate_story(...)`
- `story_app.app.build_demo()`
- `KidStoryPipeline.create_story_draft(...)`
- `KidStoryPipeline.create_story_package(...)`

Stable artifact contract per run:

- copied input image
- `description_prompt.txt`
- `description.txt`
- `story_conversation.txt`
- `story_part_1.txt`
- `story_part_2.txt`
- `story_part_3.txt`
- `image_prompt_part_1.txt`
- `image_prompt_part_2.txt`
- `image_prompt_part_3.txt`
- `story_part_1_image.png`
- `story_part_2_image.png`
- `story_part_3_image.png`
- `story_part_1_audio.wav`
- `story_part_2_audio.wav`
- `story_part_3_audio.wav`
- `story_part_1.ass`
- `story_part_2.ass`
- `story_part_3.ass`
- `story_part_1_clip.mp4`
- `story_part_2_clip.mp4`
- `story_part_3_clip.mp4`
- `final_story_video.mp4`
- `storyboard_manifest.json`

## Not Yet Implemented

The following stages are intentionally out of scope for this baseline:

- richer transitions and motion design
- word-level highlighting polish beyond the current karaoke-style implementation

## Run

```bash
pip install -e .
python -m story_app
```

For Kokoro narration in Colab, install the system phonemizer dependency once per fresh runtime:

```bash
apt-get -qq -y install espeak-ng
```

If your runtime does not already include ffmpeg, install it once per fresh runtime:

```bash
apt-get -qq -y install ffmpeg
```

## Colab Entry Surface

The integration surface used by Colab should remain stable:

```python
import story_app.app
import story_app.config

print("APP_BUILD:", story_app.config.APP_BUILD)
print("Description model:", story_app.config.DEFAULT_CONFIG.models.image_describer)
print("Story model:", story_app.config.DEFAULT_CONFIG.models.story_writer)
print("Part image model:", story_app.config.DEFAULT_CONFIG.models.part_image_generator)
print("Narration model:", story_app.config.DEFAULT_CONFIG.models.part_narrator)
print("Word timing model:", story_app.config.DEFAULT_CONFIG.models.word_aligner)

demo = story_app.app.build_demo()
demo.launch(debug=True, share=True, inline=True)
```

## Models

- Description: `Qwen/Qwen2.5-VL-3B-Instruct`
- Story drafting: `Qwen/Qwen2.5-VL-3B-Instruct`
- Storyboard images: `segmind/SSD-1B`
- Narration: `hexgrad/Kokoro-82M`
- Word timing: `tiny.en` via OpenAI Whisper
