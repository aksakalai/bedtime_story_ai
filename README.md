# Bedtime Story AI

Resident storyboard workbench for turning one uploaded drawing into:

- one grounded image description
- one full story-writing conversation
- three extracted story parts
- three generated storyboard images

## Current Scope

This repository is intentionally a Phase 2A baseline, not the full end-state product.

Current implemented flow:

`uploaded image -> grounded description -> sequential story part 1/2/3 generation -> per-part text-to-image generation -> artifact files under outputs/run_* -> Gradio debug UI`

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
- `storyboard_manifest.json`

## Not Yet Implemented

The following stages are intentionally out of scope for this baseline:

- narration or TTS
- word-level timing and highlighting
- video assembly or export

## Run

```bash
pip install -e .
python -m story_app
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

demo = story_app.app.build_demo()
demo.launch(debug=True, share=True, inline=True)
```

## Models

- Description: `Qwen/Qwen2.5-VL-3B-Instruct`
- Story drafting: `Qwen/Qwen2.5-VL-3B-Instruct`
- Storyboard images: `segmind/SSD-1B`
