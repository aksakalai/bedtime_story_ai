# Bedtime Story AI

Phase 1 debug workbench for turning one uploaded drawing into:

- one clean image description
- story part 1
- story part 2
- story part 3

This phase intentionally stops before image generation, narration, and video.

## Run

```bash
pip install -e .
python -m story_app
```

## Models

- Description: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- Story drafting: `Qwen/Qwen2.5-1.5B-Instruct`
