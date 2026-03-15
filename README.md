# Bedtime Story AI

Phase 1 debug workbench for turning one uploaded drawing into:

- one grounded image description
- one full story-writing conversation
- three extracted story parts

## Run

```bash
pip install -e .
python -m story_app
```

## Models

- Description: `Qwen/Qwen2-VL-2B-Instruct`
- Story drafting: `Qwen/Qwen2.5-1.5B-Instruct`
