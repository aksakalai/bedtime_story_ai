# Kid Drawing Story App

This project turns a child's drawing into a three-part bedtime story with matching illustrations and narrated playback. It is designed to run best on a free Google Colab GPU through a Gradio interface.

## What it does

- Upload one drawing
- Describe the drawing with a lightweight vision-language model
- Write a calm three-part bedtime story in English
- Generate three matching storybook-style illustrations
- Narrate each story part and sync the images to the audio
- Save every run into `outputs/`

## Project layout

- `story_app/`: app code and pipeline logic
- `tests/`: lightweight unit tests that do not require model downloads
- `notebooks/kid_drawing_story_colab.ipynb`: Colab launcher notebook
- `outputs/`: generated runs

## Local development

```bash
python3 -m unittest discover -s tests
python3 -m compileall story_app
```

Running the full app locally is possible, but the intended environment is Colab with a GPU:

```bash
pip install -e .
python3 -m story_app
```

## Colab workflow

1. Upload or clone this full project into Colab.
2. Open `notebooks/kid_drawing_story_colab.ipynb`.
3. Switch the runtime to `GPU`.
4. Run the cells to install dependencies and launch the Gradio app.

If you open only the notebook file without the rest of the repo, the bootstrap cell will ask you to clone or upload the full project first.
