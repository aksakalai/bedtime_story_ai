import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_colab_bootstrap_installs_ffmpeg_and_supports_repo_path(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kid_drawing_story_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            line
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            for line in cell.get("source", [])
        )

        self.assertIn('/content/bedtime_story_ai', source)
        self.assertIn('"ffmpeg"', source)
        self.assertIn('"espeak-ng"', source)


if __name__ == "__main__":
    unittest.main()
