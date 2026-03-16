import tempfile
import unittest
from pathlib import Path

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import ValidationError


VALID_DESCRIPTION = (
    "A small blue house with a red roof stands between two green trees beside a little blue car under a yellow sun."
)
VALID_PART_1 = (
    "The small blue house glowed softly under the yellow sun while the little blue car rested between the two green "
    "trees. Everything felt quiet and warm, and the stillness held one tiny question about what gentle moment might "
    "begin next."
)
VALID_PART_2 = (
    "A light breeze stirred the two green trees, and the little blue car seemed to wait patiently beside the house. "
    "The quiet scene felt full of promise, as if the warm sunlight were guiding the whole place toward a tender new "
    "moment."
)
VALID_PART_3 = (
    "By evening, the blue house, the two green trees, and the little blue car all rested beneath the fading yellow "
    "sun. The gentle scene settled into peace, and the day ended with a calm feeling that made everything seem safe "
    "and still."
)


class SharedFakeProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.description_calls = 0
        self.story_calls = 0
        self.unload_calls = 0
        SharedFakeProvider.instances.append(self)

    def describe(self, image_path, messages):
        self.description_calls += 1
        return VALID_DESCRIPTION

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        step_index = ((self.story_calls - 1) % 3) + 1
        if step_index == 1:
            return VALID_PART_1
        if step_index == 2:
            return VALID_PART_2
        return VALID_PART_3

    def unload(self, clear_cache=False):
        self.unload_calls += 1
        return None


class PackageFakeProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.description_calls = 0
        self.story_calls = 0
        self.image_prompt_calls = []
        type(self).instances.append(self)

    def describe(self, image_path, messages):
        self.description_calls += 1
        return VALID_DESCRIPTION

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        if self.story_calls == 1:
            return VALID_PART_1
        if self.story_calls == 2:
            return VALID_PART_2
        return VALID_PART_3

    def generate_image_prompt(self, messages, max_new_tokens=None):
        call_index = len(self.image_prompt_calls) + 1
        self.image_prompt_calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
            }
        )
        if call_index == 1:
            return "little blue car by the two green trees under the yellow sun"
        if call_index == 2:
            return "little blue car notices a glowing lantern near the first tree"
        return "little blue car rests by the blue house while the lantern glows softly"

    def unload(self, clear_cache=False):
        return None


class NonEosDescriptionProvider:
    def __init__(self, config):
        self.config = config

    def describe(self, image_path, messages):
        raise ValidationError("Description generation did not finish naturally before the safety limit.")

    def generate_part(self, image_path, messages):
        return VALID_PART_1

    def unload(self, clear_cache=False):
        return None


class NonEosStoryProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.story_calls = 0
        NonEosStoryProvider.instances.append(self)

    def describe(self, image_path, messages):
        return VALID_DESCRIPTION

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        if self.story_calls == 1:
            return VALID_PART_1
        raise ValidationError("Story generation did not finish naturally before the safety limit.")

    def unload(self, clear_cache=False):
        return None


class NonEosImagePromptProvider(PackageFakeProvider):
    def generate_image_prompt(self, messages, max_new_tokens=None):
        raise ValidationError("Image prompt summary generation did not finish naturally before the safety limit.")


class BudgetAwareFakeImageGenerator:
    instances = []

    def __init__(self, config):
        self.config = config
        self.prompt_token_limit_calls = []
        self.validated_prompts = []
        self.generated_prompts = []
        type(self).instances.append(self)

    def get_prompt_token_limit(self, *, buffer_tokens=0):
        self.prompt_token_limit_calls.append(buffer_tokens)
        return 41

    def validate_prompt_token_budget(self, prompt_text, *, buffer_tokens=0, strict=True):
        self.validated_prompts.append(
            {
                "prompt_text": prompt_text,
                "buffer_tokens": buffer_tokens,
                "strict": strict,
            }
        )
        return {"tokenizer": 12, "tokenizer_2": 12}

    def generate(self, *, prompt_text, negative_prompt_text, seed, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-image")
        self.generated_prompts.append(prompt_text)
        return output_path

    def unload(self, clear_cache=False):
        return None


class OverBudgetFakeImageGenerator(BudgetAwareFakeImageGenerator):
    instances = []

    def validate_prompt_token_budget(self, prompt_text, *, buffer_tokens=0, strict=True):
        raise ValidationError("Image prompt exceeds the current image model token budget: tokenizer=45>41")


class PackageFakeNarrator:
    def __init__(self, config):
        self.config = config

    def narrate(self, *, text, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio")
        return output_path, 1.2

    def unload(self, clear_cache=False):
        return None


class PackageFakeWordAligner:
    def __init__(self, config):
        self.config = config

    def transcribe_words(self, audio_path):
        return [
            {"word": "The", "start": 0.0, "end": 0.2},
            {"word": "story", "start": 0.2, "end": 0.4},
        ]

    def unload(self, clear_cache=False):
        return None


class PackageFakeVideoAssembler:
    def __init__(self, config):
        self.config = config

    def render_story_part_clip(
        self,
        *,
        image_path,
        audio_path,
        subtitle_path,
        overlay_layout,
        total_duration_seconds,
        output_path,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-clip")
        return output_path

    def concatenate_story_clips(self, *, clip_paths, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")
        return output_path


class PipelineTests(unittest.TestCase):
    def _create_input_file(self, root: Path) -> Path:
        path = root / "input.png"
        path.write_bytes(b"fake-image")
        return path

    def test_pipeline_stops_when_description_does_not_finish_with_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosDescriptionProvider,
                writer_factory=NonEosDescriptionProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_stops_before_part_3_when_part_2_does_not_finish_with_eos(self):
        NonEosStoryProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosStoryProvider,
                writer_factory=NonEosStoryProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))
            self.assertEqual(NonEosStoryProvider.instances[0].story_calls, 2)

    def test_pipeline_smoke_path_saves_minimal_story_artifacts(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            result = pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(result.description.description_text, VALID_DESCRIPTION)
            self.assertEqual(result.part_1_text, VALID_PART_1)
            self.assertEqual(result.part_2_text, VALID_PART_2)
            self.assertEqual(result.part_3_text, VALID_PART_3)
            self.assertIn("SYSTEM:", result.full_conversation_text)
            self.assertIn(VALID_DESCRIPTION, result.full_conversation_text)
            self.assertIn(VALID_PART_1, result.full_conversation_text)
            self.assertIn(VALID_PART_2, result.full_conversation_text)
            self.assertIn(VALID_PART_3, result.full_conversation_text)

            input_images = list(result.run_dir.glob("input_image*"))
            self.assertEqual(len(input_images), 1)

            expected_files = [
                "description_prompt.txt",
                "description.txt",
                "story_conversation.txt",
                "story_part_1.txt",
                "story_part_2.txt",
                "story_part_3.txt",
            ]
            for filename in expected_files:
                self.assertTrue((result.run_dir / filename).exists(), filename)

    def test_pipeline_reuses_single_provider_instance_across_runs(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            pipeline.create_story_draft(self._create_input_file(root))
            pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(len(SharedFakeProvider.instances), 1)
            self.assertEqual(SharedFakeProvider.instances[0].description_calls, 2)
            self.assertEqual(SharedFakeProvider.instances[0].story_calls, 6)
            self.assertEqual(SharedFakeProvider.instances[0].unload_calls, 0)

    def test_pipeline_can_clear_loaded_models(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            pipeline.create_story_draft(self._create_input_file(root))
            pipeline.clear_loaded_models()

            self.assertEqual(SharedFakeProvider.instances[0].unload_calls, 1)

    def test_story_package_uses_image_token_budget_and_previous_page_prompt(self):
        PackageFakeProvider.instances = []
        BudgetAwareFakeImageGenerator.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(
                    outputs_root=root / "outputs",
                    image_prompt_summary_max_tokens=5,
                ),
                describer_factory=PackageFakeProvider,
                writer_factory=PackageFakeProvider,
                image_generator_factory=BudgetAwareFakeImageGenerator,
                narrator_factory=PackageFakeNarrator,
                word_aligner_factory=PackageFakeWordAligner,
                video_assembler_factory=PackageFakeVideoAssembler,
            )
            result = pipeline.create_story_package(self._create_input_file(root))

            writer = PackageFakeProvider.instances[0]
            image_generator = BudgetAwareFakeImageGenerator.instances[0]
            self.assertEqual(
                [call["max_new_tokens"] for call in writer.image_prompt_calls],
                [41, 41, 41],
            )
            self.assertEqual(image_generator.prompt_token_limit_calls, [1])
            self.assertEqual(len(image_generator.validated_prompts), 3)
            self.assertTrue(all(call["strict"] for call in image_generator.validated_prompts))
            self.assertIn("Previous page final image prompt:", writer.image_prompt_calls[1]["messages"][1]["content"])
            self.assertIn(
                "little blue car by the two green trees under the yellow sun, children's picture-book illustration",
                writer.image_prompt_calls[1]["messages"][1]["content"],
            )
            self.assertIn(
                "little blue car notices a glowing lantern near the first tree, children's picture-book illustration",
                writer.image_prompt_calls[2]["messages"][1]["content"],
            )
            self.assertTrue(Path(result.final_story_video_path).exists())

    def test_story_package_stops_when_image_prompt_summary_does_not_finish_with_eos(self):
        PackageFakeProvider.instances = []
        BudgetAwareFakeImageGenerator.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosImagePromptProvider,
                writer_factory=NonEosImagePromptProvider,
                image_generator_factory=BudgetAwareFakeImageGenerator,
                narrator_factory=PackageFakeNarrator,
                word_aligner_factory=PackageFakeWordAligner,
                video_assembler_factory=PackageFakeVideoAssembler,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_package(self._create_input_file(root))
            self.assertEqual(BudgetAwareFakeImageGenerator.instances[0].generated_prompts, [])

    def test_story_package_stops_when_image_prompt_exceeds_budget(self):
        PackageFakeProvider.instances = []
        OverBudgetFakeImageGenerator.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=PackageFakeProvider,
                writer_factory=PackageFakeProvider,
                image_generator_factory=OverBudgetFakeImageGenerator,
                narrator_factory=PackageFakeNarrator,
                word_aligner_factory=PackageFakeWordAligner,
                video_assembler_factory=PackageFakeVideoAssembler,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_package(self._create_input_file(root))
            self.assertEqual(OverBudgetFakeImageGenerator.instances[0].generated_prompts, [])


if __name__ == "__main__":
    unittest.main()
