import tempfile
import unittest
from pathlib import Path

from story_app.config import GenerationConfig
from story_app.pipeline import KidStoryPipeline
from story_app.schemas import ValidationError


INITIAL_ANCHOR = (
    "actor: curious little blue fish\n"
    "actor_colors: blue and purple\n"
    "actor_traits: bright eyes, long fins\n"
    "scene: underwater garden\n"
    "scene_colors: teal water, green sea plants\n"
    "object_1: sea plants\n"
    "object_1_colors: green\n"
    "object_2: coral arch\n"
    "object_2_colors: orange\n"
    "object_3: none\n"
    "object_3_colors: none\n"
    "secondary_actor: none\n"
    "secondary_actor_colors: none\n"
    "page_event: calm exploration\n"
    "mood: peaceful"
)

PART_2_ANCHOR = (
    "actor: curious little blue fish\n"
    "actor_colors: blue and purple\n"
    "actor_traits: bright eyes, long fins\n"
    "scene: underwater garden near the coral arch\n"
    "scene_colors: teal water, green sea plants, orange coral\n"
    "object_1: sea plants\n"
    "object_1_colors: green\n"
    "object_2: glowing creature\n"
    "object_2_colors: pale gold\n"
    "object_3: coral arch\n"
    "object_3_colors: orange\n"
    "secondary_actor: none\n"
    "secondary_actor_colors: none\n"
    "page_event: glowing creature appears\n"
    "mood: curious"
)

PART_3_ANCHOR = (
    "actor: curious little blue fish\n"
    "actor_colors: blue and purple\n"
    "actor_traits: bright eyes, long fins\n"
    "scene: underwater garden\n"
    "scene_colors: teal water, green sea plants\n"
    "object_1: sea plants\n"
    "object_1_colors: green\n"
    "object_2: glowing creature\n"
    "object_2_colors: pale gold\n"
    "object_3: none\n"
    "object_3_colors: none\n"
    "secondary_actor: none\n"
    "secondary_actor_colors: none\n"
    "page_event: calm farewell\n"
    "mood: peaceful and warm"
)

PART_1_TEXT = (
    "A curious little blue fish with purple shimmer glided through the underwater garden beside the green sea "
    "plants. The orange coral arch glowed softly nearby while the water stayed calm and bright. The little fish "
    "slowed down as if it had just noticed something new ahead."
)

PART_2_TEXT = (
    "The curious little blue fish with purple shimmer swam closer to a pale gold glowing creature near the orange "
    "coral arch. The green sea plants bent gently as the water flickered around them. The little fish circled the "
    "glow with wide bright eyes."
)

PART_3_TEXT = (
    "The curious little blue fish with purple shimmer gave the pale gold glowing creature a calm final look. The "
    "green sea plants swayed softly as the glow faded into the teal water. The little fish drifted home through the "
    "underwater garden feeling safe and peaceful."
)


class SharedFakeProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.initial_anchor_calls = 0
        self.anchor_update_calls = 0
        self.story_calls = 0
        self.unload_calls = 0
        SharedFakeProvider.instances.append(self)

    def extract_initial_anchor(self, image_path, messages):
        self.initial_anchor_calls += 1
        return INITIAL_ANCHOR

    def generate_next_anchor(self, messages):
        self.anchor_update_calls += 1
        if self.anchor_update_calls == 1:
            return PART_2_ANCHOR
        return PART_3_ANCHOR

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        if self.story_calls == 1:
            return PART_1_TEXT
        if self.story_calls == 2:
            return PART_2_TEXT
        return PART_3_TEXT

    def unload(self, clear_cache=False):
        self.unload_calls += 1
        return None


class PackageFakeProvider(SharedFakeProvider):
    instances = []

    def __init__(self, config):
        super().__init__(config)
        self.image_prompt_calls = []
        PackageFakeProvider.instances.append(self)

    def generate_image_prompt(self, messages, max_new_tokens=None):
        call_index = len(self.image_prompt_calls) + 1
        self.image_prompt_calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
            }
        )
        if call_index == 1:
            return "curious little blue fish with purple shimmer among green sea plants in teal water"
        if call_index == 2:
            return "curious little blue fish with purple shimmer faces a pale gold glowing creature by the orange coral arch"
        return "curious little blue fish with purple shimmer drifts calmly through teal water as the pale gold glow fades"


class NonEosInitialAnchorProvider:
    def __init__(self, config):
        self.config = config

    def extract_initial_anchor(self, image_path, messages):
        raise ValidationError("Initial anchor generation did not finish naturally before the safety limit.")

    def generate_next_anchor(self, messages):
        return PART_2_ANCHOR

    def generate_part(self, image_path, messages):
        return PART_1_TEXT

    def unload(self, clear_cache=False):
        return None


class NonEosStoryProvider:
    instances = []

    def __init__(self, config):
        self.config = config
        self.story_calls = 0
        NonEosStoryProvider.instances.append(self)

    def extract_initial_anchor(self, image_path, messages):
        return INITIAL_ANCHOR

    def generate_next_anchor(self, messages):
        return PART_2_ANCHOR

    def generate_part(self, image_path, messages):
        self.story_calls += 1
        if self.story_calls == 1:
            return PART_1_TEXT
        raise ValidationError("Story generation did not finish naturally before the safety limit.")

    def unload(self, clear_cache=False):
        return None


class NonEosAnchorUpdateProvider:
    def __init__(self, config):
        self.config = config

    def extract_initial_anchor(self, image_path, messages):
        return INITIAL_ANCHOR

    def generate_next_anchor(self, messages):
        raise ValidationError("Next-page anchor generation did not finish naturally before the safety limit.")

    def generate_part(self, image_path, messages):
        return PART_1_TEXT

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

    def test_pipeline_stops_when_initial_anchor_does_not_finish_with_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosInitialAnchorProvider,
                writer_factory=NonEosInitialAnchorProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_stops_when_part_2_does_not_finish_with_eos(self):
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
            self.assertEqual(NonEosStoryProvider.instances[-1].story_calls, 2)

    def test_pipeline_stops_when_next_anchor_does_not_finish_with_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=NonEosAnchorUpdateProvider,
                writer_factory=NonEosAnchorUpdateProvider,
            )
            with self.assertRaises(ValidationError):
                pipeline.create_story_draft(self._create_input_file(root))

    def test_pipeline_smoke_path_saves_story_and_anchor_artifacts(self):
        SharedFakeProvider.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline = KidStoryPipeline(
                config=GenerationConfig(outputs_root=root / "outputs"),
                describer_factory=SharedFakeProvider,
                writer_factory=SharedFakeProvider,
            )
            result = pipeline.create_story_draft(self._create_input_file(root))

            self.assertEqual(result.description.description_text, INITIAL_ANCHOR)
            self.assertEqual(result.part_1_text, PART_1_TEXT)
            self.assertEqual(result.part_2_text, PART_2_TEXT)
            self.assertEqual(result.part_3_text, PART_3_TEXT)
            self.assertTrue((result.run_dir / "story_part_1_anchor.txt").exists())
            self.assertTrue((result.run_dir / "story_part_2_anchor.txt").exists())
            self.assertTrue((result.run_dir / "story_part_3_anchor.txt").exists())
            self.assertTrue((result.run_dir / "story_conversation.txt").exists())

    def test_pipeline_reuses_one_describer_and_one_writer_across_runs(self):
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

            self.assertEqual(len(SharedFakeProvider.instances), 2)
            self.assertEqual(SharedFakeProvider.instances[0].initial_anchor_calls, 2)
            self.assertEqual(SharedFakeProvider.instances[0].story_calls, 0)
            self.assertEqual(SharedFakeProvider.instances[1].initial_anchor_calls, 0)
            self.assertEqual(SharedFakeProvider.instances[1].anchor_update_calls, 4)
            self.assertEqual(SharedFakeProvider.instances[1].story_calls, 6)

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
            self.assertEqual(SharedFakeProvider.instances[1].unload_calls, 1)

    def test_story_package_uses_page_anchors_for_image_prompts(self):
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

            writer = PackageFakeProvider.instances[-1]
            image_generator = BudgetAwareFakeImageGenerator.instances[0]
            self.assertEqual([call["max_new_tokens"] for call in writer.image_prompt_calls], [41, 41, 41])
            self.assertIn("Current page anchor:", writer.image_prompt_calls[0]["messages"][1]["content"])
            self.assertIn("Use only the current page anchor, not any previous story text", writer.image_prompt_calls[1]["messages"][1]["content"])
            self.assertNotIn("Story moment:", writer.image_prompt_calls[0]["messages"][1]["content"])
            self.assertEqual(image_generator.prompt_token_limit_calls, [1])
            self.assertEqual(len(image_generator.validated_prompts), 3)
            self.assertTrue(all(call["strict"] for call in image_generator.validated_prompts))
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
