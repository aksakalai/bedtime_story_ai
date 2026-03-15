import unittest

from story_app.json_utils import extract_json_payload, parse_json_response
from story_app.schemas import DrawingDescription, SchemaError


class JsonUtilsTests(unittest.TestCase):
    def test_extract_json_payload_from_wrapped_text(self):
        raw = """
        Here is the result:
        {"text":"A moonlit fox waits in a quiet forest while blue and gold light glows across the trees."}
        """
        payload = extract_json_payload(raw)
        self.assertIn("moonlit fox", payload["text"])

    def test_parse_json_response_validates_schema(self):
        raw = '{"text":"A moonlit fox waits in a quiet forest while blue and gold light glows across the trees."}'
        description = parse_json_response(raw, DrawingDescription)
        self.assertIn("moonlit fox", description.text)

    def test_extract_json_payload_raises_on_missing_object(self):
        with self.assertRaises(SchemaError):
            extract_json_payload("No JSON here at all")


if __name__ == "__main__":
    unittest.main()
