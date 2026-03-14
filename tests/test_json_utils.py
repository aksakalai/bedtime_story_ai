import unittest

from story_app.json_utils import extract_json_payload, parse_json_response
from story_app.schemas import DrawingDescription, SchemaError


class JsonUtilsTests(unittest.TestCase):
    def test_extract_json_payload_from_wrapped_text(self):
        raw = """
        Here is the result:
        {"summary":"A moonlit fox","characters":["fox"],"setting":"forest","visual_style":"crayon","color_palette":["blue","gold"],"safety_notes":["gentle","sleepy"]}
        """
        payload = extract_json_payload(raw)
        self.assertEqual(payload["summary"], "A moonlit fox")

    def test_parse_json_response_validates_schema(self):
        raw = '{"summary":"A moonlit fox","characters":["fox"],"setting":"forest","visual_style":"crayon","color_palette":["blue","gold"],"safety_notes":["gentle","sleepy"]}'
        description = parse_json_response(raw, DrawingDescription)
        self.assertEqual(description.characters, ["fox"])

    def test_extract_json_payload_raises_on_missing_object(self):
        with self.assertRaises(SchemaError):
            extract_json_payload("No JSON here at all")


if __name__ == "__main__":
    unittest.main()
