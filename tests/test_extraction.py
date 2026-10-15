"""Extraction safeguards tested without endpoint access."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extractor", ROOT / "extraction/markdown_to_json.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResponseRulesTests(unittest.TestCase):
    def test_nonfinite_json_and_nonobject_are_rejected(self):
        for response in ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', "[]"]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                MODULE.parse_response(response)

    def test_json_fence_is_stripped(self):
        record = MODULE.parse_response('```json\n{"x": 1}\n```')
        self.assertEqual(record, {"x": 1})


class HarnessSessionTests(unittest.TestCase):
    def test_failed_response_does_not_create_record(self):
        def failing_complete(system, user, max_tokens):
            raise ValueError("Model response stopped early (length)")

        client = SimpleNamespace(complete=failing_complete)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "article.md"
            output = Path(folder) / "record.json"
            source.write_text("Formulation text")
            with self.assertRaises(ValueError):
                MODULE.extract_file(source, output, client, "test", 50.0, 1000, 100)
            self.assertFalse(output.exists())

    def test_empty_record_flags_manual_review(self):
        reply = SimpleNamespace(
            text="{}", finish_reason="stop", model="test-model", backend="harness"
        )
        client = SimpleNamespace(complete=lambda system, user, max_tokens: reply)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "article.md"
            output = Path(folder) / "record.json"
            source.write_text("Article text " * 20)
            metadata = MODULE.extract_file(source, output, client, "test", 0.0, 10, 100)
            self.assertTrue(metadata["manual_review_required"])
            self.assertTrue(metadata["validation_warnings"])
            # tiny max_chars splits the article into many chunks and the
            # harness keeps working through them while the goal is unmet
            self.assertGreaterEqual(metadata["session_chunks"], 1)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
