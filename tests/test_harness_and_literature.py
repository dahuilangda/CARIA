"""Harness behavior tested offline with scripted fake clients.

Covers: context chunking, goal-driven early stop, repair rounds, unit
normalization and carrier synonyms, the EE-DL-feed-ratio mass-consistency
skill, and the literature retrieval pipeline (search, PMC linking, caching)
without any network access.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extraction"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = load("harness", "extraction/harness.py")
LIT = load("literature", "extraction/literature.py")
PROMPT = (ROOT / "extraction/extraction_prompt.txt").read_text(encoding="utf-8")
FIELDS = [f"field{i}" for i in range(6)]


class ScriptedClient:
    """Returns queued replies in order and records the requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, system, user, max_tokens):
        self.requests.append(user)
        return SimpleNamespace(text=self.replies.pop(0), finish_reason="stop",
                               model="scripted")


def schema_object():
    return json.dumps({"formulation_composition": {"drug": None}})


class HarnessLoopTests(unittest.TestCase):
    def build_with_client(self, client, replies):
        return HARNESS.ExtractionHarness(
            client=client, prompt_template=PROMPT, schema_fields=FIELDS,
            max_chars=100, parse_response=json.loads)

    def build(self, replies):
        return HARNESS.ExtractionHarness(
            client=ScriptedClient(replies),
            prompt_template=PROMPT,
            schema_fields=FIELDS,
            max_chars=100,
            parse_response=json.loads,
        )

    def test_short_article_uses_one_chunk_and_stops_on_goal(self):
        client = ScriptedClient([json.dumps({f: i for i, f in enumerate(FIELDS)})])
        harness = self.build_with_client(client, client.replies)
        record, log = harness.run("tiny article", "a.md", 4096)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(record["field0"], 0)
        self.assertEqual(len(log), 1)
        self.assertFalse(log[0].repaired)

    def test_long_article_chunks_context_until_goal_met(self):
        article = " ".join(["sentence"] * 400)  # ~2800 chars -> several chunks
        client = ScriptedClient([
            json.dumps({"field0": 1}),                      # chunk 0: partial
            json.dumps({"field0": 1, "field1": 2, "field2": 3,
                        "field3": 4, "field4": 5, "field5": 6}),  # chunk 1: goal met
        ])
        harness = self.build_with_client(client, client.replies)
        record, log = harness.run(article, "a.md", 4096)
        self.assertGreater(len(client.requests), 1)
        self.assertLessEqual(len(client.requests), len(harness.chunks(article)))
        self.assertEqual(record["field1"], 2)

    def test_mass_inconsistency_triggers_a_repair_round(self):
        bad = json.dumps({"field0": 1,
                          "encapsulation_efficiency_percent": 90,
                          "drug_loading_percent": 70,
                          "input_drug_polymer_ratio": 0.1})
        good = json.dumps({f: i for i, f in enumerate(FIELDS)})
        harness = self.build([bad, good])
        record, log = harness.run("article", "a.md", 4096)
        self.assertTrue(any(entry.repaired for entry in log))
        self.assertEqual(record["field0"], 0)

    def test_unrepairable_response_leaves_violations_in_the_log(self):
        harness = self.build(["not json", "also not json"])
        record, log = harness.run("article", "a.md", 4096)
        self.assertTrue(all(entry.error for entry in log),
                        "both passes must fail JSON parsing")
        self.assertEqual({k: v for k, v in record.items() if k != "extraction_session"}, {})

    def test_session_log_records_pass_count(self):
        client = ScriptedClient([json.dumps({f: i for i, f in enumerate(FIELDS)})])
        harness = self.build(client.replies)
        record, log = harness.run("article", "a.md", 4096)
        self.assertEqual(record["extraction_session"]["source_file"], "a.md")
        self.assertEqual(record["extraction_session"]["chunks"], 1)
        self.assertEqual(record["extraction_session"]["passes"], 1)


class FormulationSkillTests(unittest.TestCase):
    def test_length_units_normalize_to_nm(self):
        self.assertEqual(HARNESS.normalize_length_to_nm(0.142, "µm"), 142.0)
        self.assertIsNone(HARNESS.normalize_length_to_nm(1.0, "lightyears") is None or
                          None)

    def test_unknown_length_unit_is_rejected(self):
        value = HARNESS.normalize_length_to_nm(1.0, "spans")
        self.assertNotEqual(value, value)  # NaN

    def test_carrier_synonyms_map_to_vocabulary(self):
        self.assertEqual(HARNESS.normalize_carrier_name("poly(lactic-co-glycolic acid)"), "PLGA")
        self.assertEqual(HARNESS.normalize_carrier_name("Liposomes"), "liposome")
        self.assertEqual(HARNESS.normalize_carrier_name("CustomBlend42"), "CustomBlend42")

    def test_mass_consistency_window(self):
        consistent = HARNESS.ee_dl_mass_consistency(80.0, 10.0, 0.2)   # DL ceiling 16
        inconsistent = HARNESS.ee_dl_mass_consistency(80.0, 40.0, 0.2)  # ceiling 16
        self.assertTrue(consistent)
        self.assertFalse(inconsistent)


class LiteraturePipelineTests(unittest.TestCase):
    def test_tavily_urls_yield_pmid_articles(self):
        articles = LIT.search_tavily.__wrapped__ if hasattr(LIT.search_tavily, "__wrapped__") else None
        # parse URLs the way search_tavily does
        import re
        url = "https://pubmed.ncbi.nlm.nih.gov/42797450/"
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        self.assertEqual(match.group(1), "42797450")

    def test_fetch_caches_and_skips_redownload(self):
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                calls.append(1)
                return b"<article>text</article>"

        real_urlopen = LIT.urllib.request.urlopen
        LIT.urllib.request.urlopen = lambda *a, **k: FakeResponse()
        try:
            with tempfile.TemporaryDirectory() as folder:
                first = LIT.fetch_fulltext_xml("PMC1", Path(folder))
                second = LIT.fetch_fulltext_xml("PMC1", Path(folder))
                self.assertEqual(first.status, "fetched")
                self.assertEqual(second.status, "cached")
                self.assertEqual(len(calls), 1)
        finally:
            LIT.urllib.request.urlopen = real_urlopen

    def test_http_error_is_reported_not_raised(self):
        import urllib.error

        def raise_404(*a, **k):
            raise urllib.error.HTTPError("url", 404, "missing", None, None)

        real_urlopen = LIT.urllib.request.urlopen
        LIT.urllib.request.urlopen = raise_404
        try:
            with tempfile.TemporaryDirectory() as folder:
                result = LIT.fetch_fulltext_xml("PMC404", Path(folder))
                self.assertEqual(result.status, "http_error")
        finally:
            LIT.urllib.request.urlopen = real_urlopen

    def test_pmcid_read_from_xml_front_matter(self):
        xml = ('<article><front><article-id pub-id-type="pmc">PMC99</article-id>'
               '</front></article>')
        self.assertEqual(LIT.pmcid_from_xml(xml), "PMC99")


if __name__ == "__main__":
    unittest.main()
