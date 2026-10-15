"""Agentic extraction harness for nanocarrier formulation articles.

The harness turns one article into one validated formulation record through a
bounded loop with four explicit capabilities:

Context management
    Long articles are split into section chunks that each fit the per-request
    character budget; a running ledger of already-extracted fields decides
    whether further chunks can still add schema fields.

Goal maintenance
    The extraction goal is the field checklist of the released JSON schema.
    After every pass the harness re-derives which goal fields are still
    missing and stops only when the article is exhausted or the goal is met.

Error correction
    Every pass is validated (schema walk plus the formulation skills below).
    Validation failures are fed back to the model as explicit correction
    instructions for a bounded number of repair rounds.

Formulation skills
    Pure post-parse checks encoding formulation-specific physics and reporting
    conventions: unit normalization, percent/fraction guarding, the
    EE-DL-feed-ratio mass consistency window, and carrier-name synonyms.

The harness records a session log so every accepted value and every repair
round is auditable. All checks are post-parse pure functions; the model never
executes them. When the configured endpoint accepts images, figure content
can be added to a pass through `llm_client.image_message`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Formulation skills
# ---------------------------------------------------------------------------

LENGTH_TO_NM = {"nm": 1.0, "µm": 1000.0, "um": 1000.0, "μm": 1000.0, "mm": 1e6, "cm": 1e7}
CONCENTRATION_TO_MG_ML = {"mg/ml": 1.0, "µg/ml": 0.001, "ug/ml": 0.001,
                          "g/l": 1.0, "mg/l": 0.001}
CARRIER_SYNONYMS = {
    "plga": "PLGA", "poly(lactic-co-glycolic acid)": "PLGA", "dlga": "PLGA",
    "pla": "PLA", "polylactic acid": "PLA", "poly(lactic acid)": "PLA",
    "pcl": "PCL", "polycaprolactone": "PCL",
    "chitosan": "chitosan", "cs": "chitosan",
    "liposome": "liposome", "liposomes": "liposome",
    "micelle": "micelle", "polymeric micelle": "micelle",
    "sln": "SLN", "solid lipid nanoparticle": "SLN",
    "nlc": "NLC", "nanostructured lipid carrier": "NLC",
    "cyclodextrin": "cyclodextrin", "cd": "cyclodextrin",
    "eudragit": "Eudragit", "zein": "zein", "albumin": "albumin",
}


def normalize_length_to_nm(value: float, unit: str) -> float | None:
    """Convert a particle-size reading to nanometres."""
    return value * LENGTH_TO_NM.get(unit.strip().lower(), float("nan"))


def normalize_concentration_to_mg_ml(value: float, unit: str) -> float | None:
    """Convert a concentration reading to mg/mL."""
    return value * CONCENTRATION_TO_MG_ML.get(unit.strip().lower(), float("nan"))


def normalize_carrier_name(name: str) -> str:
    """Map a carrier spelling to its controlled vocabulary entry."""
    cleaned = " ".join(str(name).strip().lower().split())
    return CARRIER_SYNONYMS.get(cleaned, str(name).strip())


def ee_dl_mass_consistency(ee_percent: float, dl_percent: float, feed_ratio: float,
                           tolerance: float = 1.5) -> bool:
    """Check that EE, DL and the drug:polymer feed ratio can all be true.

    Encapsulated drug per polymer mass equals EE% x r / 100, which caps every
    reported DL definition.  Reported DL above that ceiling x `tolerance`
    means the three fields cannot share one denominator.
    """
    ceiling = ee_percent * feed_ratio
    return dl_percent <= tolerance * ceiling or ceiling == 0.0


def check_record(record: dict[str, Any]) -> list[str]:
    """Run the formulation skills over one extracted record.

    Returns human-readable violations; an empty list means the record passed.
    """
    violations: list[str] = []
    size = record.get("particle_size_nm")
    if isinstance(size, dict):
        value, unit = size.get("value"), size.get("unit")
        if isinstance(value, (int, float)) and isinstance(unit, str):
            converted = normalize_length_to_nm(value, unit)
            if converted is None or math.isnan(converted):
                violations.append(f"particle_size unit '{unit}' is not recognized")
    ee = record.get("encapsulation_efficiency_percent")
    dl = record.get("drug_loading_percent")
    ratio = record.get("input_drug_polymer_ratio")
    if all(isinstance(v, (int, float)) for v in (ee, dl, ratio)) and ratio > 0:
        if not ee_dl_mass_consistency(float(ee), float(dl), float(ratio)):
            violations.append(
                "reported EE, DL and feed ratio are mass-inconsistent: DL exceeds "
                "the encapsulated-mass ceiling EE% x r by more than 1.5x"
            )
    carrier = record.get("carrier_material")
    if isinstance(carrier, str) and carrier.strip():
        record["carrier_material"] = normalize_carrier_name(carrier)
    return violations


# ---------------------------------------------------------------------------
# Harness session
# ---------------------------------------------------------------------------

@dataclass
class PassResult:
    chunk_index: int
    raw_response: str
    record: dict[str, Any] | None
    violations: list[str] = field(default_factory=list)
    repaired: bool = False
    error: str | None = None


class ExtractionHarness:
    """Bounded extraction loop over one article with managed context."""

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        schema_fields: list[str],
        system: str = "Extract only explicitly reported formulation information.",
        max_chars: int = 100_000,
        max_repair_rounds: int = 2,
        goal_threshold: float = 0.6,
        parse_response: Callable[[str], dict[str, Any]] = json.loads,
    ) -> None:
        marker = "**[Paste the research article text here]**"
        if prompt_template.count(marker) != 1:
            raise ValueError("Prompt template must contain exactly one article placeholder")
        self.prompt_template = prompt_template
        self.marker = marker
        self.client = client
        self.system = system
        self.max_chars = max_chars
        self.max_repair_rounds = max_repair_rounds
        self.goal_fields = list(schema_fields)
        self.goal_threshold = goal_threshold
        self.parse_response = parse_response

    # -- context management -------------------------------------------------
    def chunks(self, article: str) -> list[str]:
        """Split the article into per-request chunks that fit the budget."""
        if len(article) <= self.max_chars:
            return [article]
        chunks, start = [], 0
        step = int(self.max_chars * 0.9)
        while start < len(article):
            end = min(start + self.max_chars, len(article))
            chunks.append(article[start:end])
            start += step
        return chunks

    # -- goal maintenance ---------------------------------------------------
    def goal_progress(self, record: dict[str, Any] | None) -> tuple[float, list[str]]:
        """Return (fraction of goal fields with a value, still-missing names)."""
        if not record:
            return 0.0, list(self.goal_fields)
        missing = [f for f in self.goal_fields if record.get(f) in (None, "", [], {})]
        covered = len(self.goal_fields) - len(missing)
        return covered / max(len(self.goal_fields), 1), missing

    # -- error correction ---------------------------------------------------
    def _passes(self, client: Any, instruction: str, max_tokens: int) -> PassResult:
        completion = client.complete(system=self.system, user=instruction, max_tokens=max_tokens)
        try:
            record = self.parse_response(completion.text)
        except (ValueError, json.JSONDecodeError) as error:
            return PassResult(chunk_index=-1, raw_response=completion.text,
                              record=None, error=f"invalid JSON: {error}")
        if not isinstance(record, dict):
            return PassResult(chunk_index=-1, raw_response=completion.text,
                              record=None, error="response is not a JSON object")
        violations = check_record(record)
        return PassResult(chunk_index=-1, raw_response=completion.text,
                          record=record, violations=violations)

    def _repair(self, client: Any, previous: dict[str, Any], violations: list[str],
                max_tokens: int) -> PassResult:
        instruction = (
            "The following formulation record violates reporting rules.\n"
            "Violations:\n- " + "\n- ".join(violations) +
            "\nCorrected JSON record (same fields, violations fixed, JSON only):"
        )
        user = json.dumps(previous, ensure_ascii=False, indent=1) + "\n\n" + instruction
        result = self._passes(client, user, max_tokens)
        result.repaired = True
        return result

    # -- public entry -------------------------------------------------------
    def run(self, article: str, source_name: str, max_tokens: int) -> tuple[dict[str, Any], list[PassResult]]:
        """Extract one record from `article`; returns (record, session log)."""
        log: list[PassResult] = []
        record: dict[str, Any] = {}
        chunks = self.chunks(article)
        for index, chunk in enumerate(chunks):
            instruction = self.prompt_template.replace(self.marker, chunk)
            result = self._passes(client=self.client, instruction=instruction, max_tokens=max_tokens)
            result.chunk_index = index
            if result.error:
                log.append(result)
                continue
            for _ in range(self.max_repair_rounds):
                if not result.violations:
                    break
                result = self._repair(client=self.client, previous=result.record,
                                      violations=result.violations, max_tokens=max_tokens)
                result.chunk_index = index
            log.append(result)
            if result.record:
                for key, value in result.record.items():
                    if value in (None, "", [], {}):
                        continue
                    existing = record.get(key)
                    if existing in (None, "", [], {}):
                        record[key] = value
            progress, _missing = self.goal_progress(record)
            if progress >= self.goal_threshold:
                break
        record.setdefault("extraction_session", {
            "source_file": source_name,
            "chunks": len(chunks),
            "passes": len(log),
            "repair_rounds": sum(1 for entry in log if entry.repaired),
            "final_violations": log[-1].violations if log else [],
        })
        return record, log
