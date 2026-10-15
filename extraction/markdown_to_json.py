"""Extract literature records and flag structural issues for source review."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from harness import ExtractionHarness
from pathlib import Path


def validate_extracted_data(json_data):
    from pathlib import Path

    schema = json.loads(
        Path(__file__).with_name("extraction_schema.json").read_text(encoding="utf-8")
    )
    if not isinstance(json_data, dict):
        raise ValueError("Extraction must be a JSON object")
    warnings = []
    filled_fields = 0
    total_fields = 0

    def present(value):
        return value is not None and str(value).strip().lower() not in (
            "",
            "unknown",
            "null",
            "none",
            "n/a",
        )

    def validate(value, spec, path="", exists=True):
        nonlocal filled_fields, total_fields
        kind = spec["type"]
        if kind == "object":
            obj = value if isinstance(value, dict) else {}
            if not isinstance(value, dict):
                warnings.append(f"Missing or invalid object at {path or '$'}")
            for key in set(obj) - set(spec["properties"]):
                warnings.append(f"Unexpected field at {path}.{key}")
            for key, child in spec["properties"].items():
                if key not in obj:
                    warnings.append(f"Missing required field at {path}.{key}")
                validate(obj.get(key), child, f"{path}.{key}", key in obj)
        elif kind == "array":
            if not isinstance(value, list):
                warnings.append(f"Missing or invalid array at {path}")
            items = value if isinstance(value, list) and value else [None]
            for i, item in enumerate(items):
                validate(item, spec["items"], f"{path}[{i}]", item is not None)
        else:
            total_fields += 1
            valid = True
            if value is not None and "number" in kind:
                valid = (
                    isinstance(value, (int, float))
                    and (not isinstance(value, bool))
                    and math.isfinite(value)
                )
                if valid:
                    valid = value >= spec.get(
                        "minimum", -math.inf
                    ) and value <= spec.get("maximum", math.inf)
            elif value is not None and "string" in kind:
                valid = isinstance(value, str)
            if not valid:
                warnings.append(f"Invalid value/type at {path}")
            if exists and valid and present(value):
                filled_fields += 1
            if (
                path.endswith("zeta_potential_mv")
                and valid
                and (value is not None)
                and (abs(value) > 300)
            ):
                warnings.append(
                    f"Unusual zeta potential at {path}; check units and original source"
                )

    validate(json_data, schema)
    completeness = 100 * filled_fields / total_fields if total_fields else 0
    return {
        "quality_score": round(max(0, min(100 - 5 * len(warnings), completeness)), 1),
        "completeness_percent": round(completeness, 1),
        "warnings": warnings,
        "total_fields": total_fields,
        "filled_fields": filled_fields,
    }


def parse_response(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON response fence")
        text = "\n".join(lines[1:-1])

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    record = json.loads(
        text,
        parse_float=finite_float,
        parse_constant=reject_constant,
        object_pairs_hook=unique_fields,
    )
    if not isinstance(record, dict):
        raise ValueError("Extraction must return a JSON object")
    return record


def extract_file(
    source: Path,
    output: Path,
    client,
    model: str,
    review_threshold: float,
    max_chars: int,
    max_tokens: int,
) -> dict:
    """Extract one article through the harness and write the validated record."""
    article = source.read_text(encoding="utf-8")
    if not article.strip():
        raise ValueError("Article text is empty")
    prompt_template = (
        Path(__file__).with_name("extraction_prompt.txt").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (Path(__file__).with_name("extraction_schema.json")).read_text(encoding="utf-8")
    )

    def leaf_fields(properties, prefix=""):
        for name, spec in properties.items():
            path = f"{prefix}.{name}" if prefix else name
            if spec.get("type") == "object" and "properties" in spec:
                yield from leaf_fields(spec["properties"], path)
            else:
                yield path

    goal_fields = list(leaf_fields(schema["properties"]))
    harness = ExtractionHarness(
        client=client,
        prompt_template=prompt_template,
        schema_fields=goal_fields,
        max_chars=max_chars,
        parse_response=parse_response,
    )
    record, session_log = harness.run(article, source.name, max_tokens)
    validation = validate_extracted_data(record)
    last = session_log[-1] if session_log else None
    record["extraction_metadata"] = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": model,
        "source_file": source.name,
        "original_length_chars": len(article),
        "session_chunks": len({e.chunk_index for e in session_log}),
        "repair_rounds": sum(1 for e in session_log if e.repaired),
        "session_violations": last.violations if last else [],
        "data_quality_score": validation["quality_score"],
        "completeness_percent": validation["completeness_percent"],
        "review_threshold": review_threshold,
        "validation_warnings": validation["warnings"],
        "manual_review_required": bool(validation["warnings"])
        or validation["quality_score"] < review_threshold,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return record["extraction_metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--markdown-file", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("harness", "openai"),
        default=os.environ.get("EXTRACTION_BACKEND", "harness"),
        help=(
            "harness: Anthropic-compatible /v1/messages endpoint "
            "(EXTRACTION_BASE_URL / EXTRACTION_API_KEY / EXTRACTION_MODEL); "
            "openai: OpenAI-compatible endpoint (OPENAI_BASE_URL / OPENAI_API_KEY)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Declared model identifier; defaults depend on --backend",
    )
    parser.add_argument("--review-threshold", type=float, default=50.0)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if (
        not 0 <= args.review_threshold <= 100
        or min(args.max_chars, args.max_tokens) <= 0
    ):
        parser.error("Invalid review threshold or text/token limits")
    files = (
        [args.markdown_file]
        if args.markdown_file
        else sorted(args.input_dir.glob("*.md"))
    )
    if not files or any(not item.is_file() for item in files):
        parser.error("No readable Markdown input files")
    from llm_client import resolve_client

    try:
        client = resolve_client(args.backend, args.model)
    except KeyError as missing:
        parser.error(
            f"Set {missing.args[0]} for the {args.backend} extraction backend"
        )
    except ValueError as error:
        parser.error(str(error))
    declared_model = client.model
    failed = 0
    for item in files:
        output = args.output_dir / (item.stem + ".json")
        if args.skip_existing and output.exists():
            continue
        try:
            metadata = extract_file(
                item,
                output,
                client,
                declared_model,
                args.review_threshold,
                args.max_chars,
                args.max_tokens,
            )
            print(f"{item.name}: review_required={metadata['manual_review_required']}")
        except Exception as error:
            failed += 1
            print(f"{item.name}: {type(error).__name__}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
