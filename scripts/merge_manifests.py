#!/usr/bin/env python3
"""Merge JSONL VAD manifests while preserving relative file references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvad.manifest import rebase_manifest_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for raw_input in args.input:
        manifest = raw_input.resolve()
        count = 0
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {manifest} line {line_number}"
                    ) from exc
                records.append(
                    rebase_manifest_record(record, manifest.parent, output.parent)
                )
                count += 1
        counts[str(manifest)] = count

    output.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "items": len(records), "inputs": counts}))


if __name__ == "__main__":
    main()
