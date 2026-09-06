"""Prepare voltage traces for the Fused simulator."""

from __future__ import annotations

import math
from pathlib import Path


def prepare_fused_trace(trace: str, fused_dir: str | Path = "fusedBin") -> Path:
    """Resolve and normalize a single-column voltage trace for Fused.

    Fused parses every input line with ``std::stod``. In particular, it cannot
    parse a UTF-8 byte-order mark (BOM) on the first sample. The bundled trace
    files may contain that marker, so write a normalized copy beside the Fused
    binary and return its path. Relative input paths are accepted both from the
    CHAI repository root and from ``fusedBin`` for backwards compatibility.
    """
    fused_path = Path(fused_dir).resolve()
    requested = Path(trace).expanduser()
    candidates = [requested] if requested.is_absolute() else [requested, fused_path / requested]

    resolved_candidates = [path.resolve() for path in candidates]
    source = next((path for path in resolved_candidates if path.is_file()), None)
    if source is None:
        checked = ", ".join(str(path) for path in resolved_candidates)
        raise FileNotFoundError(f"Voltage trace '{trace}' was not found. Checked: {checked}")

    samples: list[str] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        sample = raw_line.strip()
        if not sample:
            continue
        try:
            value = float(sample)
        except ValueError as error:
            raise ValueError(
                f"Fused voltage trace {source}:{line_number} must contain one numeric sample per line."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Fused voltage trace {source}:{line_number} contains a non-finite sample.")
        samples.append(sample)

    if not samples:
        raise ValueError(f"Fused voltage trace {source} contains no numeric samples.")

    fused_path.mkdir(parents=True, exist_ok=True)
    normalized_path = fused_path / "trace_input.csv"
    normalized_path.write_text("\n".join(samples) + "\n", encoding="ascii")
    return normalized_path

