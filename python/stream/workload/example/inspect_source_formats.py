#!/usr/bin/env python3
"""Correctness demo for the two SourceFormat readers in arachne.workload.

Builds tiny synthetic files in each supported source layout *by hand*
(i.e. without using arachne's own writers, so this is an independent
check of the readers against the documented byte layouts) and confirms
`open_source_reader` reads back exactly what was written:

  1. SourceFormat.VECS  -- the classic INRIA/TEXMEX fvecs layout (no global
     header; every vector individually prefixed by its own int32 dim).
  2. SourceFormat.XBIN with dtype=int8 -- Microsoft SPACEV1B's vectors.bin
     layout (same global-header layout as the SIFT1B xbin files already
     used elsewhere in this project, just a signed-byte dtype).

See python/arachne/README.md ("Input file format") for where these
layouts come from.

Run with:
    PYTHONPATH=python python3 python/arachne/workload/example/inspect_source_formats.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arachne.workload import SourceFormat, open_source_reader

OUTPUT_DIR = Path(__file__).parent / "output"


def write_fvecs_by_hand(path: Path, vectors: np.ndarray) -> None:
    """Writes `vectors` (float32) in fvecs layout: for every row, a little-
    endian int32 dim prefix followed by that row's `dim` float32 values --
    written record-by-record, deliberately not reusing any arachne code."""
    with open(path, "wb") as f:
        for row in vectors:
            np.array([row.shape[0]], dtype="<i4").tofile(f)
            row.astype("<f4").tofile(f)


def write_spacev_style_by_hand(path: Path, vectors: np.ndarray) -> None:
    """Writes `vectors` (int8) in SPACEV1B's vectors.bin layout: one
    (uint32 count, uint32 dim) header, then the raw int8 data -- the exact
    same global-header layout as the project's own SIFT1B xbin files."""
    num_vectors, dim = vectors.shape
    with open(path, "wb") as f:
        np.array([num_vectors, dim], dtype="<u4").tofile(f)
        vectors.astype("<i1").tofile(f)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)

    # --- 1) fvecs (SourceFormat.VECS) -------------------------------------
    fvecs_path = OUTPUT_DIR / "demo.fvecs"
    fvecs_vectors = rng.uniform(-1, 1, size=(200, 16)).astype(np.float32)
    write_fvecs_by_hand(fvecs_path, fvecs_vectors)
    print(f"[demo] wrote hand-rolled fvecs file: {fvecs_path} (200 x 16, float32)")

    with open_source_reader(fvecs_path, SourceFormat.VECS, dtype=np.float32) as reader:
        assert reader.num_vectors == 200, reader.num_vectors
        assert reader.dim == 16, reader.dim
        # spot-check a contiguous range and a scattered gather
        contiguous = reader.read_range(50, 10)
        assert np.array_equal(contiguous, fvecs_vectors[50:60])
        scattered_idx = np.array([199, 0, 87, 42])
        scattered = reader.read_rows(scattered_idx)
        assert np.array_equal(scattered, fvecs_vectors[scattered_idx])
    print("[demo] VecsReader (fvecs) matches the hand-written reference exactly")

    # --- 2) SPACEV-style xbin, dtype=int8 (SourceFormat.XBIN) -------------
    spacev_path = OUTPUT_DIR / "demo_spacev_style.bin"
    spacev_vectors = rng.integers(-128, 127, size=(300, 100), dtype=np.int8)
    write_spacev_style_by_hand(spacev_path, spacev_vectors)
    print(f"[demo] wrote hand-rolled SPACEV-style file: {spacev_path} (300 x 100, int8)")

    with open_source_reader(spacev_path, SourceFormat.XBIN, dtype=np.int8) as reader:
        assert reader.num_vectors == 300, reader.num_vectors
        assert reader.dim == 100, reader.dim
        contiguous = reader.read_range(0, 300)
        assert np.array_equal(contiguous, spacev_vectors)
    print("[demo] XBinReader(dtype=int8) matches the hand-written SPACEV-style reference exactly")
    print(
        "[demo] -> SPACEV1B needs no separate reader class: it is the same "
        "xbin layout already used for SIFT1B, just with dtype=int8."
    )

    print(f"\n[demo] all checks passed. Files left in {OUTPUT_DIR} for inspection.")


if __name__ == "__main__":
    main()
