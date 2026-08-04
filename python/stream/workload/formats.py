"""arachne.workload.formats

Binary I/O for arachne.workload: readers for the *source* dataset, and
writers for this package's own pool/groundtruth *output*.

## Source formats (SourceFormat, open_source_reader)

Two source layouts are supported, both real, pre-existing conventions --
neither is invented by this package:

- **SourceFormat.XBIN** (`XBinReader`): the format big-ann-benchmarks'
  own tooling calls "xbin" (see `xbin_mmap`/`u8bin_write` in
  https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/dataset_io.py):

      [uint32 num_vectors][uint32 dim][ raw vector rows, row-major, dtype D ]

  one global header, then the flat vector data. This is exactly the format
  svfusion's own loader already reads (`svfusion/examples/src/datasets.hpp`:
  `Dataset::init_data_stream()`/`read_batch_pos()`), and it's what the
  `base.1B.u8bin`/`query.public.10K.u8bin` files already on disk for this
  project actually are. It is *also* the layout Microsoft's SPACEV1B
  dataset uses for its `vectors.bin`/`query.bin` (int8 dtype) and
  `truth.bin` (matches this module's XBIN groundtruth layout exactly --
  see write_groundtruth) -- SPACEV1B needs no separate reader class, just
  `XBinReader(path, dtype=np.int8)` (see SourceFormat.XBIN docstring).

- **SourceFormat.VECS** (`VecsReader`): the classic INRIA/TEXMEX
  "fvecs/ivecs/bvecs" format from corpus-texmex.irisa.fr (the original
  ANN_SIFT1M/1B release, still used by many ANN benchmark forks). There is
  *no* global header; instead *every single vector* is individually
  prefixed by its own dimension:

      [int32 dim][ dtype D x dim values ]   (repeated num_vectors times)

  `dtype=float32` is "fvecs", `dtype=int32` is "ivecs", `dtype=uint8` is
  "bvecs" -- the layout is identical in all three cases, differing only in
  the value dtype, so one `VecsReader(path, dtype=...)` class covers all
  three (mirroring how one `XBinReader` already covers u8bin/fbin/i8bin).

## Output formats (OutputFormat, PoolWriter, write_groundtruth)

Pool files (base/insert/query) and the groundtruth file can each be
written in one of two OutputFormat encodings:
  - OutputFormat.XBIN: identical header+raw-data layout to the XBIN source
    format above, so svfusion's existing C++ loader can read it back with
    zero code changes. Pool *filenames* use the dtype-specific extension
    big-ann-benchmarks/cuVS's own tooling uses -- not a fixed ".xbin" --
    see `xbin_extension_for_dtype`.
  - OutputFormat.NUMPY: a plain .npy (pools) / .npz (groundtruth) file,
    convenient for inspection or further processing from Python. No JSON
    is used anywhere in this package.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from arachne.workload.logging_utils import get_logger

logger = get_logger(__name__)

# xbin header: two little-endian uint32 fields, (num_vectors, dim) --
# matches big-ann-benchmarks' documented "uint32_t num_points, uint32_t
# num_dimensions" header exactly.
_HEADER_DTYPE: np.dtype = np.dtype("<u4")
_HEADER_NUM_FIELDS: int = 2
_HEADER_SIZE_BYTES: int = _HEADER_DTYPE.itemsize * _HEADER_NUM_FIELDS

# vecs (fvecs/ivecs/bvecs) per-vector dimension prefix: one little-endian
# int32 before every vector's raw values.
_VECS_DIM_PREFIX_DTYPE: np.dtype = np.dtype("<i4")

# groundtruth (XBIN-format) header: two little-endian uint32 fields,
# (num_queries, k). Identical to SPACEV1B's truth.bin and the common
# bigann-benchmark gt100.bin convention.
_GT_HEADER_DTYPE: np.dtype = np.dtype("<u4")
_GT_ID_DTYPE: np.dtype = np.dtype("<i4")
_GT_DIST_DTYPE: np.dtype = np.dtype("<f4")

# id-list (XBIN-format) header: one little-endian uint32 count, followed
# by that many little-endian int32 ids -- used for delete-step files
# (streaming.py): a plain list of node ids, no vector payload.
_ID_LIST_HEADER_DTYPE: np.dtype = np.dtype("<u4")
_ID_LIST_DTYPE: np.dtype = np.dtype("<i4")

# Vector-data filename extension per dtype, exactly matching the
# convention documented at
# https://big-ann-benchmarks.com/neurips21.html and
# https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets
# (e.g. this project's own base.1B.u8bin/query.public.10K.u8bin files) --
# the header+raw-data layout is identical for every dtype; only the
# filename differs, purely so a file self-documents its own dtype.
_XBIN_DTYPE_EXTENSIONS: dict[np.dtype, str] = {
    np.dtype(np.uint8): "u8bin",
    np.dtype(np.int8): "i8bin",
    np.dtype(np.float16): "f16bin",
    np.dtype(np.float32): "fbin",
}


def xbin_extension_for_dtype(dtype: np.dtype) -> str:
    """Big-ann-benchmarks/cuVS-style filename extension (without a
    leading dot) for `dtype`, e.g. np.uint8 -> "u8bin"."""
    resolved = np.dtype(dtype)
    try:
        return _XBIN_DTYPE_EXTENSIONS[resolved]
    except KeyError:
        supported = ", ".join(sorted(d.name for d in _XBIN_DTYPE_EXTENSIONS))
        raise ValueError(
            f"no big-ann-benchmarks/cuVS xbin filename convention for dtype "
            f"{resolved} -- supported dtypes are: {supported}"
        ) from None


@runtime_checkable
class SourceVectorReader(Protocol):
    """Structural interface every source-dataset reader satisfies
    (XBinReader, VecsReader, and any future format added the same way).

    This is a `typing.Protocol`, not a base class: readers do not need to
    inherit from anything to satisfy it (duck typing), which keeps the
    class hierarchy flat per format (no XBinReader -> BaseReader -> ...
    chain) while still giving `clustering.py`/`pool_split.py`/
    `groundtruth.py`/`organizer.py` one concrete type to write against.
    """

    @property
    def num_vectors(self) -> int: ...

    @property
    def dim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def read_range(self, start_idx: int, count: int) -> np.ndarray: ...

    def read_rows(self, row_indices: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...


class SourceFormat(enum.Enum):
    """Which reader `open_source_reader` constructs for the raw source
    dataset. See the module docstring for exactly what each layout is and
    which real-world datasets use it."""

    XBIN = "xbin"
    VECS = "vecs"


def open_source_reader(path: Path, source_format: SourceFormat, dtype: np.dtype) -> SourceVectorReader:
    """Factory used by organizer.py so it never hardcodes a reader class:
    swapping `source_format` is the entire adaptation needed to point this
    package at a differently-laid-out source dataset."""
    if source_format is SourceFormat.XBIN:
        return XBinReader(path, dtype=dtype)
    elif source_format is SourceFormat.VECS:
        return VecsReader(path, dtype=dtype)
    else:
        raise ValueError(f"unhandled SourceFormat: {source_format}")


class OutputFormat(enum.Enum):
    """Serialization format for pool/groundtruth output files.

    XBIN: the same raw binary layout as the source SIFT dataset -- directly
        consumable by svfusion's existing Dataset loader with zero C++
        changes. Use this to feed a benchmark run directly.
    NUMPY: a plain .npy array per pool and an .npz for groundtruth --
        convenient for quick inspection/further processing from Python
        (numpy.load), but not understood by svfusion's C++ loader without a
        separate converter.

    This is deliberately a flat, closed enum: a new output format is added
    as one more enum value plus one more branch in PoolWriter/
    write_groundtruth, not a new class hierarchy.
    """

    XBIN = "xbin"
    NUMPY = "numpy"

    def pool_file_suffix(self, dtype: np.dtype) -> str:
        """Filename suffix for a pool's vector-data file. XBIN uses the
        dtype-specific big-ann-benchmarks/cuVS extension (e.g. ".u8bin"
        for uint8, ".fbin" for float32 -- see xbin_extension_for_dtype),
        matching how real datasets like SIFT1B/SPACEV1B actually name
        their files, rather than one fixed ".xbin" -- the on-disk layout
        itself does not depend on dtype, only the filename does. NUMPY
        is always ".npy" regardless of dtype."""
        if self is OutputFormat.XBIN:
            return f".{xbin_extension_for_dtype(dtype)}"
        return ".npy"

    @property
    def groundtruth_file_suffix(self) -> str:
        return ".bin" if self is OutputFormat.XBIN else ".npz"

    @property
    def id_list_file_suffix(self) -> str:
        return ".ids" if self is OutputFormat.XBIN else ".npy"


@dataclass(frozen=True)
class XBinHeader:
    """Parsed (num_vectors, dim) header of an xbin file."""

    num_vectors: int
    dim: int


class XBinReader:
    """Random-access reader for one xbin file (SourceFormat.XBIN).

    Mirrors svfusion's Dataset::read_batch_pos(): supports reading an
    arbitrary contiguous row range without loading the whole file into
    memory, which matters for base datasets that can be 100GB+ (e.g.
    SIFT1B). Keeps a single open file handle for the reader's lifetime
    instead of reopening per call.

    Also reads Microsoft SPACEV1B's vectors.bin/query.bin as-is: those
    files share this exact [uint32 count][uint32 dim][data] layout, just
    with `dtype=np.int8` (SPACEV1B stores signed-byte vector components).
    SPACEV1B's base set additionally ships pre-split across multiple
    physical files (vectors_1.bin, vectors_2.bin, ...) sharing one logical
    address space; reading that multi-file split is not implemented here
    (only a single file per XBinReader instance) -- concatenate the shards
    into one file first if you need the full 1.4B-vector SPACEV1B base set.
    """

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self._path: Path = Path(path)
        self._dtype: np.dtype = np.dtype(dtype)
        self._file = open(self._path, "rb")
        self._header: XBinHeader = self._read_header(self._file)
        logger.debug(
            "opened xbin source %s: num_vectors=%d dim=%d dtype=%s",
            self._path, self._header.num_vectors, self._header.dim, self._dtype,
        )

    @staticmethod
    def _read_header(file) -> XBinHeader:
        file.seek(0)
        raw = np.fromfile(file, dtype=_HEADER_DTYPE, count=_HEADER_NUM_FIELDS)
        if raw.size != _HEADER_NUM_FIELDS:
            raise ValueError("truncated xbin header")
        return XBinHeader(num_vectors=int(raw[0]), dim=int(raw[1]))

    @property
    def header(self) -> XBinHeader:
        return self._header

    @property
    def num_vectors(self) -> int:
        return self._header.num_vectors

    @property
    def dim(self) -> int:
        return self._header.dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "XBinReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read_range(self, start_idx: int, count: int) -> np.ndarray:
        """Read `count` contiguous vectors starting at row index `start_idx`.

        Returns an (count, dim) array of dtype self.dtype.
        """
        if start_idx < 0 or count < 0 or start_idx + count > self.num_vectors:
            raise IndexError(
                f"requested range [{start_idx}, {start_idx + count}) is out "
                f"of bounds for {self.num_vectors} vectors"
            )
        row_bytes = self.dim * self._dtype.itemsize
        byte_offset = _HEADER_SIZE_BYTES + start_idx * row_bytes
        self._file.seek(byte_offset)
        flat = np.fromfile(self._file, dtype=self._dtype, count=count * self.dim)
        if flat.size != count * self.dim:
            raise IOError(
                f"short read from {self._path}: expected {count * self.dim} "
                f"elements, got {flat.size}"
            )
        return flat.reshape(count, self.dim)

    def read_rows(self, row_indices: np.ndarray) -> np.ndarray:
        """Read a (possibly non-contiguous, unsorted) set of row indices.

        Internally sorts the requested indices so disk seeks are
        monotonically increasing (cheaper for large sequential-ish
        gathers, e.g. a cluster's rows), then restores the caller's
        original order before returning. This does one seek+read per row,
        so it is meant for offline preprocessing (thousands to low
        millions of rows), not a performance-critical hot path.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-D")

        sort_order = np.argsort(row_indices)
        sorted_rows = row_indices[sort_order]

        gathered_sorted = np.empty((row_indices.shape[0], self.dim), dtype=self._dtype)
        for i, row in enumerate(sorted_rows):
            gathered_sorted[i] = self.read_range(int(row), 1)[0]

        gathered = np.empty_like(gathered_sorted)
        gathered[sort_order] = gathered_sorted
        return gathered


class VecsReader:
    """Random-access reader for the TEXMEX/INRIA "vecs" family
    (SourceFormat.VECS): fvecs (dtype=float32), ivecs (dtype=int32), bvecs
    (dtype=uint8) -- see corpus-texmex.irisa.fr, the original
    ANN_SIFT1M/1B release. All three share the same per-vector-prefixed
    layout and differ only in value dtype, so this one class covers all
    three (pass the matching `dtype`).

    Unlike xbin, there is no global header: every vector is individually
    prefixed by its own dimension as a little-endian int32, i.e. the file
    is just a back-to-back concatenation of (dim:int32, values:dtype[dim])
    records. `num_vectors` is therefore not stored anywhere and is derived
    from the file size (file_size // record_size); `dim` is read from the
    first record and, since every real fvecs/ivecs/bvecs file in practice
    uses one fixed dimension throughout, validated (cheaply, in bulk) to
    stay constant across every subsequent record read.
    """

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self._path: Path = Path(path)
        self._dtype: np.dtype = np.dtype(dtype)
        self._file = open(self._path, "rb")

        self._dim: int = self._read_dim_prefix(self._file, byte_offset=0)
        self._record_dtype: np.dtype = np.dtype(
            [("dim", _VECS_DIM_PREFIX_DTYPE), ("values", self._dtype, (self._dim,))]
        )
        record_size_bytes = self._record_dtype.itemsize
        file_size_bytes = self._path.stat().st_size
        if file_size_bytes % record_size_bytes != 0:
            raise ValueError(
                f"{path}: file size {file_size_bytes} is not a multiple of the "
                f"per-vector record size {record_size_bytes} (dim={self._dim}, "
                f"dtype={self._dtype}) -- wrong dtype, or not a vecs-format file?"
            )
        self._num_vectors: int = file_size_bytes // record_size_bytes
        logger.debug(
            "opened vecs source %s: num_vectors=%d dim=%d dtype=%s",
            self._path, self._num_vectors, self._dim, self._dtype,
        )

    @staticmethod
    def _read_dim_prefix(file, byte_offset: int) -> int:
        file.seek(byte_offset)
        raw = np.fromfile(file, dtype=_VECS_DIM_PREFIX_DTYPE, count=1)
        if raw.size != 1:
            raise ValueError("truncated vecs file: missing a vector's dimension prefix")
        return int(raw[0])

    @property
    def num_vectors(self) -> int:
        return self._num_vectors

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "VecsReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read_range(self, start_idx: int, count: int) -> np.ndarray:
        """Read `count` contiguous vectors starting at row index `start_idx`.

        Reads the whole range as one structured-dtype `fromfile` call
        (record = dim-prefix + values), so this is a single I/O call
        regardless of `count` -- not one read per vector -- then validates
        in bulk that every record's own dim prefix still matches the
        dimension detected at open() time.
        """
        if start_idx < 0 or count < 0 or start_idx + count > self.num_vectors:
            raise IndexError(
                f"requested range [{start_idx}, {start_idx + count}) is out "
                f"of bounds for {self.num_vectors} vectors"
            )
        byte_offset = start_idx * self._record_dtype.itemsize
        self._file.seek(byte_offset)
        records = np.fromfile(self._file, dtype=self._record_dtype, count=count)
        if records.shape[0] != count:
            raise IOError(
                f"short read from {self._path}: expected {count} records, got {records.shape[0]}"
            )
        if count > 0 and not np.all(records["dim"] == self._dim):
            raise ValueError(
                f"non-uniform per-vector dim prefix found in range "
                f"[{start_idx}, {start_idx + count}) -- expected {self._dim} "
                f"for every vector (a mixed-dimension vecs file is not supported)"
            )
        # records["values"] is a view into the structured array; copy() so
        # the caller owns plain, independent (count, dim) memory.
        return records["values"].copy()

    def read_rows(self, row_indices: np.ndarray) -> np.ndarray:
        """Read a (possibly non-contiguous, unsorted) set of row indices --
        same sort/gather/restore approach as XBinReader.read_rows (see
        there for the performance caveat: one read_range(row, 1) call per
        requested row)."""
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-D")

        sort_order = np.argsort(row_indices)
        sorted_rows = row_indices[sort_order]

        gathered_sorted = np.empty((row_indices.shape[0], self.dim), dtype=self._dtype)
        for i, row in enumerate(sorted_rows):
            gathered_sorted[i] = self.read_range(int(row), 1)[0]

        gathered = np.empty_like(gathered_sorted)
        gathered[sort_order] = gathered_sorted
        return gathered


class XBinWriter:
    """Streaming writer for the xbin format (backs OutputFormat.XBIN pools).

    `num_vectors` must be known up front, matching the fixed-size header
    the C++ reader expects; `close()` (or the context-manager exit)
    verifies the number of rows actually written matches what was
    declared, to catch pool-construction bugs early rather than silently
    producing a file with a wrong header.
    """

    def __init__(self, path: Path, dim: int, dtype: np.dtype, num_vectors: int) -> None:
        self._path: Path = Path(path)
        self._dim: int = dim
        self._dtype: np.dtype = np.dtype(dtype)
        self._num_vectors_declared: int = num_vectors
        self._num_vectors_written: int = 0
        self._file = open(self._path, "wb")
        header = np.array([num_vectors, dim], dtype=_HEADER_DTYPE)
        header.tofile(self._file)

    def write_rows(self, rows: np.ndarray) -> None:
        if rows.ndim != 2 or rows.shape[1] != self._dim:
            raise ValueError(f"expected rows of shape (*, {self._dim}), got {rows.shape}")
        if rows.dtype != self._dtype:
            rows = rows.astype(self._dtype)
        rows.tofile(self._file)
        self._num_vectors_written += rows.shape[0]

    def close(self) -> None:
        if self._num_vectors_written != self._num_vectors_declared:
            raise ValueError(
                f"declared num_vectors={self._num_vectors_declared} but wrote "
                f"{self._num_vectors_written} rows to {self._path}"
            )
        self._file.close()

    def __enter__(self) -> "XBinWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class PoolWriter:
    """Streaming writer for one pool's vectors, in either OutputFormat.

    Hides the XBIN-vs-NUMPY distinction behind one write_rows()/close()
    interface via composition (delegates to an XBinWriter, or to a numpy
    on-disk memmap for the NUMPY format) rather than subclassing, so
    callers (organizer.py) never need to branch on format themselves.
    Both branches stream in chunks -- neither needs the whole pool
    resident in memory at once, which matters for base pools that can be
    tens of millions of rows.
    """

    def __init__(
        self,
        path: Path,
        output_format: OutputFormat,
        dim: int,
        dtype: np.dtype,
        num_vectors: int,
    ) -> None:
        self._dim: int = dim
        self._dtype: np.dtype = np.dtype(dtype)
        self._num_vectors_declared: int = num_vectors
        self._num_vectors_written: int = 0
        self._xbin_writer: XBinWriter | None = None
        self._numpy_memmap: np.memmap | None = None

        if output_format is OutputFormat.XBIN:
            self._xbin_writer = XBinWriter(path, dim=dim, dtype=self._dtype, num_vectors=num_vectors)
        elif output_format is OutputFormat.NUMPY:
            # open_memmap pre-allocates the full (num_vectors, dim) .npy file
            # on disk and lets us fill it in chunks, mirroring XBinWriter's
            # streaming behavior instead of requiring the whole pool in RAM.
            self._numpy_memmap = np.lib.format.open_memmap(
                path, mode="w+", dtype=self._dtype, shape=(num_vectors, dim)
            )
        else:
            raise ValueError(f"unhandled OutputFormat: {output_format}")

        logger.debug(
            "opened %s pool writer at %s: num_vectors=%d dim=%d dtype=%s",
            output_format.value, path, num_vectors, dim, self._dtype,
        )

    def write_rows(self, rows: np.ndarray) -> None:
        if rows.ndim != 2 or rows.shape[1] != self._dim:
            raise ValueError(f"expected rows of shape (*, {self._dim}), got {rows.shape}")
        if rows.dtype != self._dtype:
            rows = rows.astype(self._dtype)

        if self._xbin_writer is not None:
            self._xbin_writer.write_rows(rows)
        else:
            start = self._num_vectors_written
            self._numpy_memmap[start : start + rows.shape[0]] = rows

        self._num_vectors_written += rows.shape[0]

    def close(self) -> None:
        if self._num_vectors_written != self._num_vectors_declared:
            raise ValueError(
                f"declared num_vectors={self._num_vectors_declared} but wrote "
                f"{self._num_vectors_written} rows"
            )
        if self._xbin_writer is not None:
            self._xbin_writer.close()
        else:
            self._numpy_memmap.flush()

    def __enter__(self) -> "PoolWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_groundtruth(
    path: Path,
    neighbor_ids: np.ndarray,
    neighbor_dists: np.ndarray,
    output_format: OutputFormat,
) -> None:
    """Writes a ground-truth file in the requested OutputFormat.

    OutputFormat.XBIN: (uint32 num_queries, uint32 k) header -- matching
        the header svfusion's Dataset::get_groundtruth() already reads
        (datasets.hpp) -- followed by the full (num_queries, k) int32
        neighbor-id matrix and then the (num_queries, k) float32 distance
        matrix. This body layout is not our own invention: it matches both
        the common bigann-benchmark gt100.bin convention and Microsoft
        SPACEV1B's truth.bin exactly, so a future svfusion reader (or any
        other tool expecting either convention) can consume it directly.
    OutputFormat.NUMPY: a single .npz archive with "neighbor_ids" and
        "neighbor_dists" arrays.
    """
    if neighbor_ids.shape != neighbor_dists.shape:
        raise ValueError("neighbor_ids and neighbor_dists must have the same shape")

    if output_format is OutputFormat.XBIN:
        num_queries, k = neighbor_ids.shape
        with open(path, "wb") as f:
            np.array([num_queries, k], dtype=_GT_HEADER_DTYPE).tofile(f)
            neighbor_ids.astype(_GT_ID_DTYPE).tofile(f)
            neighbor_dists.astype(_GT_DIST_DTYPE).tofile(f)
    elif output_format is OutputFormat.NUMPY:
        with open(path, "wb") as f:
            np.savez(
                f,
                neighbor_ids=neighbor_ids.astype(_GT_ID_DTYPE),
                neighbor_dists=neighbor_dists.astype(_GT_DIST_DTYPE),
            )
    else:
        raise ValueError(f"unhandled OutputFormat: {output_format}")

    logger.debug("wrote %s groundtruth to %s", output_format.value, path)


def write_id_list(path: Path, ids: np.ndarray, output_format: OutputFormat) -> None:
    """Writes a plain 1-D array of node ids -- no vector payload -- used
    for a streaming run's per-step delete files (streaming.py): which
    insert-stream ids get evicted at a given step, referenced by id only
    (a consumer already received the vectors themselves when they were
    inserted at an earlier step).

    OutputFormat.XBIN: (uint32 count) header followed by that many int32
        ids -- the same convention as this module's other XBIN formats,
        just without a `dim` field (ids have no vector payload).
    OutputFormat.NUMPY: a plain int32 .npy array.
    """
    if output_format is OutputFormat.XBIN:
        with open(path, "wb") as f:
            np.array([ids.shape[0]], dtype=_ID_LIST_HEADER_DTYPE).tofile(f)
            ids.astype(_ID_LIST_DTYPE).tofile(f)
    elif output_format is OutputFormat.NUMPY:
        np.save(path, ids.astype(_ID_LIST_DTYPE))
    else:
        raise ValueError(f"unhandled OutputFormat: {output_format}")

    logger.debug("wrote %s id list (%d ids) to %s", output_format.value, ids.shape[0], path)


def read_id_list(path: Path, output_format: OutputFormat) -> np.ndarray:
    """Reads back a file written by write_id_list()."""
    if output_format is OutputFormat.XBIN:
        with open(path, "rb") as f:
            count = int(np.fromfile(f, dtype=_ID_LIST_HEADER_DTYPE, count=1)[0])
            ids = np.fromfile(f, dtype=_ID_LIST_DTYPE, count=count)
        if ids.shape[0] != count:
            raise IOError(f"short read from id-list file: {path}")
        return ids
    elif output_format is OutputFormat.NUMPY:
        loaded = np.load(path)
        return loaded.astype(_ID_LIST_DTYPE)
    else:
        raise ValueError(f"unhandled OutputFormat: {output_format}")
