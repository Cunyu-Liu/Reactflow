"""Pure-standard-library NPY/NPZ codec for frozen-feature interchange.

Cycle C5 exports 100M-parameter encoder representations *offline* (see
``scripts/export_frozen_features.py``) and consumes them from the pure
standard-library ``reactflow`` core.  The interchange format therefore has two
hard requirements:

1. **NumPy interoperable.**  The offline exporter usually runs in a PyTorch
   environment where NumPy is available, so the on-disk bytes must be exactly
   the ``.npy`` / ``.npz`` layout that ``numpy.load`` / ``numpy.save`` use.  A
   human can then inspect a shard with NumPy without any ReactFlow code.
2. **Zero third-party dependencies to *read*.**  The training side is audited
   to import only the standard library, so this module parses and emits the
   NPY format with :mod:`struct`, :mod:`array`, :mod:`zipfile`, :mod:`ast` and
   nothing else.

Format reference
----------------
The NPY specification (NumPy Enhancement Proposal, format versions 1.0 and
2.0) lays out a file as

    magic := b"\\x93NUMPY"
    version := (major: u8, minor: u8)
    header_len := u16 little-endian        # v1.0
                | u32 little-endian        # v2.0
    header := ASCII dict literal, space padded, "\\n" terminated so that
              ``len(magic) + 2 + len(header_len_field) + len(header)`` is a
              multiple of 64
    data := C-contiguous little-endian raw bytes

The header dict has keys ``descr`` (a NumPy dtype string such as ``'<f4'``),
``fortran_order`` (always ``False`` here) and ``shape`` (a tuple).  ``.npz`` is
simply a ZIP archive whose members are ``<name>.npy`` files.

Determinism
-----------
Every array is written little-endian regardless of host byte order, so shard
bytes are bit-for-bit identical across platforms -- the same reproducibility
contract the rest of ReactFlow holds.  On a big-endian host the codec byte
swaps in memory before writing and after reading; on the overwhelmingly common
little-endian host both paths are no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass
import array
import ast
import struct
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

_MAGIC = b"\x93NUMPY"
_ALIGN = 64

# Mapping from a normalized little-endian NumPy dtype string to the matching
# :mod:`array` type code.  Only the fixed-width numeric kinds ReactFlow needs
# are supported; anything else raises so a silent misread is impossible.
_DESCR_TO_TYPECODE: Dict[str, str] = {
    "<f4": "f",
    "<f8": "d",
    "<i1": "b",
    "|i1": "b",
    "<i2": "h",
    "<i4": "i",
    "<i8": "q",
    "|u1": "B",
    "<u1": "B",
    "<u2": "H",
    "<u4": "I",
    "<u8": "Q",
    "|b1": "b",
}

# Preferred dtype string for each Python-facing element kind used when writing.
_KIND_TO_DESCR: Dict[str, str] = {
    "float32": "<f4",
    "float64": "<f8",
    "int32": "<i4",
    "int64": "<i8",
    "bool": "|b1",
    "uint8": "|u1",
}


def _typecode_for(descr: str) -> str:
    """Return the :mod:`array` type code for a NumPy dtype string.

    The type code is chosen so its ``itemsize`` matches the dtype width exactly;
    ``array`` type codes are platform dependent, so a runtime width check guards
    against, for example, ``'i'`` being 2 bytes on an exotic platform.

    Complexity: O(1).
    """

    normalized = descr.strip()
    if normalized not in _DESCR_TO_TYPECODE:
        raise ValueError(f"unsupported NPY dtype descriptor: {descr!r}")
    typecode = _DESCR_TO_TYPECODE[normalized]
    expected_width = int(normalized[-1])
    actual_width = array.array(typecode).itemsize
    if actual_width != expected_width:
        # Fall back to a wider code with the correct width if one exists.
        for candidate in ("i", "l", "q"):
            if array.array(candidate).itemsize == expected_width:
                return candidate
        raise ValueError(
            f"no array type code of width {expected_width} for dtype {descr!r}"
        )
    return typecode


@dataclass
class NdArray:
    """A flat, C-contiguous numeric array with an explicit dtype and shape.

    Storing a flat :class:`array.array` rather than nested Python lists keeps a
    pairwise ``L x L x 128`` tensor at four bytes per element instead of the
    large per-object overhead of nested lists, which matters when a shard holds
    hundreds of positions.

    Attributes:
        descr: normalized little-endian NumPy dtype string, e.g. ``'<f4'``.
        shape: tuple of non-negative dimension sizes.
        data: flat values in C order; ``len(data) == prod(shape)``.

    Complexity: O(1) metadata plus O(prod(shape)) flat storage.
    """

    descr: str
    shape: Tuple[int, ...]
    data: array.array

    def __post_init__(self) -> None:
        """Validate that the flat buffer length matches the declared shape."""

        expected = _product(self.shape)
        if len(self.data) != expected:
            raise ValueError(
                f"data length {len(self.data)} does not match shape {self.shape} "
                f"(expected {expected})"
            )

    @property
    def size(self) -> int:
        """Total number of elements.

        Complexity: O(1).
        """

        return len(self.data)

    def row(self, index: int) -> Tuple[float, ...]:
        """Return C-order row ``index`` of a 2-D array as a tuple.

        Complexity: O(D) for the trailing dimension ``D``.
        """

        if len(self.shape) != 2:
            raise ValueError("row() requires a 2-D array")
        stride = self.shape[1]
        start = index * stride
        if not 0 <= index < self.shape[0]:
            raise IndexError("row index out of range")
        return tuple(self.data[start : start + stride])

    def to_nested(self) -> object:
        """Materialize the array as nested Python lists (or a scalar).

        Complexity: O(prod(shape)).
        """

        if not self.shape:
            return self.data[0]
        return _reshape(list(self.data), self.shape)

    @staticmethod
    def from_nested(values: object, *, kind: str = "float32") -> "NdArray":
        """Build an :class:`NdArray` from nested lists/tuples or a scalar.

        The nested structure must be rectangular; a ragged nested list raises.

        Complexity: O(prod(shape)).
        """

        if kind not in _KIND_TO_DESCR:
            raise ValueError(f"unsupported kind {kind!r}")
        descr = _KIND_TO_DESCR[kind]
        shape = _infer_shape(values)
        flat: List[float] = []
        _flatten_into(values, shape, 0, flat)
        typecode = _typecode_for(descr)
        if kind == "bool":
            buffer = array.array(typecode, (1 if bool(v) else 0 for v in flat))
        elif kind in ("int32", "int64", "uint8"):
            buffer = array.array(typecode, (int(v) for v in flat))
        else:
            buffer = array.array(typecode, (float(v) for v in flat))
        return NdArray(descr=descr, shape=shape, data=buffer)


def _product(shape: Sequence[int]) -> int:
    """Return the product of dimension sizes (empty shape -> scalar size 1)."""

    total = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"negative dimension in shape {tuple(shape)!r}")
        total *= dim
    return total


def _infer_shape(values: object) -> Tuple[int, ...]:
    """Infer the rectangular shape of nested sequences.

    Complexity: O(depth) since only the first element is followed per level.
    """

    shape: List[int] = []
    node = values
    while isinstance(node, (list, tuple)):
        shape.append(len(node))
        node = node[0] if node else None
    return tuple(shape)


def _flatten_into(values: object, shape: Tuple[int, ...], axis: int, out: List[float]) -> None:
    """Append the C-order flattening of ``values`` into ``out`` with shape checks.

    Complexity: O(prod(shape)).
    """

    if axis == len(shape):
        out.append(values)  # type: ignore[arg-type]
        return
    if not isinstance(values, (list, tuple)) or len(values) != shape[axis]:
        raise ValueError("nested values are ragged or inconsistent with shape")
    for child in values:
        _flatten_into(child, shape, axis + 1, out)


def _reshape(flat: List[object], shape: Tuple[int, ...]) -> object:
    """Reshape a flat C-order list into nested lists.

    Complexity: O(prod(shape)).
    """

    if len(shape) == 1:
        return flat
    stride = _product(shape[1:])
    return [_reshape(flat[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])]


def _format_header(descr: str, shape: Tuple[int, ...]) -> bytes:
    """Return the padded, newline-terminated NPY header block (v1.0 or v2.0).

    Complexity: O(1) in the array size.
    """

    shape_repr = "(" + "".join(f"{dim}, " for dim in shape) + ")" if shape else "()"
    header = "{" + f"'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, " + "}"
    header_bytes = header.encode("latin1")

    # Try version 1.0 (2-byte length); fall back to 2.0 (4-byte length).
    for version, length_struct in ((b"\x01\x00", "<H"), (b"\x02\x00", "<I")):
        length_field = struct.calcsize(length_struct)
        preamble = len(_MAGIC) + len(version) + length_field
        total = preamble + len(header_bytes) + 1  # +1 for the trailing newline
        padding = (-total) % _ALIGN
        padded = header_bytes + b" " * padding + b"\n"
        if len(padded) <= (0xFFFF if length_field == 2 else 0xFFFFFFFF):
            return _MAGIC + version + struct.pack(length_struct, len(padded)) + padded
    raise ValueError("NPY header too large to encode")


def _to_little_endian_bytes(arr: NdArray) -> bytes:
    """Serialize the flat buffer as little-endian raw bytes.

    Complexity: O(nbytes).
    """

    buffer = arr.data
    if sys.byteorder != "little" and buffer.itemsize > 1:
        buffer = array.array(buffer.typecode, buffer)
        buffer.byteswap()
    return buffer.tobytes()


def dumps_npy(arr: NdArray) -> bytes:
    """Encode an :class:`NdArray` as a complete ``.npy`` byte string.

    Complexity: O(nbytes).
    """

    return _format_header(arr.descr, arr.shape) + _to_little_endian_bytes(arr)


def loads_npy(blob: bytes) -> NdArray:
    """Decode ``.npy`` bytes into an :class:`NdArray`.

    Both format versions 1.0 and 2.0 are accepted.  ``fortran_order=True`` and
    unsupported dtypes raise rather than being silently reinterpreted.

    Complexity: O(nbytes).
    """

    if blob[: len(_MAGIC)] != _MAGIC:
        raise ValueError("not an NPY buffer: bad magic")
    major = blob[len(_MAGIC)]
    offset = len(_MAGIC) + 2
    if major == 1:
        (header_len,) = struct.unpack_from("<H", blob, offset)
        offset += 2
    elif major == 2:
        (header_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
    else:
        raise ValueError(f"unsupported NPY major version {major}")

    header_text = blob[offset : offset + header_len].decode("latin1")
    offset += header_len
    header = ast.literal_eval(header_text.strip())
    if not isinstance(header, Mapping):
        raise ValueError("NPY header is not a mapping")
    descr = str(header["descr"])
    if header.get("fortran_order", False):
        raise ValueError("Fortran-ordered NPY arrays are not supported")
    shape = tuple(int(dim) for dim in header["shape"])

    byte_order = descr[0]
    typecode = _typecode_for("<" + descr[1:] if byte_order in "<>" else descr)
    itemsize = array.array(typecode).itemsize
    count = _product(shape)
    payload = blob[offset : offset + count * itemsize]
    if len(payload) != count * itemsize:
        raise ValueError("NPY payload shorter than declared shape")
    buffer = array.array(typecode)
    buffer.frombytes(payload)
    host_is_little = sys.byteorder == "little"
    file_is_little = byte_order != ">"
    if itemsize > 1 and host_is_little != file_is_little:
        buffer.byteswap()
    normalized_descr = ("<" if file_is_little else ">") + descr[1:] if byte_order in "<>" else descr
    return NdArray(descr=normalized_descr, shape=shape, data=buffer)


def save_npy(path: Union[str, Path], arr: NdArray) -> None:
    """Write an :class:`NdArray` to ``path`` as a ``.npy`` file.

    Complexity: O(nbytes).
    """

    Path(path).write_bytes(dumps_npy(arr))


def load_npy(path: Union[str, Path]) -> NdArray:
    """Read a ``.npy`` file written by this codec or by NumPy.

    Complexity: O(nbytes).
    """

    return loads_npy(Path(path).read_bytes())


def save_npz(path: Union[str, Path], arrays: Mapping[str, NdArray], *, compress: bool = False) -> None:
    """Write a mapping of named arrays as a ``.npz`` archive.

    Members are stored in sorted key order so the archive is byte-deterministic.
    ``compress=False`` matches ``numpy.savez`` (ZIP_STORED); ``compress=True``
    matches ``numpy.savez_compressed`` (ZIP_DEFLATED).

    Complexity: O(total bytes).
    """

    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(Path(path), "w", compression=mode) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(filename=f"{name}.npy")
            info.compress_type = mode
            zf.writestr(info, dumps_npy(arrays[name]))


def load_npz(path: Union[str, Path]) -> Dict[str, NdArray]:
    """Read a ``.npz`` archive into a mapping of arrays keyed by member name.

    Complexity: O(total bytes).
    """

    result: Dict[str, NdArray] = {}
    with zipfile.ZipFile(Path(path), "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                raise ValueError(f"unexpected NPZ member without .npy suffix: {member}")
            result[member[: -len(".npy")]] = loads_npy(zf.read(member))
    return result


def load_npz_member(path: Union[str, Path], name: str) -> NdArray:
    """Read a single named array from a ``.npz`` archive.

    Full-scale frozen-feature training often needs only the ``single`` array for
    one sequence at a time.  Reading one ZIP member keeps the lookup cost
    proportional to that record's ``L * d_single`` bytes instead of the whole
    child shard.  The member name is the logical key without the ``.npy`` suffix,
    matching :func:`load_npz`.

    Complexity: O(member bytes).
    """

    member = f"{name}.npy"
    with zipfile.ZipFile(Path(path), "r") as zf:
        try:
            return loads_npy(zf.read(member))
        except KeyError as exc:
            raise ValueError(f"NPZ archive is missing member {name!r}") from exc


def load_npz_members(path: Union[str, Path], names: Iterable[str]) -> Dict[str, NdArray]:
    """Read selected named arrays from a ``.npz`` archive in one ZIP session.

    Formula: for requested logical keys ``S = {s_1, ..., s_m}``, read members
    ``s_j + '.npy'`` and decode each NPY payload independently.  Opening the ZIP
    central directory once avoids the repeated ``O(directory entries)`` setup
    cost paid by :func:`load_npz_member` when a training mini-batch asks for many
    frozen-feature rows from the same shard.

    Complexity: O(sum selected member bytes + m).
    """

    requested = list(dict.fromkeys(str(name) for name in names))
    result: Dict[str, NdArray] = {}
    with zipfile.ZipFile(Path(path), "r") as zf:
        for name in requested:
            member = f"{name}.npy"
            try:
                result[name] = loads_npy(zf.read(member))
            except KeyError as exc:
                raise ValueError(f"NPZ archive is missing member {name!r}") from exc
    return result
