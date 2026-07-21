import array
import struct

import pytest

from reactflow.npio import (
    NdArray,
    dumps_npy,
    load_npy,
    load_npz,
    load_npz_member,
    load_npz_members,
    loads_npy,
    save_npy,
    save_npz,
)


def test_from_nested_infers_shape_and_flattens_c_order():
    arr = NdArray.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], kind="float32")
    assert arr.shape == (2, 3)
    assert arr.descr == "<f4"
    assert list(arr.data) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert arr.size == 6
    assert arr.row(0) == (1.0, 2.0, 3.0)
    assert arr.row(1) == (4.0, 5.0, 6.0)


def test_from_nested_rejects_ragged_and_unknown_kind():
    with pytest.raises(ValueError, match="ragged"):
        NdArray.from_nested([[1.0, 2.0], [3.0]], kind="float32")
    with pytest.raises(ValueError, match="unsupported kind"):
        NdArray.from_nested([1.0], kind="float16")


def test_ndarray_post_init_checks_length_and_negative_shape():
    with pytest.raises(ValueError, match="does not match shape"):
        NdArray(descr="<f4", shape=(2, 2), data=array.array("f", [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="negative dimension"):
        NdArray(descr="<f4", shape=(-1, 2), data=array.array("f", []))


def test_row_requires_2d_and_checks_bounds():
    vec = NdArray.from_nested([1.0, 2.0, 3.0], kind="float32")
    with pytest.raises(ValueError, match="2-D"):
        vec.row(0)
    mat = NdArray.from_nested([[1.0], [2.0]], kind="float32")
    with pytest.raises(IndexError):
        mat.row(5)


@pytest.mark.parametrize(
    "nested, kind",
    [
        ([[1.5, -2.0], [3.25, 4.0]], "float32"),
        ([[1.5, -2.0], [3.25, 4.0]], "float64"),
        ([[1, -2, 3], [4, 5, 6]], "int32"),
        ([[1, -2], [3, 4]], "int64"),
        ([[True, False], [False, True]], "bool"),
        ([[0, 255], [7, 128]], "uint8"),
    ],
)
def test_npy_roundtrip_all_supported_kinds(nested, kind):
    arr = NdArray.from_nested(nested, kind=kind)
    restored = loads_npy(dumps_npy(arr))
    assert restored.shape == arr.shape
    assert list(restored.data) == list(arr.data)


def test_npy_header_is_64_byte_aligned():
    arr = NdArray.from_nested([[1.0, 2.0, 3.0]], kind="float32")
    blob = dumps_npy(arr)
    newline = blob.index(b"\n") + 1
    assert newline % 64 == 0
    assert blob[:6] == b"\x93NUMPY"
    assert blob[6:8] == b"\x01\x00"  # version 1.0 for a small header


def test_scalar_roundtrip_and_to_nested():
    scalar = NdArray.from_nested(3.5, kind="float64")
    assert scalar.shape == ()
    assert scalar.to_nested() == 3.5
    restored = loads_npy(dumps_npy(scalar))
    assert restored.shape == ()
    assert restored.to_nested() == 3.5


def test_to_nested_reconstructs_3d_structure():
    nested = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    arr = NdArray.from_nested(nested, kind="float32")
    assert arr.shape == (2, 2, 2)
    assert arr.to_nested() == nested


def test_loads_npy_rejects_bad_magic_and_version():
    with pytest.raises(ValueError, match="bad magic"):
        loads_npy(b"not-an-npy-file")
    fake = b"\x93NUMPY" + bytes([9, 0]) + struct.pack("<H", 0)
    with pytest.raises(ValueError, match="major version"):
        loads_npy(fake)


def test_loads_npy_rejects_fortran_order():
    header = "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }"
    hb = header.encode("latin1")
    pad = (-(6 + 2 + 2 + len(hb) + 1)) % 64
    padded = hb + b" " * pad + b"\n"
    blob = b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(padded)) + padded + struct.pack("<f", 1.0)
    with pytest.raises(ValueError, match="Fortran"):
        loads_npy(blob)


def test_loads_npy_rejects_truncated_payload():
    arr = NdArray.from_nested([1.0, 2.0, 3.0, 4.0], kind="float32")
    blob = dumps_npy(arr)
    with pytest.raises(ValueError, match="shorter than declared"):
        loads_npy(blob[:-4])


def test_version_2_header_for_large_shape():
    # A very high-dimensional shape forces the header past 65535 bytes, so the
    # codec must emit format version 2.0 with a 4-byte length field.  Each "1, "
    # dimension token is 3 bytes, so ~22k dims already exceed the v1.0 limit.
    big = NdArray(descr="<f4", shape=(1,) * 25000, data=array.array("f", [1.0]))
    blob = dumps_npy(big)
    assert blob[6:8] == b"\x02\x00"
    restored = loads_npy(blob)
    assert restored.shape == big.shape


def test_save_and_load_npy_file(tmp_path):
    arr = NdArray.from_nested([[1.0, 2.0], [3.0, 4.0]], kind="float32")
    path = tmp_path / "a.npy"
    save_npy(path, arr)
    restored = load_npy(path)
    assert restored.shape == (2, 2)
    assert list(restored.data) == [1.0, 2.0, 3.0, 4.0]


def test_npz_roundtrip_sorted_and_deterministic(tmp_path):
    arrays = {
        "single": NdArray.from_nested([[1.0, 2.0]], kind="float32"),
        "pair": NdArray.from_nested([[[1.0]]], kind="float32"),
    }
    path1 = tmp_path / "a.npz"
    path2 = tmp_path / "b.npz"
    save_npz(path1, arrays)
    save_npz(path2, arrays)
    assert path1.read_bytes() == path2.read_bytes()  # byte-deterministic
    loaded = load_npz(path1)
    assert set(loaded) == {"single", "pair"}
    assert loaded["single"].row(0) == (1.0, 2.0)
    single = load_npz_member(path1, "single")
    assert single.row(0) == (1.0, 2.0)


def test_load_npz_member_rejects_missing_member(tmp_path):
    path = tmp_path / "a.npz"
    save_npz(path, {"present": NdArray.from_nested([1.0], kind="float32")})
    with pytest.raises(ValueError, match="missing member"):
        load_npz_member(path, "absent")


def test_load_npz_members_reads_selected_members_in_one_call(tmp_path):
    """Selected multi-member reads must equal independent single-member reads."""

    path = tmp_path / "a.npz"
    save_npz(
        path,
        {
            "first": NdArray.from_nested([[1.0, 2.0]], kind="float32"),
            "second": NdArray.from_nested([[3.0, 4.0]], kind="float32"),
            "third": NdArray.from_nested([[5.0, 6.0]], kind="float32"),
        },
    )

    loaded = load_npz_members(path, ["second", "first", "second"])

    assert set(loaded) == {"first", "second"}
    assert loaded["first"].row(0) == (1.0, 2.0)
    assert loaded["second"].row(0) == (3.0, 4.0)
    with pytest.raises(ValueError, match="missing member"):
        load_npz_members(path, ["absent"])


def test_npz_compressed_roundtrip(tmp_path):
    arrays = {"x": NdArray.from_nested([[float(i) for i in range(50)]], kind="float32")}
    path = tmp_path / "c.npz"
    save_npz(path, arrays, compress=True)
    loaded = load_npz(path)
    assert loaded["x"].shape == (1, 50)
    assert list(loaded["x"].data) == [float(i) for i in range(50)]


def test_load_npz_rejects_unexpected_member(tmp_path):
    import zipfile

    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("not_an_array.txt", b"hello")
    with pytest.raises(ValueError, match="without .npy suffix"):
        load_npz(path)


def _npy_with_header(header: str, payload: bytes = b"") -> bytes:
    hb = header.encode("latin1")
    pad = (-(6 + 2 + 2 + len(hb) + 1)) % 64
    padded = hb + b" " * pad + b"\n"
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(padded)) + padded + payload


def test_loads_npy_rejects_unsupported_dtype():
    blob = _npy_with_header("{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }", b"\x00" * 8)
    with pytest.raises(ValueError, match="unsupported NPY dtype"):
        loads_npy(blob)


def test_loads_npy_rejects_non_mapping_header():
    blob = _npy_with_header("[1, 2, 3]")
    with pytest.raises(ValueError, match="not a mapping"):
        loads_npy(blob)
