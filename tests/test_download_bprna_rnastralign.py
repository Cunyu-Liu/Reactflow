"""Unit tests for ``scripts/download_bprna_rnastralign.py`` (C1-1 Task 2b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import from the script module (scripts/ is not a package, so use importlib).
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_bprna_rnastralign.py"
_spec = importlib.util.spec_from_file_location("download_bprna_rnastralign", _SCRIPT_PATH)
download_module = importlib.util.module_from_spec(_spec)
sys.modules["download_bprna_rnastralign"] = download_module
_spec.loader.exec_module(download_module)


class TestDownloadSpecs:
    def test_two_specs(self):
        assert len(download_module.DOWNLOAD_SPECS) == 2

    def test_bprna_present(self):
        names = [s.name for s in download_module.DOWNLOAD_SPECS]
        assert "bpRNA" in names

    def test_rnastralign_present(self):
        names = [s.name for s in download_module.DOWNLOAD_SPECS]
        assert "RNAStrAlign" in names

    def test_specs_have_upstream_url(self):
        for s in download_module.DOWNLOAD_SPECS:
            assert s.upstream_url.startswith("http"), f"{s.name} invalid URL"

    def test_specs_have_cache_filename(self):
        for s in download_module.DOWNLOAD_SPECS:
            assert s.expected_cache_filename.endswith(".jsonl")


class TestBuildManifest:
    def test_manifest_no_cache_dir(self):
        manifest = download_module.build_manifest(download_module.DOWNLOAD_SPECS, cache_dir=None)
        assert manifest["schema_version"] == "1.0"
        assert "build_timestamp" in manifest
        assert "sources" in manifest
        assert len(manifest["sources"]) == 2
        for s in manifest["sources"]:
            assert s["downloaded"] is False
            assert s["sha256"] == "not_downloaded"

    def test_manifest_with_empty_cache_dir(self, tmp_path: Path):
        manifest = download_module.build_manifest(download_module.DOWNLOAD_SPECS, cache_dir=tmp_path)
        for s in manifest["sources"]:
            assert s["downloaded"] is False
            assert s["sha256"] == "not_downloaded"
            assert s["actual_record_count"] == 0

    def test_manifest_with_existing_cache_file(self, tmp_path: Path):
        # Create a fake bpRNA.jsonl file in the cache dir
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        bprna_path = cache_dir / "bpRNA.jsonl"
        bprna_path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        manifest = download_module.build_manifest(download_module.DOWNLOAD_SPECS, cache_dir=cache_dir)
        bprna_entry = next(s for s in manifest["sources"] if s["name"] == "bpRNA")
        assert bprna_entry["downloaded"] is True
        assert bprna_entry["sha256"] != "not_downloaded"
        assert len(bprna_entry["sha256"]) == 64  # SHA-256 hex digest
        assert bprna_entry["actual_record_count"] == 3
        assert bprna_entry["size_bytes"] > 0

    def test_manifest_serializable(self):
        manifest = download_module.build_manifest(download_module.DOWNLOAD_SPECS, cache_dir=None)
        # Should serialize to JSON without error.
        json.dumps(manifest)


class TestDownloadSource:
    def test_raises_not_implemented(self, tmp_path: Path):
        spec = download_module.DOWNLOAD_SPECS[0]
        with pytest.raises(NotImplementedError, match="Manual steps"):
            download_module.download_source(spec, tmp_path)

    def test_error_message_contains_url(self, tmp_path: Path):
        spec = download_module.DOWNLOAD_SPECS[0]
        with pytest.raises(NotImplementedError) as exc_info:
            download_module.download_source(spec, tmp_path)
        assert spec.upstream_url in str(exc_info.value)


class TestSha256OfFile:
    def test_known_content(self, tmp_path: Path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"hello world")
        sha = download_module._sha256_of_file(path)
        # Known SHA-256 of "hello world"
        assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        sha = download_module._sha256_of_file(path)
        # Known SHA-256 of empty string
        assert sha == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
