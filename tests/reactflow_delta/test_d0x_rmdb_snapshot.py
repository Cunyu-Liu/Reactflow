from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "reactflow_delta" / "d0x_freeze_rmdb_release_index.py"
SPEC = importlib.util.spec_from_file_location("d0x_freeze_rmdb_release_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payloads() -> list[tuple[bytes, dict]]:
    result = []
    for index, tag in enumerate(MODULE.RELEASE_TAGS, 1):
        accession = f"FIXTURE_{index}"
        payload = {
            "id": 100 + index,
            "tag_name": tag,
            "published_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "assets": [
                {
                    "id": 200 + index,
                    "name": accession + ".rdat",
                    "size": 1000 + index,
                    "digest": "sha256:" + (str(index) * 64),
                    "updated_at": "2026-01-02T00:00:00Z",
                    "browser_download_url": (
                        "https://github.com/DasLab/rmdb.github.io/releases/download/"
                        f"{tag}/{accession}.rdat"
                    ),
                }
            ],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        result.append((raw, payload))
    return result


class RMDBSnapshotTests(unittest.TestCase):
    def test_one_to_one_snapshot_is_deterministic_and_not_searched(self) -> None:
        registry = {f"FIXTURE_{index}" for index in range(1, 6)}
        records, summary = MODULE.normalize_release_payloads(
            payloads(),
            registry_ids=registry,
            frozen_at="2026-08-03T22:40:15+08:00",
            rmdb_commit="a" * 40,
        )
        self.assertEqual([row["source_accession"] for row in records], sorted(registry))
        self.assertTrue(summary["membership_match"])
        self.assertEqual(summary["asset_count"], 5)
        self.assertTrue(all(row["initial_disposition"] == "NOT_SEARCHED" for row in records))

    def test_missing_upstream_digest_fails_closed(self) -> None:
        items = payloads()
        items[0][1]["assets"][0]["digest"] = None
        with self.assertRaisesRegex(MODULE.SnapshotError, "upstream SHA-256"):
            MODULE.normalize_release_payloads(
                items,
                registry_ids={f"FIXTURE_{index}" for index in range(1, 6)},
                frozen_at="2026-08-03T22:40:15+08:00",
                rmdb_commit="a" * 40,
            )

    def test_registry_asset_membership_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(MODULE.SnapshotError, "membership mismatch"):
            MODULE.normalize_release_payloads(
                payloads(),
                registry_ids={"NOT_THE_ASSET_SET"},
                frozen_at="2026-08-03T22:40:15+08:00",
                rmdb_commit="a" * 40,
            )

    def test_duplicate_asset_name_across_releases_fails_closed(self) -> None:
        items = payloads()
        items[1][1]["assets"][0]["name"] = items[0][1]["assets"][0]["name"]
        with self.assertRaisesRegex(MODULE.SnapshotError, "duplicate asset name"):
            MODULE.normalize_release_payloads(
                items,
                registry_ids={f"FIXTURE_{index}" for index in range(1, 6)},
                frozen_at="2026-08-03T22:40:15+08:00",
                rmdb_commit="a" * 40,
            )


if __name__ == "__main__":
    unittest.main()
