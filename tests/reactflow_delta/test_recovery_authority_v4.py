from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "reactflow_delta" / "validate_recovery_v4.py"
SPEC = importlib.util.spec_from_file_location("validate_recovery_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoveryAuthorityV4Tests(unittest.TestCase):
    def test_historical_receipt_is_git_and_hash_valid(self) -> None:
        result = MODULE.validate_receipt(Path.cwd())
        self.assertEqual(result["result"], "PASS")
        self.assertGreaterEqual(len(result["checks"]), 15)
        self.assertTrue(all(row["status"] == "PASS" for row in result["checks"]))

    def test_duplicate_yaml_key_is_rejected(self) -> None:
        with self.assertRaises(MODULE.DuplicateKeyError):
            MODULE.strict_yaml(b"field: one\nfield: two\n", "synthetic")

    def test_ledger_must_be_sorted_unique_and_non_self_referential(self) -> None:
        digest = b"a" * 64
        with self.assertRaisesRegex(ValueError, "sorted"):
            MODULE.parse_ledger(
                digest + b"  z/path\n" + digest + b"  a/path\n"
            )
        with self.assertRaisesRegex(ValueError, "self/sentinel"):
            MODULE.parse_ledger(
                digest
                + b"  configs/reactflow_delta/authority_epoch_1.bundle.sha256\n"
            )


if __name__ == "__main__":
    unittest.main()
