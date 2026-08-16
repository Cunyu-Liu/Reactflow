#!/usr/bin/env python3
"""Sign the M0-X approval record with the user's explicit message.

The user's message "1、签署 2、可用 3、先只执行 EPRO_DEV_01-02 两轮 4、确认。请开始"
is the explicit sign-off for the M0-X authority (epoch 12).  This script records
that signature in the approval record, binds the approval-record sha256 into the
amendment, and updates the active_contract bindings via targeted text replacement
(no reformatting of the large active_contract file).  It does not change any
scientific content, data, or the M0-X grant itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


class DuplicateKeyError(ValueError):
    pass


class Loader(yaml.SafeLoader):
    pass


def _mapping(loader: Loader, node, deep: bool = False) -> dict:
    result: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DuplicateKeyError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path.cwd()
    approval_path = root / "docs/approvals/reactflow_delta_v4_m0x_approval_20260804.yaml"
    amendment_path = root / "docs/contracts/amendments/reactflow_delta_v4_m0x_20260804.yaml"
    active_path = root / "configs/reactflow_delta/active_contract.yaml"

    approval = load_yaml(approval_path)
    amendment = load_yaml(amendment_path)
    active_text = active_path.read_text(encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()

    # 1. Record the user's explicit signature in the approval record.
    approval["signature_status"] = "SIGNED_BY_EXPLICIT_USER_MESSAGE"
    approval["signed_by"] = "user (explicit message 2026-08-04)"
    approval["signed_at"] = now
    approval["external_identity_status"] = "VERIFIED_VIA_EXPLICIT_USER_MESSAGE"
    approval["scope_assurance"] = "EXPLICIT_USER_MESSAGE_CONTEXT"
    dump_yaml(approval_path, approval)
    approval_sha = digest(approval_path)

    # 2. Bind the new approval-record sha256 into the amendment.
    amendment["approval_binding"]["approval_record_sha256"] = approval_sha
    amendment["approval_binding"]["signature_status"] = "SIGNED_BY_EXPLICIT_USER_MESSAGE"
    amendment["approval_binding"]["external_identity_status"] = "VERIFIED_VIA_EXPLICIT_USER_MESSAGE"
    dump_yaml(amendment_path, amendment)
    amendment_sha = digest(amendment_path)

    # 3. Update active_contract bindings via targeted text replacement.
    buf = [active_text]

    def rep(old, new, label):
        assert old in buf[0], f"not found: {label}"
        buf[0] = buf[0].replace(old, new, 1)

    rep("  m0x_approval_record_sha256: dce0809b6dd3f7e813d294602a5bed27b5255efe23bfc17714cb303df042c820",
        f"  m0x_approval_record_sha256: {approval_sha}", "m0x_approval_record_sha256")
    rep("  m0x_amendment_sha256: add97a4863ec4680fb8fd811bf7511ad58448b9ff8c4ab69bc1f79a1a8fca6c6",
        f"  m0x_amendment_sha256: {amendment_sha}", "m0x_amendment_sha256")
    rep("  approval_record_sha256: dce0809b6dd3f7e813d294602a5bed27b5255efe23bfc17714cb303df042c820",
        f"  approval_record_sha256: {approval_sha}", "authorization approval_record_sha256")
    rep("  cryptographic_signature_status: NOT_PROVIDED_NOT_ASSERTED",
        "  cryptographic_signature_status: SIGNED_BY_EXPLICIT_USER_MESSAGE",
        "cryptographic_signature_status")

    active_path.write_text(buf[0], encoding="utf-8")

    print(json.dumps({
        "approval_record_sha256": approval_sha,
        "amendment_sha256": amendment_sha,
        "signed_at": now,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())