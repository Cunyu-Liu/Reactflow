"""Focused immutable-provenance checks for the foundation assets used in V4/V7."""

from reactflow.backbones import checkpoint_governance
from reactflow.backbones.foundation import rinalmo, rna_fm


def test_rinalmo_giga_provenance_distinguishes_code_and_weights() -> None:
    config = rinalmo.default_config()
    governed = checkpoint_governance.BACKBONE_PROVENANCE_REGISTRY["rinalmo"]

    assert rinalmo.MODEL_VARIANT == "giga-v1"
    assert rinalmo.CODE_SOURCE == "github:lbcb-sci/RiNALMo"
    assert rinalmo.CODE_REVISION == (
        "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"
    )
    assert rinalmo.CODE_LICENSE == "Apache-2.0"
    assert rinalmo.CHECKPOINT_SOURCE == "zenodo:15043668"
    assert rinalmo.CHECKPOINT_REVISION == "10.5281/zenodo.15043668"
    assert rinalmo.CHECKPOINT_LICENSE == "CC-BY-4.0"
    assert rinalmo.CODE_LICENSE != rinalmo.CHECKPOINT_LICENSE
    assert rinalmo.CITATION == "Penić et al., Nature Communications (2025)"
    assert rinalmo.CITATION_DOI == "10.1038/s41467-025-60872-5"
    assert rinalmo.DOWNLOAD_URL == (
        "https://zenodo.org/records/15043668/files/"
        "rinalmo_giga_pretrained.pt"
    )

    assert config.model_source == rinalmo.CHECKPOINT_SOURCE
    assert config.model_revision == rinalmo.CHECKPOINT_REVISION
    assert config.license == rinalmo.CHECKPOINT_LICENSE
    assert config.code_revision == rinalmo.CODE_REVISION
    assert config.downloaded is False  # The generic adapter has no loader.

    assert governed.model_source == config.model_source
    assert governed.exact_revision == config.model_revision
    assert governed.license == config.license
    assert governed.code_revision == config.code_revision
    assert governed.download_url == rinalmo.DOWNLOAD_URL
    assert checkpoint_governance.validate_provenance(governed) == []


def test_rna_fm_v4_provenance_is_pinned_and_uses_the_640d_asset() -> None:
    config = rna_fm.default_config()
    governed = checkpoint_governance.BACKBONE_PROVENANCE_REGISTRY["rna_fm"]

    assert rna_fm.MODEL_VARIANT == "rna_fm_t12"
    assert rna_fm.CODE_SOURCE == "github:ml4bio/RNA-FM"
    assert rna_fm.CODE_REVISION == (
        "348951516e0963d22bbb33b3c9fc18c89081d38e"
    )
    assert rna_fm.CODE_LICENSE == "MIT"
    assert rna_fm.CHECKPOINT_SOURCE == "huggingface:cuhkaih/rnafm"
    assert rna_fm.CHECKPOINT_REVISION == (
        "91d4a46d28d8054a7b429955e8fc0c253ba0afd6"
    )
    assert rna_fm.CHECKPOINT_LICENSE == "Apache-2.0"
    assert rna_fm.CODE_LICENSE != rna_fm.CHECKPOINT_LICENSE
    assert rna_fm.EXPECTED_SINGLE_DIM == 640
    assert "/main/" not in rna_fm.DOWNLOAD_URL
    assert rna_fm.CHECKPOINT_REVISION in rna_fm.DOWNLOAD_URL

    assert config.model_source == rna_fm.CHECKPOINT_SOURCE
    assert config.model_revision == rna_fm.CHECKPOINT_REVISION
    assert config.license == rna_fm.CHECKPOINT_LICENSE
    assert config.code_revision == rna_fm.CODE_REVISION
    assert config.frozen_feature_dim == 640
    assert config.downloaded is False  # V4 uses a separate frozen-cache loader.

    assert governed.model_source == config.model_source
    assert governed.exact_revision == config.model_revision
    assert governed.license == config.license
    assert governed.code_revision == config.code_revision
    assert governed.download_url == rna_fm.DOWNLOAD_URL
    assert checkpoint_governance.validate_provenance(governed) == []
