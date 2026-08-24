"""Frozen internal schemas shared by Model Rescue v5 stages."""

CACHE_SCHEMA = "reactflow_delta.model_rescue_v5_ensemble_cache.v1"
SOURCE_COLUMNS = ("id", "sequence", "puzzle", "method", "sub_start", "mutA")
FEATURE_NAMES = (
    "delta_unpaired_probability",
    "delta_pairing_entropy",
    "delta_source_receiver_pair_probability",
    "wt_source_receiver_pair_probability",
    "mutant_source_receiver_pair_probability",
    "receiver_bpp_row_l1_change",
    "receiver_bpp_row_l2_change",
    "delta_upstream_pairing_mass",
    "delta_downstream_pairing_mass",
    "global_bpp_frobenius_change",
    "ensemble_free_energy_change",
    "source_unpaired_probability_change",
)
