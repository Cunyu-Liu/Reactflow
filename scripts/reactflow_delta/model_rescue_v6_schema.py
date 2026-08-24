"""Frozen internal schemas shared by Model Rescue v6 stages."""

CACHE_SCHEMA = "reactflow_delta.model_rescue_v6_constrained_ensemble_cache.v1"
QUALIFICATION_SCHEMA = (
    "reactflow_delta.model_rescue_v6_constrained_ensemble_cache_qualification.v1"
)
METADATA_COLUMNS = ("id", "sequence", "puzzle", "method", "sub_start", "mutA")
FEATURE_NAMES = (
    "constrained_delta_unpaired_probability",
    "constrained_delta_pairing_entropy",
    "constrained_delta_source_receiver_pair_probability",
    "constrained_wt_source_receiver_pair_probability",
    "constrained_mutant_source_receiver_pair_probability",
    "constrained_receiver_bpp_row_l1_change",
    "constrained_receiver_bpp_row_l2_change",
    "constrained_delta_upstream_pairing_mass",
    "constrained_delta_downstream_pairing_mass",
    "constrained_global_bpp_frobenius_change",
    "constrained_ensemble_free_energy_change",
    "constrained_source_unpaired_probability_change",
)

# The cache preserves all interpretable channels, but the learning basis removes
# one exact algebraic duplicate:
# delta_unpaired + delta_upstream_pairing_mass + delta_downstream_pairing_mass = 0.
REDUNDANT_PROBE_FEATURE = "constrained_delta_downstream_pairing_mass"
PROBE_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name != REDUNDANT_PROBE_FEATURE
)
PROBE_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in PROBE_FEATURE_NAMES)

DEIGAN_SLOPE = 1.8
DEIGAN_INTERCEPT = -0.6
MISSING_REACTIVITY = -999.0
