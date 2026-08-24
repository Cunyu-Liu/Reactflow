"""Frozen schemas and constants for Model Rescue v7."""

CACHE_SCHEMA = "reactflow_delta.model_rescue_v7_rinalmo_dependency_cache.v1"
QUALIFICATION_SCHEMA = (
    "reactflow_delta.model_rescue_v7_rinalmo_dependency_cache_qualification.v1"
)

FEATURE_NAMES = (
    "rinalmo_signed_log_odds_shift_a",
    "rinalmo_signed_log_odds_shift_c",
    "rinalmo_signed_log_odds_shift_g",
    "rinalmo_signed_log_odds_shift_u",
    "rinalmo_signed_log_odds_shift_wt_receiver_base",
    "rinalmo_max_absolute_log_odds_shift",
)

RNA_BASES = ("A", "C", "G", "U")
RNA_BASE_TO_INDEX = {base: index for index, base in enumerate(RNA_BASES)}
RINALMO_VOCAB_BASE_TOKENS = ("A", "C", "G", "T")
RINALMO_ACGU_TOKEN_INDICES = (5, 6, 7, 8)
RINALMO_SEQUENCE_TOKEN_OFFSET = 1
LOG_ODDS_EPSILON = 1.0e-10

RINALMO_MODEL_NAME = "giga-v1"
RINALMO_CONFIG_NAME = "giga"
RINALMO_PARAMETER_COUNT = 650_000_000
RINALMO_CODE_COMMIT = "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"
DEPENDENCY_CODE_COMMIT = "d70b8816988eb83602f408d02fd63f1be82601e2"

FORBIDDEN_CACHE_DATASETS = frozenset(
    {
        "reactivity",
        "reactivity_error",
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "score",
        "loss",
    }
)
