from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v7m1_authorizes_only_outcome_blind_foundation_cache() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")

    assert active["authority"]["current_phase"] == "V7M1"
    assert active["runnable_phases"] == ["V7M1"]
    assert active["authorization"]["implementation_allowed"] is True
    assert (
        active["authorization"]["outcome_blind_foundation_preparation_allowed"]
        is True
    )
    assert active["authorization"]["outcome_blind_cache_preparation_allowed"] is True
    assert active["authorization"]["internal_development_probe_allowed"] is False
    assert active["training_allowed"] is False
    assert active["candidate_model_training_allowed"] is False
    assert active["outcome_blind_cache_allowed"] is True
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert v7["contract_status"] == (
        "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_AUTHORIZED"
    )

    phase_status = {row["id"]: row["status"] for row in v7["phase_graph"]}
    assert phase_status["V7M0"] == "PASS"
    assert phase_status["V7M1"] == "AUTHORIZED"
    assert phase_status["V7M2"] == "NOT_AUTHORIZED"
    assert phase_status["V7M3"] == "NOT_AUTHORIZED"


def test_v7_dependency_definition_is_one_fixed_published_intervention() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    method = v7["literature_and_implementation"]["dependency_method"]
    model = v7["literature_and_implementation"]["foundation_model"]
    dependency = v7["dependency_definition"]

    assert method["doi"] == "10.1038/s41588-025-02347-3"
    assert model["doi"] == "10.1038/s41467-025-60872-5"
    assert model["model_name"] == "giga-v1"
    assert model["parameters"] == 650000000
    assert model["weights_trainable"] is False
    assert model["runtime_pytorch"] == "2.1.0"
    assert model["runtime_cuda"] == "11.8"
    assert model["runtime_flash_attention"] == "2.3.2"
    assert model["attention_backend"] == "OFFICIAL_FLASH_ATTENTION"
    assert model["parameter_dtype"] == "FLOAT32_OFFICIAL_CHECKPOINT"
    assert model["forward_autocast_dtype"] == (
        "FLOAT16_OFFICIAL_CUDA_AUTOCAST_DEFAULT"
    )
    assert model["output_logit_and_log_odds_dtype"] == "FLOAT32"
    assert dependency["sequence_input"] == (
        "FULL_UNMASKED_WT_AND_EXACT_REGISTERED_MUTANT_SEQUENCE"
    )
    assert dependency["self_dependency"] == (
        "ZERO_AT_RECEIVER_EQUAL_TO_MUTATION_SOURCE"
    )
    assert len(dependency["fixed_feature_basis"]) == 6
    assert dependency["mutant_outcome_columns_allowed"] is False
    assert dependency["external_outcome_allowed"] is False
    assert dependency["feature_or_layer_search_allowed"] is False
    assert dependency["model_size_search_allowed"] is False


def test_v7_requires_incremental_probe_and_top_journal_model_gate() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    probe = v7["eligibility_probe"]
    candidate = v7["candidate_model"]
    gate = v7["development_gate"]

    assert probe["baseline_features"] == (
        "DIRECT_18_PLUS_V5_UNCONSTRAINED_12_PLUS_V6_CONSTRAINED_11"
    )
    assert probe["candidate_features"] == "BASELINE_PLUS_V7_RINALMO_DEPENDENCY_6"
    assert probe["implementation_invariants"]["baseline_replay"] == (
        "V7_BASELINE_MUST_MATCH_V6_CANDIDATE_PREDICTIONS"
    )
    assert probe["implementation_invariants"]["baseline_replay_atol"] == 1e-12
    assert probe["gate"]["signed_delta_relative_mae_gain_min"] == 0.01
    assert candidate["prerequisite"] == (
        "EXACT_V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE"
    )
    assert candidate["selection_allowed"] is False
    assert candidate["controls"] == [
        "EQUAL_CAPACITY_ZERO_DEPENDENCY_OPERATOR",
        "EQUAL_CAPACITY_HALF_LENGTH_CYCLIC_RECEIVER_SHIFTED_DEPENDENCY_OPERATOR",
    ]
    distance = candidate["trainable_operator"]["signed_distance_encoding"]
    assert distance["raw_distance"] == "RECEIVER_INDEX_MINUS_SOURCE_INDEX"
    assert distance["width"] == 32
    assert distance["definition"] == (
        "STANDARD_TRANSFORMER_SINUSOIDAL_NO_LEARNABLE_PARAMETERS"
    )
    assert distance["normalization_or_frequency_search_allowed"] is False
    assert gate["versus_corrected_b1"]["crps_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["signed_delta_mae_relative_gain_min"] == 0.05
    assert gate["attribution"]["primary_vs_each_control_crps_ci_lower_gt"] == 0.0
    assert v7["formal_confirmation"]["seeds"] == [0, 1, 2, 3, 4]
    checkpoint = v7["formal_confirmation"]["corrected_b1_checkpoint_policy"]
    assert checkpoint["epochs"] == 40
    assert checkpoint["learning_rate"] == 0.001
    assert checkpoint["weight_decay"] == 0.0
    assert checkpoint["seed_zero_reuse"] == (
        "EXACT_R3C3_QUALIFIED_CHECKPOINT_FOR_MATCHING_OUTER_FOLD"
    )
    assert checkpoint["sharing"] == (
        "SAME_FOLD_SEED_CHECKPOINT_SHARED_BY_BASELINE_PRIMARY_AND_BOTH_CONTROLS"
    )
    assert checkpoint["checkpoint_or_seed_selection_allowed"] is False
    assert v7["claim_policy"]["publication_ready"] is False


def test_v7_preserves_all_prior_terminal_states_and_running_v3() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    v6 = _yaml("configs/reactflow_delta/model_rescue_v6_amendment.yaml")
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")

    assert v7["parent"]["v3_status_at_fork"] == (
        "R3C3_CORRECTED_EXPERT_REBUILD_IN_PROGRESS"
    )
    assert v7["parent"]["v6_terminal_status"] == "MODEL_RESCUE_V6_FAIL"
    assert v6["contract_status"] == (
        "TERMINAL_V6M2_MODEL_RESCUE_V6_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v5["contract_status"] == (
        "TERMINAL_V5M2_MODEL_RESCUE_V5_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v4["contract_status"] == (
        "TERMINAL_V4M3_MODEL_RESCUE_V4_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v2["contract_status"] == (
        "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    )


def test_v7m1_controller_waits_for_frozen_setup_and_never_scores() -> None:
    controller = ROOT / "scripts/reactflow_delta/run_model_rescue_v7_cache_controller.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")

    assert "EXPECTED_WEIGHT_BYTES=2603787622" in text
    assert "POLL_SECONDS=900" in text
    assert 'runtime_v7_clean/bin/python' in text
    assert "runtime_setup_complete" in text
    assert "weight_download_complete" in text
    assert "--max-constructs 2" in text
    assert "--expected-constructs 160" in text
    assert "--expected-mutants 13976" in text
    assert "build_model_rescue_v7_dependency_cache.py" in text
    assert "qualify_model_rescue_v7_dependency_cache.py" in text
    assert "score_model_rescue" not in text
    assert "run_model_rescue_v7_probe" not in text


def test_v7_runtime_recovery_builds_a_clean_frozen_environment() -> None:
    recovery = ROOT / "scripts/reactflow_delta/recover_model_rescue_v7_runtime.sh"
    subprocess.run(["bash", "-n", str(recovery)], check=True)
    text = recovery.read_text(encoding="utf-8")

    assert "pip=23.3" in text
    assert "setuptools=68.2.2" in text
    assert "wheel=0.41.2" in text
    assert "runtime_v7_clean" in text
    assert "pytorch=2.1.0=py3.11_cuda11.8_cudnn8.7.0_0" in text
    assert "pytorch-cuda=11.8" in text
    assert "cuda-nvcc=11.8" in text
    assert "numpy=1.24.4" in text
    assert "pandas=2.0.3" in text
    assert "h5py=3.9.0" in text
    assert "pyyaml=6.0.1" in text
    assert "packaging==23.2" in text
    assert "ninja==1.11.1.1" in text
    assert "einops==0.6.1" in text
    assert "ml-collections==0.1.1" in text
    assert "gdown==5.1.0" in text
    assert "flash_attn-2.3.2-cp311-cp311-linux_x86_64.whl" in text
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in text
    assert "pip wheel" in text
    assert '--wheel-dir "$BASE/wheels"' in text
    assert "--no-build-isolation" in text
    assert 'pip install --no-deps "$FLASH_WHEEL"' in text
    assert "CONDA_PKGS_DIRS" in text
    assert 'PIP_CACHE_DIR="$BASE/pip_cache"' in text
    assert 'TMPDIR="$BASE/tmp"' in text
    assert "TORCH_CUDA_ARCH_LIST=8.0" in text
    assert "runtime_setup_complete" in text
    assert "score_model_rescue" not in text
