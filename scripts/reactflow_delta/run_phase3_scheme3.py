#!/usr/bin/env python3
"""Phase 3 scheme-3 (contract §9.4): repaired EPRO propagation operator, nested
leave-one-publication-out, compared against the same-capacity scheme-2 generic.

Variants trained per fold on the TRUE CHANGERS (caller_v3):
  * epro        : repaired EPRO propagation over sparse base-pair contacts
  * epro_local  : EPRO with propagation DISABLED (no-propagation identity) —
                  isolates the nonlocal-propagation increment
  * epro_random : EPRO with PERMUTED (random) contacts — verifies that real
                  contacts do not merely give any-graph gain
  * generic     : same-capacity scheme-2 concat baseline ([WT,Mut,cond])
  * trivial     : train-fold weighted-mean constant (endpoint_v5 baseline)

Metric (endpoint_v5): conditional WMAE skill vs trivial. Scheme-3 acceptance =
the repaired EPRO beats the same-capacity generic with paired publication-block
bootstrap CI lower bound > 0 across 5 seeds, random contacts give NO equal gain,
and the propagation ablation degrades in the pre-registered direction.

All neural models run on CUDA; CUDA-unavailable => STOP (no silent CPU fallback).
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import (  # noqa: E402
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned,
    require_cuda, edited_index, _base_oh, _norm_react, _norm_err, BASE_MAP,
)
from caller_v3 import CallerV3  # noqa: E402
from evaluate_v5 import conditional_wmae_skill, paired_bootstrap_skill_ci  # noqa: E402
from samplers import publication_folds, pair_magnitude  # noqa: E402
from models.epro_v1 import (
    EPROMagnitude, GLOB_DIM, POS_DIM, base_oh,
)
from models.pair_v1 import CapacityMatchedMLP, count_params  # noqa: E402
from models.pair_v2 import _condition_feat, build_scheme2_features  # noqa: E402
from train_v2 import train_flat, predict_flat, _assert_cuda  # noqa: E402
from run_phase3 import paired_skill_diff_ci  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TARGET_PARAMS = 11777
EPOCHS = 30
LR = 1e-3
BS = 128


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def build_fullseq_features(pair, wt_rec):
    """Full-length per-position features (L, POS_DIM): allowed WT inputs only."""
    seq = wt_rec.get("canonical_sequence") or ""
    rl = wt_rec.get("reactivity_layers", {})
    tf = rl.get("train_frozen", {}) or rl.get("raw", {})
    react = np.nan_to_num(np.asarray(tf.get("reactivity") or [], dtype=np.float32), nan=0.0)
    err = np.nan_to_num(np.asarray(tf.get("error") or [], dtype=np.float32), nan=0.0)
    n = len(seq)
    X = np.zeros((n, POS_DIM), dtype=np.float32)
    for i in range(n):
        X[i] = np.concatenate([_base_oh(seq[i]),
                               [_norm_react(react[i] if i < len(react) else 0.0),
                                _norm_err(err[i] if i < len(err) else 0.0)]])
    return X, seq


def build_glob(pair, wt_rec, n):
    """(GLOB_DIM,) condition + edit-position feature (allowed inputs)."""
    cond = _condition_feat(pair, wt_rec)  # (31,)
    ei = edited_index(pair)
    extra = np.array([float(ei) / max(n, 1), 1.0, 1.0], dtype=np.float32)
    return np.concatenate([cond, extra]).astype(np.float32)


def make_epro_batches(pids, feat, contacts, device, seed=0,
                      random_contacts=False, y_w=True):
    """Group pairs by sequence into same-graph batches for the EPRO.

    Pairs sharing the same WT sequence share the same contact graph, so we
    batch them together (same L, same edges) for efficient GPU training.
    """
    rng = np.random.RandomState(seed)
    by_seq = {}
    for pid in pids:
        by_seq.setdefault(feat[pid]["seq_hash"], []).append(pid)

    batches = []
    ordered_pids = []
    for seq_h, pids_g in by_seq.items():
        cont = contacts[seq_h]
        n_g = len(pids_g)
        B = n_g
        L = feat[pids_g[0]]["L"]
        X = np.stack([feat[pid]["X"] for pid in pids_g]).astype(np.float32)  # (B,L,d)
        mask = np.ones((B, L), dtype=np.float32)
        elig = np.stack([feat[pid]["elig"] for pid in pids_g]).astype(np.float32)
        edit = np.array([feat[pid]["edit"] for pid in pids_g], dtype=np.int64)
        ref = np.stack([feat[pid]["ref"] for pid in pids_g]).astype(np.float32)
        alt = np.stack([feat[pid]["alt"] for pid in pids_g]).astype(np.float32)
        glob = np.stack([feat[pid]["glob"] for pid in pids_g]).astype(np.float32)
        if cont["n_edges"] > 0:
            edges = np.stack([cont["edges"]] * B)  # (B,E,2)
            w = np.stack([cont["weights"]] * B)    # (B,E)
            if random_contacts:
                perm = rng.permutation(cont["edges"][:, 1])
                edges = np.stack([cont["edges"][:, 0], perm]).transpose(1, 0)
                edges = np.stack([edges] * B)
        else:
            edges = np.zeros((B, 0, 2), dtype=np.int64)
            w = np.zeros((B, 0), dtype=np.float32)
        bt = {
            "X": torch.from_numpy(X).float().to(device),
            "mask": torch.from_numpy(mask).float().to(device),
            "elig": torch.from_numpy(elig).float().to(device),
            "edit": torch.from_numpy(edit).long().to(device),
            "ref": torch.from_numpy(ref).float().to(device),
            "alt": torch.from_numpy(alt).float().to(device),
            "edges": torch.from_numpy(edges).long().to(device),
            "weights": torch.from_numpy(w).float().to(device),
            "glob": torch.from_numpy(glob).float().to(device),
            "L": L, "seq_hash": seq_h, "pids": pids_g,
        }
        if y_w:
            bt["y"] = torch.tensor([feat[pid]["y"] for pid in pids_g],
                                   dtype=torch.float32, device=device)
            bt["w"] = torch.tensor([feat[pid]["w"] for pid in pids_g],
                                   dtype=torch.float32, device=device)
        batches.append(bt)
        ordered_pids.extend(pids_g)
    return batches, ordered_pids


def train_epro(model, batches, device, seed=0):
    _assert_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    n = len(batches)
    for _ in range(EPOCHS):
        order = list(range(n))
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
        for bi in order:
            bt = batches[bi]
            pred = model(bt["X"], bt["mask"], bt["elig"], bt["edit"], bt["ref"],
                         bt["alt"], bt["edges"], bt["weights"], bt["glob"])
            loss = (bt["w"] * (pred - bt["y"]).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_epro(model, batches, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for bt in batches:
            p = model(bt["X"], bt["mask"], bt["elig"], bt["edit"], bt["ref"],
                      bt["alt"], bt["edges"], bt["weights"], bt["glob"])
            preds.append(p.cpu().numpy())
    # flatten per-pair predictions across sequence-grouped batches (order matches
    # the pids the batches were built from, i.e. the changers list order)
    return np.concatenate(preds).astype(np.float32)


def fit_variant(variant, model, Xtr, ytr, wtr, tr_batches, Xte, te_batches,
                seed, device):
    if variant == "trivial":
        c = float(np.sum(np.asarray(wtr) * np.asarray(ytr)) / max(np.sum(wtr), 1e-12))
        return np.full(Xte.shape[0], c, dtype=np.float32)
    if variant == "generic":
        m = CapacityMatchedMLP(Xtr.shape[1], TARGET_PARAMS, seed=seed)
        train_flat(m, Xtr, ytr, wtr, device, seed)
        return predict_flat(m, Xte, device).astype(np.float32)
    # EPRO variants
    m = EPROMagnitude(hidden=64, glob_dim=GLOB_DIM,
                      neumann_iter=8, seed=seed).to(device)
    train_epro(m, tr_batches, device, seed)
    return predict_epro(m, te_batches, device).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="3")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after CUDA_VISIBLE_DEVICES=" + args.cuda_device +
                           ". Contract: STOP, no silent CPU fallback.")
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[scheme3] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import yaml
    split = yaml.safe_load(Path(args.split_yaml).read_text(encoding="utf-8"))
    pub_map = split["publication_map"]
    study_roles = split["study_roles"]

    with open(args.cache, "rb") as fh:
        cache = pickle.load(fh)
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"])
    test_studies = {s for s, r in study_roles.items() if r == "test"}
    pool_studies = pool_studies - test_studies

    with open(args.contacts, "rb") as fh:
        cdata = pickle.load(fh)
    contacts = cdata["contacts"]
    # map seq -> seq_hash
    seq_hash_of = {}
    for h, v in contacts.items():
        seq_hash_of[v["seq"]] = h

    pair_recs = {}
    for p in pairs:
        if _study_of(p["source_accession"]) in test_studies:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pair_recs[p["source_accession"] + ":" + str(p["mutant_profile_index"])] = {
            "pair": p, "wt": wt, "mut": mu,
            "study": _study_of(p["source_accession"]),
            "pub": pub_map.get(_study_of(p["source_accession"]), "UNKNOWN"),
        }
    print(f"[scheme3] pool_studies={sorted(pool_studies)} n_pairs_usable={len(pair_recs)}", flush=True)

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    caller_seed = 20260807

    # Precompute fold-invariant EPRO features + generic features + contacts
    pf_all = {pid: build_pair_features_aligned(pr["pair"], pr["wt"], pr["mut"])
              for pid, pr in pair_recs.items()}
    feat = {}
    fx_generic = {}
    missing_contact = 0
    for pid, pr in pair_recs.items():
        X, seq = build_fullseq_features(pr["pair"], pr["wt"])
        seq_h = seq_hash_of.get(seq)
        if seq_h is None or seq_h not in contacts:
            missing_contact += 1
            continue
        pf = pf_all[pid]
        L = len(seq)
        elig = np.zeros(L, dtype=np.float32)
        n_elig = min(L, len(pf.eligibility_mask))
        for i in range(n_elig):
            elig[i] = 1.0 if pf.eligibility_mask[i] else 0.0
        mval, wval = pair_magnitude(pf)
        feat[pid] = {
            "X": X, "L": L, "seq_hash": seq_h,
            "edit": edited_index(pr["pair"]),
            "ref": base_oh(pr["pair"].get("ref_allele")),
            "alt": base_oh(pr["pair"].get("alt_allele")),
            "glob": build_glob(pr["pair"], pr["wt"], L),
            "elig": elig,
            "mag": mval if mval is not None else 0.0,
            "w": wval,
        }
        fx_generic[pid] = build_scheme2_features(pr["pair"], pr["wt"], False, True)
    print(f"[scheme3] precomputed features for {len(feat)} pairs, missing_contact={missing_contact}",
          flush=True)

    VARIANT_ORDER = ["epro", "epro_local", "epro_random", "generic", "trivial"]
    heldout = {v: {s: {"pub": [], "y": [], "w": [], "pred": []} for s in SEEDS}
               for v in VARIANT_ORDER}
    fold_info = {}

    for fold in publication_folds(pubs, pair_recs, pub_study):
        held_pub = fold["held_pub"]
        t0 = time.time()
        train_groups = rep_groups_for_train(all_rep_groups, fold["train_studies"])
        caller = CallerV3(seed=caller_seed).fit(train_groups, [], noise_replicate_groups=all_rep_groups)
        labels = {}
        mags = {}
        for pid, pr in pair_recs.items():
            lab = caller.call(pf_all[pid]).label
            labels[pid] = lab
            if lab == "1":
                mval, wval = pair_magnitude(pf_all[pid])
                mags[pid] = (mval, wval) if mval is not None else (None, 0)
        tr_ch = [pid for pid in fold["train_pids"] if pid in feat
                 and labels.get(pid) == "1" and mags.get(pid, (None, 0))[1] > 0]
        he_ch = [pid for pid in fold["held_pids"] if pid in feat
                 and labels.get(pid) == "1" and mags.get(pid, (None, 0))[1] > 0]
        fold_info[held_pub] = {"n_train_changers": len(tr_ch), "n_held_changers": len(he_ch)}
        if not tr_ch or not he_ch:
            print(f"[fold] held={held_pub} SKIP (train_ch={len(tr_ch)} held_ch={len(he_ch)})",
                  flush=True)
            continue

        # assign mag target into feat
        for pid in tr_ch + he_ch:
            feat[pid]["y"] = mags[pid][0]
            feat[pid]["w"] = mags[pid][1]

        ytr = np.array([mags[pid][0] for pid in tr_ch], dtype=np.float32)
        wtr = np.array([mags[pid][1] for pid in tr_ch], dtype=np.float32)
        yte = np.array([mags[pid][0] for pid in he_ch], dtype=np.float32)
        wte = np.array([mags[pid][1] for pid in he_ch], dtype=np.float32)
        Xtr = np.stack([fx_generic[pid] for pid in tr_ch])
        Xte = np.stack([fx_generic[pid] for pid in he_ch])

        def _zero_local(batches):
            for bt in batches:
                bt["edges"] = torch.zeros(1, 0, 2, dtype=torch.long, device=device)
                bt["weights"] = torch.zeros(1, 0, dtype=torch.float32, device=device)

        for v in VARIANT_ORDER:
            if v in ("epro", "epro_local", "epro_random"):
                for seed in SEEDS:
                    tr_bv, _ = make_epro_batches(tr_ch, feat, contacts, device, seed=seed,
                                                 random_contacts=(v == "epro_random"))
                    te_bv, te_pids = make_epro_batches(he_ch, feat, contacts, device, seed=seed,
                                                       random_contacts=(v == "epro_random"))
                    if v == "epro_local":
                        _zero_local(tr_bv)
                        _zero_local(te_bv)
                    pred = fit_variant(v, None, Xtr, ytr, wtr, tr_bv, Xte, te_bv, seed, device)
                    h = heldout[v][seed]
                    h["pub"].extend([held_pub] * len(pred))
                    h["y"].extend([mags[pid][0] for pid in te_pids])
                    h["w"].extend([mags[pid][1] for pid in te_pids])
                    h["pred"].extend(np.clip(pred, 0.0, None).tolist())
            else:
                for seed in SEEDS:
                    pred = fit_variant(v, None, Xtr, ytr, wtr, [], Xte, [], seed, device)
                    h = heldout[v][seed]
                    h["pub"].extend([held_pub] * len(pred))
                    h["y"].extend(yte.tolist())
                    h["w"].extend(wte.tolist())
                    h["pred"].extend(np.clip(pred, 0.0, None).tolist())
        fold_info[held_pub]["seconds"] = round(time.time() - t0, 1)
        print(f"[fold] held={held_pub} train_ch={len(tr_ch)} held_ch={len(he_ch)} "
              f"t={time.time()-t0:.1f}s", flush=True)

    # ---- evaluate conditional WMAE skill per variant x seed ----
    table = {}
    for v in VARIANT_ORDER:
        for seed in SEEDS:
            h = heldout[v][seed]
            triv = heldout["trivial"][seed]
            skill_res = conditional_wmae_skill(h["pub"], h["y"], h["w"], h["pred"], triv["pred"])
            ci = paired_bootstrap_skill_ci(h["pub"], h["y"], h["w"], h["pred"], triv["pred"],
                                           seed=seed, n_boot=1000)
            table[(v, seed)] = {"skill": skill_res.get("skill"),
                                "wmae_model": skill_res.get("wmae_model"),
                                "wmae_baseline": skill_res.get("wmae_baseline"),
                                "n_changers": skill_res.get("n_changers"),
                                "n_publications": skill_res.get("n_publications"),
                                "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high")}
    print("\n[scheme3] skill by variant x seed:", flush=True)
    for v in VARIANT_ORDER:
        print(f"  {v}: {[table[(v, s)]['skill'] for s in SEEDS]}", flush=True)

    # ---- scheme-3 acceptance: EPRO beats same-capacity generic ----
    epro_skills = [table[("epro", s)]["skill"] for s in SEEDS]
    generic_skills = [table[("generic", s)]["skill"] for s in SEEDS]
    diff_ci = {}
    for s in SEEDS:
        h = heldout["epro"][s]; g = heldout["generic"][s]; t = heldout["trivial"][s]
        diff_ci[s] = paired_skill_diff_ci(h["pub"], h["y"], h["w"], h["pred"],
                                          g["pred"], t["pred"], seed=s, n_boot=1000)
    diff_ci_low_min = min(d["ci_low"] for d in diff_ci.values()
                          if isinstance(d["ci_low"], (int, float)))
    epro_beats_generic = bool(diff_ci_low_min > 0.0)
    all_identifiable = all(isinstance(x, (int, float)) for x in epro_skills + generic_skills)

    # ablation directions
    local_skills = [table[("epro_local", s)]["skill"] for s in SEEDS]
    random_skills = [table[("epro_random", s)]["skill"] for s in SEEDS]
    mean = lambda xs: float(np.mean([x for x in xs if isinstance(x, (int, float))]))
    propagation_positive = mean(epro_skills) >= mean(local_skills)
    random_no_equal_gain = mean(random_skills) <= mean(epro_skills)

    verdict = {
        "schema": "reactflow_delta.phase3.scheme3.v1",
        "run_id": out.name,
        "authority_epoch": 18,
        "endpoint": "endpoint_v5",
        "phase": "PHASE3-ARCH",
        "scheme": "repaired_epro_v1_nonlocal_propagation",
        "epro_mean_skill": float(np.mean(epro_skills)) if all_identifiable else None,
        "generic_mean_skill": float(np.mean(generic_skills)) if all_identifiable else None,
        "epro_vs_generic_skill_diff_ci_low_min": diff_ci_low_min,
        "epro_beats_same_capacity_generic": epro_beats_generic,
        "ablation": {
            "epro_local_mean_skill": mean(local_skills),
            "epro_random_mean_skill": mean(random_skills),
            "propagation_positive_vs_local": propagation_positive,
            "random_no_equal_gain": random_no_equal_gain,
        },
        "criteria_checks": {
            "estimand_identifiable_all_seeds": all_identifiable,
            "epro_ci_low_gt_0_vs_generic": epro_beats_generic,
            "random_contacts_no_equal_gain": random_no_equal_gain,
            "propagation_ablation_direction": propagation_positive,
            "n_seeds_5": True,
        },
        "per_variant_seed_skill": {v: [table[(v, s)]["skill"] for s in SEEDS]
                                   for v in VARIANT_ORDER},
        "diff_ci_by_seed": {str(s): diff_ci[s] for s in SEEDS},
        "n_distinct_publications": len(pubs),
        "fold_info": fold_info,
        "target_params": TARGET_PARAMS,
        "epro_params": EPROMagnitude(hidden=64, glob_dim=GLOB_DIM).param_count(),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Phase 3 scheme-3 repaired EPRO. Candidate = sparse top-k "
                 "base-pair contact propagation (ViennaRNA BPP), antisymmetric "
                 "vector forcing, Neumann solve with rho<1; vs same-capacity "
                 "scheme-2 generic [WT,Mut,cond]. Ablations: epro_local "
                 "(no propagation), epro_random (permuted contacts)."),
    }
    (out / "phase3_scheme3_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"schema": "reactflow_delta.phase3.scheme3.v1",
                                                  "table": {f"{v}|{s}": table[(v, s)]
                                                            for v in VARIANT_ORDER for s in SEEDS},
                                                  "verdict": verdict},
                                                 indent=2, default=str), encoding="utf-8")
    for v in VARIANT_ORDER:
        for s in SEEDS:
            h = heldout[v][s]
            np.savez_compressed(out / f"heldout_{v}_seed{s}.npz",
                                pub=np.array(h["pub"]), y=np.array(h["y"]),
                                w=np.array(h["w"]), pred=np.array(h["pred"]))
    print(f"[scheme3] wrote verdict -> {out/'phase3_scheme3_verdict.json'}", flush=True)
    print(f"[scheme3] DONE. epro_beats_same_capacity_generic={epro_beats_generic} "
          f"diff_ci_low_min={diff_ci_low_min}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
