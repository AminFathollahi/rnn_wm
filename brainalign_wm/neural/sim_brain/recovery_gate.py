"""Simulated-spike geometry-recovery gate. Must pass before any real neural
data is analyzed. Runs the analysis pipeline in full (rates, crossnobis RDM,
representational similarity analysis with a noise ceiling, demixed PCA, and
cross-temporal decoding) against `SimulatedBrain` and checks that the
planted representational structure is recovered with the correct sign,
while a scrambled control is not (establishing both sensitivity and
specificity).

A model with matching planted geometry is operationalized as a second,
independent `SimulatedBrain` draw: a different seed produces different
units and spike noise but identical generative parameters and planted
geometry, the simulated analogue of two subjects performing the same
underlying computation. Its representational-similarity alignment to the
first draw should sit near that draw's own noise ceiling; alignment to the
scrambled control should not.

Every check below prints its observed value against its threshold, and a
failure causes the script to exit non-zero with a specific diagnostic --
this is a validation gate, not a status report, and it does not catch and
suppress its own failures.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import yaml


def _log(msg: str) -> None:
    print(f"[recovery_gate] {msg}", flush=True)


def run(cfg: dict) -> bool:
    from brainalign_wm.neural.sim_brain.spiking_generator import SimulatedBrain, make_scrambled
    from brainalign_wm.analysis.rsa import (
        _session_condition_rdm,
        compare_rdms,
        noise_ceiling_from_dataset,
        normalized_alignment,
    )
    from brainalign_wm.analysis.dpca import dpca_components
    from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding, stability_index

    sim_cfg = cfg["sim_brain"]
    ok = True

    t0 = time.time()
    _log("building SimulatedBrain (seed=0, real) ...")
    brain0 = SimulatedBrain(sim_cfg, seed=sim_cfg.get("seed", 0))
    _log(f"  done in {time.time()-t0:.1f}s: {len(brain0.units())} units, {len(brain0.trials())} trials")

    t0 = time.time()
    _log("building SimulatedBrain (seed=1, real, 'matching-geometry model') ...")
    brain1 = SimulatedBrain(sim_cfg, seed=sim_cfg.get("seed", 0) + 1)
    _log(f"  done in {time.time()-t0:.1f}s")

    t0 = time.time()
    _log("building scrambled control ...")
    brain_scrambled = make_scrambled(sim_cfg, seed=sim_cfg.get("seed", 0))
    _log(f"  done in {time.time()-t0:.1f}s")

    epoch = "maintain"
    region = None  # single pseudo-region "sim"

    # ---- 1. noise ceiling sanity ----
    t0 = time.time()
    lower, upper = noise_ceiling_from_dataset(brain0, region, epoch)
    _log(f"noise ceiling (brain0, {epoch}): lower={lower:.3f} upper={upper:.3f}  [{time.time()-t0:.1f}s]")
    check = 0.0 <= lower <= upper <= 1.0 + 1e-6 and upper > 0.1
    ok &= check
    _log(f"  CHECK bounds sane & upper>0.1: {'PASS' if check else 'FAIL'}")

    # ---- 2. RSA: matching-geometry alignment near ceiling; scrambled far below ----
    rdm0, conds0 = _session_condition_rdm(brain0, brain0.sessions()[0], region, epoch, brain0.bin_ms)
    # pool across ALL sessions' response patterns for a stabler dataset-level RDM comparison
    from brainalign_wm.analysis.rdm import crossnobis_rdm

    def dataset_rdm(ds):
        patt = ds.response_patterns(region, epoch)  # [n_cond, n_units]
        labels = [c.coarse_key() for c in ds.conditions]
        rdm, _ = crossnobis_rdm(patt, labels, n_folds=min(4, patt.shape[0]))
        return rdm

    rdm_brain0 = dataset_rdm(brain0)
    rdm_brain1 = dataset_rdm(brain1)
    rdm_scrambled = dataset_rdm(brain_scrambled)

    align_matching = compare_rdms(rdm_brain0, rdm_brain1)
    align_scrambled = compare_rdms(rdm_brain0, rdm_scrambled)
    norm_matching = normalized_alignment(align_matching, upper)
    _log(f"RSA alignment brain0<->brain1 (matching geometry): raw={align_matching:.3f} "
         f"norm={norm_matching:.3f} (ceiling upper={upper:.3f})")
    _log(f"RSA alignment brain0<->scrambled: raw={align_scrambled:.3f}")
    check_matching = norm_matching >= 0.5
    check_scrambled = align_scrambled < 0.3
    check_specificity = align_matching > align_scrambled + 0.2
    ok &= check_matching and check_scrambled and check_specificity
    _log(f"  CHECK matching-geometry norm>=0.5: {'PASS' if check_matching else 'FAIL'}")
    _log(f"  CHECK scrambled raw<0.3: {'PASS' if check_scrambled else 'FAIL'}")
    _log(f"  CHECK matching >> scrambled (margin>0.2): {'PASS' if check_specificity else 'FAIL'}")

    # ---- 3. dPCA: load marginalization recovers the planted load axis ----
    def load_marginalization_strength(ds) -> float:
        # NOTE: with a single factor, its marginalization trivially captures
        # ~100% of "total variance" (there's nothing else to marginalize out
        # against) -- both real and scrambled data would score 1.0, which is
        # not a real check. Cross `load` with `item` so the load fraction is
        # measured against a nontrivial multi-factor total (real: load
        # should carry a genuine share; scrambled: near-zero on both).
        trials = ds.trials()
        rates = ds.rates(region, ds.bin_ms, [epoch])  # [n_units, n_trials, n_bins]
        loads = sorted(trials.load.unique())
        items = sorted(trials.item_id.unique())
        n_units = rates.shape[0]
        n_bins = rates.shape[2]
        R = np.zeros((n_units, len(loads), len(items), n_bins))
        for li, load in enumerate(loads):
            for ii, item in enumerate(items):
                mask = ((trials.load == load) & (trials.item_id == item)).values
                if mask.sum() > 0:
                    R[:, li, ii, :] = rates[:, mask, :].mean(axis=1)
        # "time" must be declared as its own factor (R's last axis) so
        # condition-independent time-locked variance (e.g. onset transients)
        # gets its own marginalization instead of leaking equally into both
        # "load" and "item" and masking the true per-factor signal.
        comp = dpca_components(R, ["load", "item", "time"], n_components=1)
        return comp[("load",)]["fraction_of_total_variance"], comp[("item",)]["fraction_of_total_variance"]

    t0 = time.time()
    load_frac_real, item_frac_real = load_marginalization_strength(brain0)
    load_frac_scrambled, item_frac_scrambled = load_marginalization_strength(brain_scrambled)
    _log(f"dPCA load-marginalization fraction-of-variance: real={load_frac_real:.3f} "
         f"scrambled={load_frac_scrambled:.3f}  [{time.time()-t0:.1f}s]")
    _log(f"dPCA item-marginalization fraction-of-variance: real={item_frac_real:.3f} "
         f"scrambled={item_frac_scrambled:.3f}")
    check_dpca = (load_frac_real > load_frac_scrambled + 0.03) and (item_frac_real > item_frac_scrambled + 0.03)
    ok &= check_dpca
    _log(f"  CHECK real load- and item-marginalization > scrambled (+0.03 margin each): {'PASS' if check_dpca else 'FAIL'}")

    # ---- 4. cross-temporal decoding: item identity decodable above chance in real, not in scrambled ----
    def item_decode_diag_acc(ds) -> float:
        trials = ds.trials()
        rates = ds.rates(region, ds.bin_ms, [epoch])  # [n_units, n_trials, n_bins]
        X = rates.transpose(1, 2, 0)  # [n_trials, n_bins, n_units]
        y = trials.item_id.values
        mat = cross_temporal_decoding(X, y, n_folds=3, seed=0)
        return float(np.nanmean(np.diag(mat)))

    t0 = time.time()
    acc_real = item_decode_diag_acc(brain0)
    acc_scrambled = item_decode_diag_acc(brain_scrambled)
    chance = 1.0 / brain0.item_dim
    _log(f"cross-temporal item-decoding diagonal accuracy: real={acc_real:.3f} "
         f"scrambled={acc_scrambled:.3f} chance={chance:.3f}  [{time.time()-t0:.1f}s]")
    check_decode = acc_real > chance + 0.1 and acc_scrambled < chance + 0.15
    ok &= check_decode
    _log(f"  CHECK real decodable above chance (+0.1), scrambled ~chance: {'PASS' if check_decode else 'FAIL'}")

    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))

    passed = run(cfg)
    if passed:
        _log("RECOVERY GATE: PASS -- pipeline recovers planted geometry and rejects the scrambled control.")
        return 0
    else:
        _log("RECOVERY GATE: FAIL -- see CHECK lines above for which comparison(s) failed and by how much. "
             "This is a result to report, not to work around: do not proceed to alignment analysis on "
             "real neural data until this gate passes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
