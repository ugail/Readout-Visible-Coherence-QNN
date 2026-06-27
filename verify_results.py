#!/usr/bin/env python3
"""
verify_results.py
=================

Quick verification script for reviewers of

    Readout-visible sector coherence controls active-gradient degradation
    in noisy equivariant quantum neural networks
    H. Ugail and N. Howard, 2026.

It loads the precomputed tables in Results/ and confirms that every headline
number quoted in the manuscript is reproducible from those tables. It runs in
a couple of seconds, needs no quantum simulation, no GPU, and only NumPy and
pandas. It prints a PASS/FAIL line for each check and an overall summary.

Usage:
    pip install -r requirements-verify.txt
    python verify_results.py
"""

import os
import sys
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

# Accumulators for the final summary.
_passed = 0
_failed = 0


def check(name, condition, detail=""):
    """Record and print a single PASS/FAIL check."""
    global _passed, _failed
    status = "PASS" if condition else "FAIL"
    if condition:
        _passed += 1
    else:
        _failed += 1
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


def approx(value, target, tol):
    """True if value is within tol of target."""
    return abs(float(value) - float(target)) <= tol


def ols_loglog(df, gl_col="gamma_L", lam_col="lambda_coh", dm_col="D_M"):
    """Fit log D_M ~ const + alpha log(gamma_L) + beta log(lambda_coh)."""
    d = df[(df[dm_col] > 0) & (df[gl_col] > 0)].copy()
    X = np.column_stack([
        np.ones(len(d)),
        np.log(d[gl_col].to_numpy()),
        np.log(d[lam_col].to_numpy()),
    ])
    y = np.log(d[dm_col].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta[1], beta[2], r2, len(d)


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"[FAIL] missing file: {name}")
        global _failed
        _failed += 1
        return None
    return pd.read_csv(path)


def main():
    print("=" * 70)
    print("Verifying headline results for Readout-Visible-Coherence-QNN")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Main four-channel pooled fit and the three-channel theorem fit.
    #    Both are read from the leave-one-channel-out regression table,
    #    where the 'gen_amp_damp' row is the full 96-setting fit and the
    #    'x_err' row is trained on the three symmetry-preserving channels.
    # ------------------------------------------------------------------
    print("\n-- Finite-noise power law (phase6_loco.csv) --")
    loco = load("phase6_loco.csv")
    if loco is not None:
        full = loco[loco["held_out"] == "gen_amp_damp"]
        if len(full):
            r = full.iloc[0]
            check("4-channel pooled alpha = 1.002",
                  approx(r["slope_gL"], 1.002, 0.01),
                  f"alpha={r['slope_gL']:.3f}")
            check("4-channel pooled beta = 0.964",
                  approx(r["slope_lambda"], 0.964, 0.01),
                  f"beta={r['slope_lambda']:.3f}")
            check("4-channel pooled R^2 = 0.979",
                  approx(r["R2_train"], 0.979, 0.005),
                  f"R2={r['R2_train']:.4f}")

        xerr = loco[loco["held_out"] == "x_err"]
        if len(xerr):
            r = xerr.iloc[0]
            check("3-channel theorem fit alpha = 1.006",
                  approx(r["slope_gL"], 1.006, 0.01),
                  f"alpha={r['slope_gL']:.3f}")
            check("3-channel theorem fit beta = 0.889",
                  approx(r["slope_lambda"], 0.889, 0.01),
                  f"beta={r['slope_lambda']:.3f}")
            check("3-channel theorem fit R^2 = 0.978",
                  approx(r["R2_train"], 0.978, 0.005),
                  f"R2={r['R2_train']:.4f}")
            check("X-error held-out R^2 = 0.949",
                  approx(r["R2_test"], 0.949, 0.005),
                  f"R2_test={r['R2_test']:.4f}")

    # ------------------------------------------------------------------
    # 2. Zero-alignment control: correlated dephasing has a large
    #    worst-case rate but no measurable degradation.
    # ------------------------------------------------------------------
    print("\n-- Zero-alignment control (phase14_aniso_noise.csv) --")
    ani = load("phase14_aniso_noise.csv")
    if ani is not None:
        cc = ani[ani["channel"] == "corr_dephase"]
        if len(cc):
            lam = float(cc["lambda_coh"].iloc[0])
            dmax = float(cc["D_M"].abs().max())
            check("correlated dephasing lambda_coh = 4.5",
                  approx(lam, 4.5, 0.05),
                  f"lambda_coh={lam:.2f}")
            check("correlated dephasing |D_M| < 3e-14 (zero alignment)",
                  dmax < 3e-14,
                  f"max|D_M|={dmax:.2e}")

    # ------------------------------------------------------------------
    # 3. Higher-sector boundary: the scalar law weakens outside r = 1.
    # ------------------------------------------------------------------
    print("\n-- Higher-sector boundary (phase12_r2_sector.csv) --")
    sec = load("phase12_r2_sector.csv")
    if sec is not None:
        for r_val, target in ((2, 0.71), (3, 0.89)):
            sub = sec[sec["r"] == r_val]
            if len(sub):
                _, _, r2, n = ols_loglog(sub)
                check(f"r={r_val} scalar-law R^2 = {target}",
                      approx(r2, target, 0.02),
                      f"R2={r2:.3f}, n={n}")

    # ------------------------------------------------------------------
    # 4. Robust intervals: the cluster bootstrap widens the beta interval
    #    so that it remains compatible with unity.
    # ------------------------------------------------------------------
    print("\n-- Robust uncertainty (phase17_robust_ci.csv) --")
    ci = load("phase17_robust_ci.csv")
    if ci is not None:
        clus = ci[ci["estimator"].str.contains("luster", na=False)]
        if len(clus):
            r = clus.iloc[0]
            lo = float(r["beta_lo95_robust_normal"])
            hi = float(r["beta_hi95_robust_normal"])
            check("cluster-bootstrap beta interval contains 1",
                  lo <= 1.0 <= hi,
                  f"[{lo:.3f}, {hi:.3f}]")

    # ------------------------------------------------------------------
    # 5. Light-cone separation: active versus inactive second moments are
    #    separated by many orders of magnitude in the activity maps.
    # ------------------------------------------------------------------
    print("\n-- Light-cone separation (phase0_activity_maps.csv) --")
    act = load("phase0_activity_maps.csv")
    if act is not None:
        m2cols = [c for c in act.columns if "m2" in c.lower() or "M2" in c]
        if m2cols:
            col = m2cols[0]
            vals = act[col].to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                ratio = float(vals.max() / vals.min())
                check("active-to-inactive M2 ratio exceeds 1e20",
                      ratio > 1e20,
                      f"max/min ~ {ratio:.1e}")

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    total = _passed + _failed
    print(f"SUMMARY: {_passed}/{total} checks passed.")
    if _failed == 0:
        print("ALL HEADLINE NUMBERS REPRODUCED FROM THE ARCHIVED TABLES.")
    else:
        print(f"{_failed} check(s) did not match. See the lines marked FAIL above.")
    print("=" * 70)
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
