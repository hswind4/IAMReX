#!/usr/bin/env python3
"""
Acceptance test for RKPM mapping files (rkpm_mappings.id / rkpm_mappings.lag).

It checks the two discrete conservation conditions that the diffused-IB method
relies on, evaluated IN THE SOLVER'S grid frame (cell centers at (n+0.5)*dx from
the global origin 0). These are the quantities that actually govern force and
torque conservation when the solver applies the weights:

    (0th moment)  Sum_c w_c            == 1     -> conserves total force
    (1st moment)  Sum_c w_c (x_c - x_l) == 0     -> conserves torque (no spurious
                                                    force dipole / torque)

The classic failure mode is a constant nonzero 1st moment caused by the RKPM
generator building its Euler grid on an origin that is not an integer multiple
of dx (int(sx/dx) truncation in mapping.py). That shows up here as a per-marker
1st moment that is (a) nonzero and (b) IDENTICAL across all markers. After the
grid-alignment fix it should drop to ~1e-12 cells.

Usage:
    python verify_rkpm_alignment.py [dx] [id_file] [lag_file]

    dx        finest-level cell size. Default = 6/(256*8) for the
              flow_past_ellipsoid case (prob length 6, n_cell 256, max_level 3).
    id_file   default rkpm_mappings.id
    lag_file  default rkpm_mappings.lag

Exit code 0 if PASS, 1 if FAIL.
"""

import sys
import re
import ast
import numpy as np

TOL_SUMW = 1e-9          # |Sum w - 1|
TOL_MOMENT_CELLS = 1e-6  # |Sum w (x_c - x_l)| / dx  (cells)


def load_id(path):
    with open(path) as f:
        d = ast.literal_eval(f.read())
    n = len(d)
    pts = np.zeros((n, 3))
    for k, v in d.items():
        pts[int(k)] = v
    return pts


def load_lag(path):
    txt = open(path).read()
    blocks = re.findall(r'(\d+)\s*:\s*\[(.*?)\]', txt, re.S)
    out = {}
    for bid, body in blocks:
        entries = re.findall(r'\{([^}]*)\}', body)
        rows = []
        for e in entries:
            kv = dict(re.findall(r'"(\w+)"\s*:\s*([-\d.eE]+)', e))
            rows.append((int(kv['i']), int(kv['j']), int(kv['k']),
                         float(kv['w']), float(kv['Vcell']), float(kv['eps'])))
        out[int(bid)] = np.array(rows, dtype=float)
    return out


def main():
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0 / (256 * 8)
    id_file = sys.argv[2] if len(sys.argv) > 2 else 'rkpm_mappings.id'
    lag_file = sys.argv[3] if len(sys.argv) > 3 else 'rkpm_mappings.lag'

    print(f"dx = {dx:.10g}")
    print(f"id  = {id_file}")
    print(f"lag = {lag_file}\n")

    pts = load_id(id_file)
    lag = load_lag(lag_file)
    N = len(lag)
    print(f"markers: {N} (id), {len(pts)} (lag)")

    sizes = np.array([len(lag[b]) for b in lag])
    print(f"stencil size: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.2f}")

    sumw = np.zeros(N)
    moment = np.zeros((N, 3))   # in cells
    for n, b in enumerate(sorted(lag.keys())):
        a = lag[b]
        I, J, K, W = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        xc = (I + 0.5) * dx
        yc = (J + 0.5) * dx
        zc = (K + 0.5) * dx
        xl, yl, zl = pts[b]
        sumw[n] = W.sum()
        moment[n, 0] = np.sum(W * (xc - xl)) / dx
        moment[n, 1] = np.sum(W * (yc - yl)) / dx
        moment[n, 2] = np.sum(W * (zc - zl)) / dx

    err_sumw = np.abs(sumw - 1.0).max()
    abs_moment = np.abs(moment)
    max_moment = abs_moment.max(axis=0)
    mean_moment = moment.mean(axis=0)
    std_moment = moment.std(axis=0)

    print()
    print(f"[0th moment]  max|Sum w - 1|          = {err_sumw:.3e}   "
          f"(tol {TOL_SUMW:.0e})")
    print(f"[1st moment]  max|Sum w (x-x_l)|/dx   = "
          f"({max_moment[0]:.3e}, {max_moment[1]:.3e}, {max_moment[2]:.3e}) cells   "
          f"(tol {TOL_MOMENT_CELLS:.0e})")
    print(f"              mean over markers       = "
          f"({mean_moment[0]:+.3e}, {mean_moment[1]:+.3e}, {mean_moment[2]:+.3e}) cells")
    print(f"              std  over markers       = "
          f"({std_moment[0]:.3e}, {std_moment[1]:.3e}, {std_moment[2]:.3e}) cells")

    # Diagnose the classic constant-shift (grid misalignment) signature.
    if max_moment.max() > TOL_MOMENT_CELLS and std_moment.max() < 1e-6:
        print()
        print("  >> DIAGNOSIS: 1st moment is a NONZERO CONSTANT across all markers.")
        print("     This is the generator/solver grid-origin misalignment "
              "(int(sx/dx) truncation).")
        print(f"     Effective constant shift = "
              f"({mean_moment[0]:+.3f}, {mean_moment[1]:+.3f}, {mean_moment[2]:+.3f}) cells.")
        print("     Fix: snap sx,sy,sz to integer multiples of dx in main.py "
              "(and dx = Lx/nxc must equal the solver finest dx).")

    ok = (err_sumw < TOL_SUMW) and (max_moment.max() < TOL_MOMENT_CELLS)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
