import sys
import os
import argparse
import numpy as np
import amrex.space3d as amr

amr.initialize([])

def _select_components(all_names, requested):
    if not requested:
        return list(range(len(all_names)))
    missing = [n for n in requested if n not in all_names]
    if missing:
        raise ValueError(f"Components not found in plotfile: {missing}")
    return [all_names.index(n) for n in requested]

def _resolve_pltfile(path):
    if not os.path.isdir(path):
        return path
    candidates = [
        (int(n[3:]), n) for n in os.listdir(path)
        if n.startswith("plt") and n[3:].isdigit()
    ]
    if not candidates:
        raise ValueError(f"No plt* directories found in {path}")
    return os.path.join(path, sorted(candidates)[-1][1])

parser = argparse.ArgumentParser(description="Convert AMReX plt files to a stacked NPY tensor.")
parser.add_argument("pltfiles", nargs="+", help="plt directories or case folders; folders use the latest plt* inside")
parser.add_argument("-o", "--output", default="dataset.npy", help="output .npy path (default: dataset.npy)")
parser.add_argument("-l", "--level", type=int, default=0, help="AMR level to export (default: 0)")
parser.add_argument("-c", "--components", nargs="*",
                    default=["x_velocity", "y_velocity", "avg_pressure"],
                    help="component names to export (default: x_velocity y_velocity avg_pressure)")
args = parser.parse_args()

samples = []
selected_names = None

for path in args.pltfiles:
    pltfile = _resolve_pltfile(path)
    print(f"Reading {pltfile}")
    pfd = amr.PlotFileData(pltfile)
    if args.level > pfd.finestLevel():
        raise ValueError(f"Requested level {args.level} exceeds finest level {pfd.finestLevel()}")
    mf = pfd.get(args.level)
    var_names = list(pfd.varNames())
    comp_idx = _select_components(var_names, args.components)
    names = [var_names[i] for i in comp_idx]

    # mf[:] shape: (nx, ny, nz, ncomp) for 3-D, or (nx, ny, ncomp) for 2-D
    full_np = mf[:]
    if full_np.ndim == 4:
        full_np = full_np[:, :, :, comp_idx]
        if full_np.shape[2] == 1:
            full_np = full_np[:, :, 0, :]       # squeeze pseudo-2D z dim
        sample = np.transpose(full_np, (2, 0, 1))   # (ncomp, nx, ny)
    elif full_np.ndim == 3:
        full_np = full_np[:, :, comp_idx]
        sample = np.transpose(full_np, (2, 0, 1))   # (ncomp, nx, ny)
    else:
        raise ValueError(f"Unexpected array ndim {full_np.ndim}")

    if selected_names is None:
        selected_names = names
    elif names != selected_names:
        raise ValueError(f"Component mismatch in {pltfile}: {names} vs {selected_names}")
    samples.append(sample)

data = np.stack(samples, axis=0)   # (n_samples, ncomp, nx, ny)
np.save(args.output, data)
print(f"Saved {args.output}  shape={data.shape}  components={selected_names}")

amr.finalize()
