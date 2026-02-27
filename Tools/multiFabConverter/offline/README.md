# multiFabConverter Offline Tools

This directory provides offline tools to read AMReX plot files (`.plt`) and convert them to NumPy arrays or 2D slices, using **pyAMReX** (MPI-enabled).

---

## `main.py` — parallel visualization

### Purpose

Reads an AMReX plot file in a parallel MPI run. AMReX's `DistributionMapping` automatically distributes boxes across ranks. The script:

1. Prints how many boxes each MPI rank owns.
2. Gathers the full domain into a NumPy array via `input_mf[:]` (collective operation across all ranks).
3. On rank 0, extracts a 2D slice of `x_velocity` at a given physical z-coordinate and saves it as `slice_z.png`.

### Dependencies

- Python 3.8+
- `numpy`, `matplotlib`, `mpi4py`
- `pyAMReX` (MPI-enabled build): `conda install -c conda-forge "pyamrex=*=*mpi_openmpi*"`

### Usage

```bash
mpirun -np <N> python main.py <pltfile> [lev] [z_real]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `pltfile` | Path to the AMReX plot file directory (e.g. `plt00002`) | required |
| `lev` | AMR level to read (0 = coarsest) | `0` |
| `z_real` | Physical z-coordinate for the 2D slice | `5.0` |

**Examples:**

```bash
mpirun -np 4 python main.py plt00002
mpirun -np 4 python main.py ../../../Tutorials/FlowPastSphere/plt00002 0 5.0
```

### Output

- **Terminal**: AMReX/MPI initialization info; per-rank box count.
- **File**: `slice_z.png` written by rank 0 in the current directory.

> **Note:** AMReX is initialized with `amr.initialize([])`. Do not pass `sys.argv` to it — AMReX's ParmParse will misinterpret the arguments and abort.

---

## `script_plt_to_npz.py` — batch conversion to NPY

### Purpose

Single-process script (no `mpirun` needed) that reads one or more AMReX plot files, selects specified field components, and stacks them into a NumPy array of shape `(n_samples, ncomp, nx, ny)` saved as a `.npy` file. Intended for machine learning workflows or offline analysis.

### Usage

```bash
python script_plt_to_npz.py <pltfile> [pltfile2 ...] [-o output.npy] [-l lev] [-c comp1 comp2 ...]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `pltfile` | One or more plot file paths, or folders containing `plt*` subdirectories (uses the latest one) | required |
| `-o` | Output `.npy` file path | `dataset.npy` |
| `-l` | AMR level to export | `0` |
| `-c` | Space-separated list of component names to export | `x_velocity y_velocity avg_pressure` |

**Examples:**

```bash
# Single plot file with default components
python script_plt_to_npz.py plt00002

# Multiple plot files, export x_velocity only
python script_plt_to_npz.py plt00001 plt00002 plt00003 -c x_velocity -o uvel.npy

# Folder containing multiple plt* subdirectories (uses the latest)
python script_plt_to_npz.py ../../../Tutorials/LidDrivenCavity/Re_100/
```

### Output

- **Terminal**: path of each plot file read; final array shape and component names.
- **File**: `.npy` array of shape `(n_samples, ncomp, nx, ny)`.
