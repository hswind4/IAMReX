import sys
import numpy as np
from mpi4py import MPI
import amrex.space3d as amr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

amr.initialize([])

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

## Check if you have MPI
# you might get nompi version of pyamrex from conda install
if mpi_rank == 0:
    print(
        f"MPI size={mpi_size}, AMReX NProcs={amr.ParallelDescriptor.NProcs()}, "
        f"Config.have_mpi={amr.Config.have_mpi}"
    )

argv = sys.argv
if len(argv) < 2:
    if mpi_rank == 0:
        print("Usage: python main.py pltfile [lev] [z_real]")
    amr.finalize()
    sys.exit(0)

pltfilename = argv[1]
lev = int(argv[2]) if len(argv) > 2 else 0

## Read plot file; AMReX distributes boxes across MPI ranks automatically
input_plt = amr.PlotFileData(pltfilename)
input_mf = input_plt.get(lev)

nBox_total = input_mf.box_array().size
local_fabs = input_mf.to_numpy()
nBox_local = len(local_fabs)
comm.Barrier()
for r in range(mpi_size):
    if mpi_rank == r:
        print(f"Rank {r}: {nBox_local}/{nBox_total} boxes")
    comm.Barrier()

## Gather the full domain array for visualization (all ranks participate)
full_np = input_mf[:]

## Draw a z-slice of x_velocity
names = list(input_plt.varNames())
idxcomp = names.index("x_velocity")
z_real = float(argv[3]) if len(argv) > 3 else 5.0
dz = input_plt.cellSize(lev)[2]
z0 = input_plt.probLo()[2]
k = int(round((z_real - (z0 + 0.5 * dz)) / dz))

if mpi_rank == 0:
    slice_2d = full_np[:, :, k, idxcomp].T
    plt.figure(figsize=(6, 5))
    im = plt.imshow(slice_2d, origin="lower")
    plt.title(f"{names[idxcomp]} at z≈{z_real} (k={k})")
    plt.colorbar(im, shrink=0.8)
    out_path = "slice_z.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

comm.Barrier()

amr.finalize()
