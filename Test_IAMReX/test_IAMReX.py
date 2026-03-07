# SPDX-FileCopyrightText: 2025 Shuai He<hswind53@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

# Common overrides for fast CI runs: minimal steps, small grids, no file output
CI_OVERRIDES = "max_step=2 amr.plot_int=-1 amr.check_int=-1"

def run_test (name, working_dir, build_cmd, run_cmd, print_output):
    """Build and run a single test case."""
    full_cmd = f"{build_cmd} && {run_cmd}"
    subprocess.run(
        full_cmd,
        shell=True,
        check=True,
        cwd=working_dir,
        stdout=None if print_output else subprocess.DEVNULL,
        stderr=None if print_output else subprocess.DEVNULL
    )
    print(f"Test {name} succeed")

def main():
    # if print_output = false, the information of compile and running doesn't display in terminal
    print_output = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script Directory:", script_dir)

    # LidDrivenCavity (2D, no MPI — basic compilation and run test)
    working_dir = os.path.join(script_dir, "../Tutorials/LidDrivenCavity")
    print("Test Working Directory:", os.path.abspath(working_dir))
    run_test(
        "LidDrivenCavity",
        working_dir,
        "make -j8",
        f"./amr2d.gnu.ex inputs.2d.lid_driven_cavity {CI_OVERRIDES}",
        print_output
    )

    # RSV (2D, MPI, level set)
    working_dir = os.path.join(script_dir, "../Tutorials/RSV")
    print("Test Working Directory:", os.path.abspath(working_dir))
    run_test(
        "RSV",
        working_dir,
        "make -j8",
        f"./amr2d.gnu.MPI.ex inputs.2d.rsv {CI_OVERRIDES}",
        print_output
    )

    # DraftingKissingTumbling (3D, MPI, particles/IBM)
    working_dir = os.path.join(script_dir, "../Tutorials/DraftingKissingTumbling")
    print("Test Working Directory:", os.path.abspath(working_dir))
    run_test(
        "DraftingKissingTumbling",
        working_dir,
        "make -j8 USE_CUDA=FALSE USE_MPI=TRUE DEBUG=FALSE",
        f"mpiexec -np 2 ./amr3d.gnu.MPI.ex inputs.3d.DKT max_step=1 amr.n_cell=16 8 8 amr.plot_int=-1 amr.check_int=-1",
        print_output
    )


if __name__ == "__main__":
    main()
