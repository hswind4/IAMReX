import os
import subprocess


# os.chdir("../Tutorials")
# os.chdir("LidDrivenCavity")
# subprocess.run(["make", "-j8"], check=True)


subprocess.run("cd ../Tutorials/LidDrivenCavity && make -j8 && ./amr2d.gnu.ex inputs.2d.lid_driven_cavity", shell=True, check=True)
