import os
import subprocess


# os.chdir("../Tutorials")
# os.chdir("LidDrivenCavity")
# subprocess.run(["make", "-j8"], check=True)


# subprocess.run("cd ../Tutorials/LidDrivenCavity && make -j8 && ./amr2d.gnu.ex inputs.2d.lid_driven_cavity", shell=True, check=True)
# home = os.getcwd() 

# working_dir = home + "/Tutorials/LidDrivenCavity"

# # 使用 cwd 参数指定工作目录
# subprocess.run(
#     "make -j8 && ./amr2d.gnu.ex inputs.2d.lid_driven_cavity", 
#     shell=True, 
#     check=True, 
#     cwd=working_dir
# )

def test_lidDrivenCanvity(working_dir, print_output):
    subprocess.run(
    "make -j8 && ./amr2d.gnu.ex inputs.2d.lid_driven_cavity", 
    shell=True, 
    check=True, 
    cwd=working_dir,
    stdout=None if print_output else subprocess.DEVNULL,
    stderr=None if print_output else subprocess.DEVNULL
    )
    print("test_lidDrivenCanvity succceed")

def test_RSV(working_dir, print_output):
    subprocess.run(
    "make -j8 && ./amr2d.gnu.MPI.ex inputs.2d.rsv", 
    shell=True, 
    check=True, 
    cwd=working_dir,
    stdout=None if print_output else subprocess.DEVNULL,
    stderr=None if print_output else subprocess.DEVNULL
    )
    print("test_RSV succceed")
    
    


def main():
    # if print_output = fasle，the information of compile and running doesn't display in termianl 
    print_output =False 

    
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    print("Script Directory:", script_dir)

    working_dir = os.path.join(script_dir, "../Tutorials/LidDrivenCavity")
    print("Test Working Directory:", os.path.abspath(working_dir))
    test_lidDrivenCanvity(working_dir, print_output)

    working_dir = os.path.join(script_dir, "../Tutorials/RSV")
    print("Test Working Directory:", os.path.abspath(working_dir))
    test_RSV(working_dir, False)
    


if __name__== "__main__" :
    main()

