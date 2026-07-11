
# About RKPM flow past cylinder/ellipsoid case settings

The current configuration file only provides the settings for RKPM in `inputs`.
Please use the Python script provided in `tools/RKPM_weight` to generate the RKPM weight information. 
The generated information is contained in two files, namely `rkpm_mappings.id` and `rkpm_mappings.lag`. 
This script provides the functionality to **rotate geometries**, and rotating geometries does **not require any modification of the inputs file**.


# how to use

example
```bash
python main.py --inputs inputs.3d.flow_past_sphere --gemotry test_cylinder.txt --angle -45 --body-frame
```

The command-line arguments are as follows: 
- `inputs` is the inputs file for the case you want to run
- `geometry` is the Lagrange point distribution file you provide
- `body-frame` is a parameter to enable coordinate translation. If your Lagrange point distribution file is in its `original coordinates without offset`, then use this to offset it to the particle_center position in the corresponding inputs file
- `angle` represents the angle of rotation of the object along the z-axis in the x-y plane.
