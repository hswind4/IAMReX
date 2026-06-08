
# About RKPM flow past cylinder/ellipsoid case settings

The current configuration file only provides the settings for RKPM in `inputs`.
Please use the Python script provided in `tools/RKPM_weight` to generate the RKPM weight information. 
The generated information is contained in two files, namely `rkpm_mappings.id` and `rkpm_mappings.lag`. 
This script provides the functionality to **rotate geometries**, and rotating geometries does **not require any modification of the inputs file**.

