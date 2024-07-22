# %%
# List of paths to folders where things are saved

# Locations within local machine
scratch_path = "/scratch/jte254/telecom/"
home_path = "/home/jte254/telecom/"
image_path = "/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif"
image_r_path = "/scratch/work/public/singularity/ubuntu-20.04.1.sif"
overlay_path = "/scratch/jte254/telecom-env/telecom-cuda11.0.ext3"
r_overlay_1_path = "/scratch/work/public/singularity/texlive-ubuntu20.04.1.sqf"
r_overlay_2_path = "/scratch/work/public/singularity/r4.1.2-ubuntu20.04.1-20211129.sqf"

# Paths that must be defined and are referenced in other sections of the code
data_path = scratch_path + "data/"
arrays_path = scratch_path + "arrays/"
graphs_path = home_path + "graphs/"
tables_path = home_path + "tables/"
stats_path = home_path + "stats/"
