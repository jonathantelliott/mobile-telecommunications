#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_datacons
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7GB

# Load Stata
module purge
module load stata/17.0

# Calculate mean data consumption
stata-mp -b do ${HOMELOC}code/dbar.do ${SCRATCHLOC}
