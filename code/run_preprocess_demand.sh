#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_demanddata
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

# Load Stata
module purge
module load stata/17.0

# Create dataset used for demand estimation
stata-mp -b do ${HOMELOC}code/demand_data.do ${SCRATCHLOC}
