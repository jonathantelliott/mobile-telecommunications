#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_chset
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7GB

# Load Stata
module purge
module load stata/17.0

# Create choice set
stata-mp -b do ${HOMELOC}code/chset.do ${SCRATCHLOC}
