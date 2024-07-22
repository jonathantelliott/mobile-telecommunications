#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_income
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

# Load Stata
module purge
module load stata/17.0

# Calculate market sizes
stata-mp -b do ${HOMELOC}code/income.do ${SCRATCHLOC}
