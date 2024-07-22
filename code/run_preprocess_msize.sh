#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_msize
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7GB

# Load Stata
module purge
module load stata/17.0

# Calculate market sizes and aggregate shares
stata-mp -b do ${HOMELOC}code/msize.do ${SCRATCHLOC}
stata-mp -b do ${HOMELOC}code/agg_shares.do ${SCRATCHLOC}
