#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=process_cntrfctls
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB

# Convert coordinates to markets
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/process_counterfactuals.py > ${SCRATCHLOC}slurm/log_process_counterfactuals.txt
"
