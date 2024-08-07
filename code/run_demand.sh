#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-1
#SBATCH --job-name=demand_est
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
 
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/demand.py $SLURM_ARRAY_TASK_ID > ${SCRATCHLOC}slurm/log_demand_$SLURM_ARRAY_TASK_ID.txt
"
