#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-1
#SBATCH --job-name=cntrfctls
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12GB

singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/preprocess_counterfactuals.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_preprocess_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
python ${HOMELOC}code/counterfactuals.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
python ${HOMELOC}code/process_counterfactual_arrays.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_process_counterfactual_arrays_$SLURM_ARRAY_TASK_ID.txt
"
