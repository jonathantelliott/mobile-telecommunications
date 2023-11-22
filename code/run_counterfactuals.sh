#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0
#SBATCH --job-name=cntrfctls
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=2-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jte254@nyu.edu

#SCRATCHLOC=$(sed -n -e '/^scratch_path/p' paths.py)
#SCRATCHLOC=${SCRATCHLOC#*\"}
#SCRATCHLOC=${SCRATCHLOC%%\"}
#HOMELOC=$(sed -n -e '/^home_path/p' paths.py)
#HOMELOC=${HOMELOC#*\"}
#HOMELOC=${HOMELOC%%\"}
#ENVLOC="/scratch/jte254/telecom-env/telecom-cuda11.0.ext3"

singularity exec $nv \
	    --overlay ${ENVLOC}:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/counterfactuals.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
"
