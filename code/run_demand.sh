#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-1
#SBATCH --job-name=demand_est
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=75GB
#SBATCH --mail-type=BEGIN,END,FAIL
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
python ${HOMELOC}code/demand.py $SLURM_ARRAY_TASK_ID > ${SCRATCHLOC}slurm/log_demand_$SLURM_ARRAY_TASK_ID.txt
"
