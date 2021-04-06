#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=cntrfctls
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
	    --overlay /scratch/jte254/telecom-env/telecom-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
cd /home/jte254/telecom/
python code/counterfactuals.py $SLURM_ARRAY_TASK_ID > /scratch/jte254/telecom/slurm/log_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
"
