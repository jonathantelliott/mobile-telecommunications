#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=pre_quality
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB

# Load Stata
module purge
module load stata/17.0

# Import Ookla data
stata-mp -b do ${HOMELOC}code/import_ookla.do ${SCRATCHLOC}

# Convert coordinates to markets
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/get_insee_code.py ${SCRATCHLOC} > ${SCRATCHLOC}slurm/log_preprocess_quality_insee.txt
"

# Construct average download speeds for each MNO-market
stata-mp -b do ${HOMELOC}code/ookla_quality.do ${SCRATCHLOC}
