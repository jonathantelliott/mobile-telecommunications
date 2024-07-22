#!/bin/bash -e
#
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=7GB
#SBATCH --job-name=pre_infrastructure
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Add population distributions
module purge
singularity \
    exec  \
    --overlay ${ROVERLAYLOC1}:ro \
    --overlay ${ROVERLAYLOC2}:ro \
    --overlay ${OVERLAYLOC}:ro \
    ${IMAGERLOC} \
    /bin/bash -c "
source /ext3/env.sh
export R_LIBS=/ext3/R_LIBS
Rscript ${HOMELOC}code/pop_dist.R ${SCRATCHLOC} > ${SCRATCHLOC}slurm/log_preprocess_infr_popdistR.txt
"

# Import base station data
module purge
module load stata/17.0
stata-mp -b do ${HOMELOC}code/base_stations.do ${SCRATCHLOC}

# Add base station processing
module purge
singularity \
    exec  \
    --overlay ${ROVERLAYLOC1}:ro \
    --overlay ${ROVERLAYLOC2}:ro \
    --overlay ${OVERLAYLOC}:ro \
    ${IMAGERLOC} \
    /bin/bash -c "
source /ext3/env.sh
export R_LIBS=/ext3/R_LIBS
Rscript ${HOMELOC}code/base_stations.R ${SCRATCHLOC} > ${SCRATCHLOC}slurm/log_preprocess_infr_basestationsR.txt
"

# Combine to market-operator-specific level
module purge
module load stata/17.0
stata-mp -b do ${HOMELOC}code/infrastructure_calibration.do ${SCRATCHLOC}
