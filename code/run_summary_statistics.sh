#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=summ_stats
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB

# Load Stata
module purge
module load stata/17.0

# Process contracts data
stata-mp -b do ${HOMELOC}code/contracts_table.do ${SCRATCHLOC}

# Process OSIRIS data consumption
stata-mp -b do ${HOMELOC}code/data_consumption_patterns.do ${SCRATCHLOC} ${HOMELOC}

# Convert contracts data to .tex table
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/contracts_table.py > ${SCRATCHLOC}slurm/log_contracts_table.txt
"

# Create descriptive graphs
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/descriptive_graphs.py > ${SCRATCHLOC}slurm/log_descriptive_graphs.txt
"

# Create summary statistics table
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/summary_stats.py > ${SCRATCHLOC}slurm/log_summary_stats.txt
"

# Create data consumption statistics
singularity exec $nv \
	    --overlay ${OVERLAYLOC}:ro \
	    ${IMAGELOC} \
	    /bin/bash -c "
source /ext3/env.sh
python ${HOMELOC}code/data_consumption_patterns.py > ${SCRATCHLOC}slurm/log_data_consumption_patterns.txt
"
