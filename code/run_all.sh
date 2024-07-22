#!/bin/sh

# RUN ENTIRE PROJECT
# Note: working directory must be same as where run_all.sh is saved

# Define locations, import from paths.py
SCRATCHLOC=$(sed -n -e '/^scratch_path/p' paths.py)
SCRATCHLOC=${SCRATCHLOC#*\"}
SCRATCHLOC=${SCRATCHLOC%%\"}
HOMELOC=$(sed -n -e '/^home_path/p' paths.py)
HOMELOC=${HOMELOC#*\"}
HOMELOC=${HOMELOC%%\"}
IMAGELOC=$(sed -n -e '/^image_path/p' paths.py)
IMAGELOC=${IMAGELOC#*\"}
IMAGELOC=${IMAGELOC%%\"}
IMAGERLOC=$(sed -n -e '/^image_r_path/p' paths.py)
IMAGERLOC=${IMAGERLOC#*\"}
IMAGERLOC=${IMAGERLOC%%\"}
OVERLAYLOC=$(sed -n -e '/^overlay_path/p' paths.py)
OVERLAYLOC=${OVERLAYLOC#*\"}
OVERLAYLOC=${OVERLAYLOC%%\"}
ROVERLAYLOC1=$(sed -n -e '/^r_overlay_1_path/p' paths.py)
ROVERLAYLOC1=${ROVERLAYLOC1#*\"}
ROVERLAYLOC1=${ROVERLAYLOC1%%\"}
ROVERLAYLOC2=$(sed -n -e '/^r_overlay_2_path/p' paths.py)
ROVERLAYLOC2=${ROVERLAYLOC2#*\"}
ROVERLAYLOC2=${ROVERLAYLOC2%%\"}

# Create directories if they don't currently exist
mkdir -p ${SCRATCHLOC}slurm/
mkdir -p ${SCRATCHLOC}data/
mkdir -p ${SCRATCHLOC}arrays/
mkdir -p ${HOMELOC}graphs/
mkdir -p ${HOMELOC}tables/
mkdir -p ${HOMELOC}stats/

# Export paths
export SCRATCHLOC
export HOMELOC
export IMAGELOC
export IMAGERLOC
export OVERLAYLOC
export ROVERLAYLOC1
export ROVERLAYLOC2

# Run data pre-processing files
cd ${SCRATCHLOC}slurm/
RES1=$(sbatch ${HOMELOC}code/run_preprocess_msize.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES2=$(sbatch ${HOMELOC}code/run_preprocess_chset.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES3=$(sbatch --dependency=afterok:${RES1##* } ${HOMELOC}code/run_preprocess_quality.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES4=$(sbatch --dependency=afterok:${RES1##* },${RES2##* } ${HOMELOC}code/run_preprocess_dataconsumption.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES5=$(sbatch --dependency=afterok:${RES1##* } ${HOMELOC}code/run_preprocess_income.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES6=$(sbatch --dependency=afterok:${RES1##* },${RES3##* } ${HOMELOC}code/run_preprocess_infrastructure.sh)
sleep 1

# Run demand construction files
cd ${SCRATCHLOC}slurm/
RES7=$(sbatch --dependency=afterok:${RES1##* },${RES2##* },${RES3##* },${RES4##* },${RES5##* },${RES6##* } ${HOMELOC}code/run_preprocess_demand.sh)
sleep 1

# Create summary statistics
cd ${SCRATCHLOC}slurm/
RES8=$(sbatch --dependency=afterok:${RES7##* } ${HOMELOC}code/run_summary_statistics.sh)
sleep 1

# Estimate demand
cd ${SCRATCHLOC}slurm/
RES9=$(sbatch --dependency=afterok:${RES7##* } ${HOMELOC}code/run_demand.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES10=$(sbatch --dependency=afterok:${RES9##* } ${HOMELOC}code/run_process_demand.sh)
sleep 1

# Run counterfactuals
cd ${SCRATCHLOC}slurm/
RES11=$(sbatch --dependency=afterok:${RES9##* } ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
cd ${SCRATCHLOC}slurm/
RES12=$(sbatch --dependency=afterok:${RES11##* } ${HOMELOC}code/run_process_counterfactuals.sh)
sleep 1
