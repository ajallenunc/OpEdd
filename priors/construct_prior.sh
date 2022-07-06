#!/bin/bash
#

#### replace 'myCondEnv3' with your anaconda3 created python environment
conda activate myCondEnv3

module load ants

# CHANGE FOR SPECIFIC SBATCH OPTIONS
OPTIONS="-p dmi --qos abcd"

# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
# get global config 
source config.txt

TRAIN_IDS=${OpEddFolder}/training_ids.txt
PRIOR_DIR=${DaTaFolder}/priors/${PRIOR_NAME}
DATA_DIR=${DaTaFolder}/DWI_unregistered

# create a unique job name prefix
JID=$(uuidgen | tr '-' ' ' | awk {'print $1}')

# get all training subject names
mapfile -t subjects < $TRAIN_IDS

## Build Prior 
# Step 1: fit diffuison signal
for i in $(seq 1 ${#subjects[@]}); do
        idx=$((i - 1))
        sbatch fit_historical_signal.sh $DATA_DIR $PRIOR_DIR ${subjects[$idx]}
done
wait 
# Step 2: Register diffusion signal to template 
for i in $(seq 1 ${#subjects[@]}); do
    idx=$((i - 1))
    sbatch warp_diffusion_signal.sh $DATA_DIR $PRIOR_DIR ${subjects[$idx]}
done
wait
# Step 3: Estimate mean and covariance functions on the template space 
sbatch estimate_mean_cov.sh $PRIOR_DIR
