#!/bin/bash
#
#### replace 'OpTryEnv' with your anaconda3 created python environment
source ~/.bashrc_opedd
conda activate OpTryEnv

module load ants

# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
# get global config 
source config.txt

PRIOR_NAME=${1}
TRAIN_IDS=${OpEddFolder}/training_ids.txt
PRIOR_DIR=${DaTaFolder}/priors/${PRIOR_NAME}
DATA_DIR=${DaTaFolder}/priors/${PRIOR_NAME}/DWI_unregistered

# create a unique job name prefix
JID=$(uuidgen | tr '-' ' ' | awk {'print $1}')

# get all training subject names
mapfile -t subjects < $TRAIN_IDS

# Directory to hold slurm outputs 
mkdir -p out_slurm 

## Build Prior 
declare -a job_ids_step2
for i in $(seq 1 ${#subjects[@]}); do
        idx=$((i - 1))

        # Create training directory and copy over files
        mkdir -p $PRIOR_DIR/training/${subjects[$idx]}
        cp $DATA_DIR/${subjects[$idx]}/geom_field.nii.gz $PRIOR_DIR/training/${subjects[$idx]}

        # Step 1: Fit diffusion signal
        job_id_step1=$(sbatch -o out_slurm/fit_historical_signal_${subjects[$idx]}_%j.out fit_historical_signal.sh $DATA_DIR $PRIOR_DIR ${subjects[$idx]} | awk '{print $NF}')

        # Step 2: Register diffusion signal to template
        job_id_step2=$(sbatch --dependency=afterok:$job_id_step1 -o out_slurm/warp_diffusion_signal_${subjects[$idx]}_%j.out warp_diffusion_signal.sh $DATA_DIR $PRIOR_DIR ${subjects[$idx]} | awk '{print $NF}')

done

# Combine job IDs into a comma-separated list for dependency
job_dependency_step2=$(IFS=, ; echo "${job_ids_step2[*]}")

# Step 3: Estimate mean and covariance functions on the template space
sbatch --dependency=afterok:$job_dependency_step2 -o out_slurm/estimate_mean_cov_2_%j.out estimate_mean_cov_2.sh $PRIOR_DIR

