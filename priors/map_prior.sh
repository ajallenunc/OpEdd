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

PRIOR_DIR=${DaTaFolder}/priors/${PRIOR_NAME}

# Step 4: Map mean and covariance functions into the subject space 
sbatch warp_prior.sh $PRIOR_DIR $SUBJECT_DIR

# Step 5: Create EB object for each training subject
sbatch compute_EB.sh $SUBJECT_DIR

# Step 6 (optional) Run example analysis 

##python example.py --subject_dir $SUBJECT_DIR --sh_order 8 --Bval 2000 --n_voxel_samps 100 --M