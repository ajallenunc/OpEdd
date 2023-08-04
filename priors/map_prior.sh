#!/bin/bash
#

#### replace 'myCondEnv3' with your anaconda3 created python environment
conda activate OptTryEnv

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
PRIOR_DIR=${DaTaFolder}/priors/${PRIOR_NAME}
test_ids=${PRIOR_DIR}/testing_ids.txt
budget_list="5 10 20 30 40 50"

while IFS= read -r sub
do
    echo $sub
    # Step 1: Map Prior to Subject Space
    job1=$(sbatch --output=out_slurm/warp_prior_${sub}.out --error=out_slurm/warp_prior_${sub}.err warp_prior.sh $PRIOR_DIR "$PRIOR_DIR"/testing/$sub | cut -f 4 -d' ')

    # Step 2: Create EB object for each test subject
    job2=$(sbatch --output=out_slurm/compute_EB_${sub}.out --error=out_slurm/compute_EB_${sub}.err --dependency=afterok:$job1 compute_EB.sh "$PRIOR_DIR"/testing/$sub | cut -f 4 -d' ')

    # Step 3: Estimate fODF
    for b in $budget_list
    do
        sbatch --mem=200G --time=2- --job-name=sub_${sub}_${b}_opedd_fodf_job --output=out_slurm/opedd_signal_to_fodf_${sub}_${b}.out --error=out_slurm/opedd_signal_to_fodf_${sub}_${b}.err --dependency=afterok:$job2 --wrap="python -u opedd_signal_to_fodf.py --subject_dir=\"$PRIOR_NAME\"/testing/$sub --Bval=2000 --M=$b --sh_order=8"

    done
done < "$test_ids"

