#!/bin/bash

# Define an array of datasets
dataset=("halfcheetah-medium-replay-v2")
#        "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" \
#         "walker2d-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-medium-v2")


#dataset=("halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-v2" \
#        "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" \
#         "walker2d-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-medium-v2")

# Loop over dataset array
for dataset_i in "${!dataset[@]}"
do
    # Loop over seeds
    for seed in {0..0}
    do
        # Loop over n_diffusion_step values
        for n_diffusion_step in 8 4
        do
            echo "Running train.py with dataset: ${dataset[$dataset_i]}, seed: $seed, and n_diffusion_step: $n_diffusion_step"

            # Run the Python script
            OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
                --dataset "${dataset[$dataset_i]}" --seed "$seed" --n_diffusion_steps "$n_diffusion_step"
        done
    done
done
