#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p output/diffuser_plan_w,h

# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # Run first job on GPU 4
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=4 python scripts/plan_guided.py \
    --dataset walker2d-medium-expert-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
    --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 1 \
    --seed $seed \
    --discount 0.99 > output/diffuser_plan_1,2/output_0_seed_${seed}.log 2>&1 &

  pid1=$!  # Capture process ID of first job

  # Run second job on GPU 5
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=5 python scripts/plan_guided.py \
    --dataset walker2d-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
    --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 1 \
    --seed $seed \
    --discount 0.99 > output/diffuser_plan_1,2/output_1_seed_${seed}.log 2>&1 &

  pid2=$!  # Capture process ID of second job

  # Run third job on GPU 6
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=6 python scripts/plan_guided.py \
    --dataset hopper-medium-expert-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
    --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 1 \
    --seed $seed \
    --discount 0.99 > output/diffuser_plan_1,2/output_2_seed_${seed}.log 2>&1 &

  pid3=$!  # Capture process ID of third job

  # Run fourth job on GPU 7
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=7 python scripts/plan_guided.py \
    --dataset hopper-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
    --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 1 \
    --seed $seed \
    --discount 0.99 > output/diffuser_plan_1,2/output_3_seed_${seed}.log 2>&1 &

  pid4=$!  # Capture process ID of fourth job

  echo "Started jobs for seed $seed on GPU 4, GPU 5, GPU 6, and GPU 7"

  # Wait for all background jobs to finish before moving to the next seed
  wait $pid1 $pid2 $pid3 $pid4

  echo "Completed jobs for seed $seed"
done

echo "All jobs have been completed."
