#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p output/diffuser_plan_guideX

# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # Run first job on GPU 4
  # OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=4 python scripts/plan_guided.py \
  #   --dataset walker2d-medium-expert-v2 \
  #   --logbase logs \
  #   --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
  #   --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
  #   --horizon 4 \
  #   --n_diffusion_steps 1 \
  #   --seed $seed \
  #   --discount 0.99 \
  #   --prefix 'plans/guideX' > output/diffuser_plan_guideX/output_0_seed_${seed}.log 2>&1 &

  # pid1=$!  # Capture process ID of first job

  # Run second job on GPU 5
  n_steps_walker=1
  gpu_id_walker=6
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${gpu_id_walker} python scripts/plan_guided.py \
    --dataset walker2d-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath "f:diffusion/diffuser_H4_T${n_steps_walker}_S0" \
    --value_loadpath "f:values/diffusion_H4_T${n_steps_walker}_S0_d0.99" \
    --horizon 4 \
    --n_diffusion_steps ${n_steps_walker} \
    --seed $seed \
    --discount 0.99 \
    --prefix 'plans/guideX' > output/diffuser_plan_guideX/output_${gpu_id_walker}_seed_${seed}.log 2>&1 &

  pid2=$!  # Capture process ID of second job

  # Run third job on GPU 6
  # OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=6 python scripts/plan_guided.py \
  #   --dataset hopper-medium-expert-v2 \
  #   --logbase logs \
  #   --diffusion_loadpath 'f:diffusion/diffuser_H4_T1_S0' \
  #   --value_loadpath 'f:values/diffusion_H4_T1_S0_d0.99' \
  #   --horizon 4 \
  #   --n_diffusion_steps 1 \
  #   --seed $seed \
  #   --discount 0.99 \
  #   --prefix 'plans/guideX' > output/diffuser_plan_guideX/output_2_seed_${seed}.log 2>&1 &

  # pid3=$!  # Capture process ID of third job

  # Run fourth job on GPU 7
  n_steps_hopper=1
  gpu_id_hopper=7
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${gpu_id_hopper} python scripts/plan_guided.py \
    --dataset hopper-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath "f:diffusion/diffuser_H4_T${n_steps_hopper}_S0" \
    --value_loadpath "f:values/diffusion_H4_T${n_steps_hopper}_S0_d0.99" \
    --horizon 4 \
    --n_diffusion_steps ${n_steps_hopper} \
    --seed $seed \
    --discount 0.99 \
    --prefix 'plans/guideX' > output/diffuser_plan_guideX/output_${gpu_id_hopper}_seed_${seed}.log 2>&1 &

  pid4=$!  # Capture process ID of fourth job
  echo "GPU ${gpu_id_walker}, GPU ${gpu_id_hopper}에서 시드 $seed로 작업이 시작되었습니다"
  echo "GPU ${gpu_id_walker}: Walker2d-medium-replay-v2 데이터셋 (n_steps=$n_steps_walker)"
  echo "GPU ${gpu_id_hopper}: Hopper-medium-replay-v2 데이터셋 (n_steps=$n_steps_hopper)"

  # Wait for all background jobs to finish before moving to the next seed
  wait $pid2  $pid4
#$pid1 $pid3
  echo "Completed jobs for seed $seed"
done

echo "All jobs have been completed."
