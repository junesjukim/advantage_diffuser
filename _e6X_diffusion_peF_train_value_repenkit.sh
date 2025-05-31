#!/bin/bash
# Create output directories if they don't exist

prefix_str="peF_repenkit"

mkdir -p output
mkdir -p output/diffusion_value_${prefix_str}

n_diffusion_steps=16
seeds=(10 20)

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(3 4)
# 데이터셋 배열 정의  
declare -a DATASETS=(
  "pen-cloned-v0"
  "kitchen-partial-v0"
)

# 각 시드에 대해 실행
for seed in "${seeds[@]}"; do
  # 각 GPU에서 작업 실행
  pids=()
  for i in "${!GPU_DEVICES[@]}"; do
    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/train_values.py \
      --dataset ${DATASETS[$i]} \
      --logbase logs \
      --horizon 32 \
      --n_diffusion_steps ${n_diffusion_steps} \
      --n_saves 2 \
      --seed $seed \
      --prefix "values/diffusion_${prefix_str}" > output/diffusion_value_${prefix_str}/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

    pids+=($!)
    echo "Started job for seed $seed on GPU ${GPU_DEVICES[$i]}"
  done

  # Wait for all background jobs to finish
  wait "${pids[@]}"
  echo "All jobs for seed $seed have been completed."
done

echo "All jobs have been completed." 