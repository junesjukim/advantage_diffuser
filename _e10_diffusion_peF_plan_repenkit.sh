#!/bin/bash
# Create output directories if they don't exist

prefix_str="peF_repenkit"

mkdir -p output
mkdir -p output/diffusion_plan_${prefix_str}

n_diffusion_steps=16
prefix_path="diffusion_plan/${prefix_str}"

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(2 3)
# 데이터셋 배열 정의
declare -a DATASETS=(
  "pen-cloned-v0"
  "kitchen-partial-v0"
)

# n_sample_timesteps 변수 정의
declare -a n_sample_timesteps=(
  16
  16
)

# Train seeds와 value seeds 정의
train_seeds=(10 20 30)
value_seeds=(10 20)

# 각 train seed와 value seed 조합에 대해 실행
for train_seed in "${train_seeds[@]}"; do
  for value_seed in "${value_seeds[@]}"; do
    # 0부터 29까지의 planning seed 실행
    for plan_seed in {0..29}; do
      # 각 GPU에서 작업 실행
      pids=()
      for i in "${!GPU_DEVICES[@]}"; do
        OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/plan_guided.py \
          --dataset ${DATASETS[$i]} \
          --logbase logs \
          --diffusion_loadpath "f:diffusion/diffusion_${prefix_str}_H32_T${n_diffusion_steps}_S${train_seed}" \
          --value_loadpath "f:values/diffusion_${prefix_str}_H32_T${n_diffusion_steps}_S${value_seed}_d0.99" \
          --horizon 32 \
          --n_diffusion_steps ${n_diffusion_steps} \
          --seed $plan_seed \
          --n_sample_timesteps ${n_sample_timesteps[$i]} \
          --discount 0.99 \
          --prefix ${prefix_path} > output/diffusion_plan_${prefix_str}/output_${GPU_DEVICES[$i]}_train${train_seed}_value${value_seed}_plan${plan_seed}.log 2>&1 &

        pids+=($!)
        echo "----------------------------------------"
        echo "[작업 시작] GPU ${GPU_DEVICES[$i]}"
        echo "- 데이터셋: ${DATASETS[$i]}"
        echo "- Train Seed: $train_seed"
        echo "- Value Seed: $value_seed"
        echo "- Plan Seed: $plan_seed"
        echo "- Diffusion Steps: ${n_diffusion_steps}"
        echo "- Sample Timesteps: ${n_sample_timesteps[$i]}"
        echo "- PID: $!"
        echo "- Prefix: ${prefix_path}"
        echo "----------------------------------------"
      done

      # Wait for all background jobs to finish before moving to the next seed
      wait "${pids[@]}"
      echo "Train Seed $train_seed, Value Seed $value_seed, Plan Seed $plan_seed에 대한 모든 작업이 완료되었습니다"
    done
  done
done

echo "모든 작업이 완료되었습니다." 