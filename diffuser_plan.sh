#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p output/diffuser_plan_kit

# 변수 정의
n_diffusion_steps=16

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(8)

# 데이터셋 배열 정의
declare -a DATASETS=(
  #pen-cloned-v0"
  "kitchen-partial-v0"
)

# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # 각 GPU에서 작업 실행
  pids=()
  for i in "${!GPU_DEVICES[@]}"; do
    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/plan_guided.py \
      --dataset ${DATASETS[$i]} \
      --logbase logs \
      --diffusion_loadpath "f:diffusion/diffuser_H32_T${n_diffusion_steps}_S0" \
      --value_loadpath "f:values/diffusion_H32_T${n_diffusion_steps}_S0_d0.99" \
      --horizon 32 \
      --n_diffusion_steps ${n_diffusion_steps} \
      --seed $seed \
      --n_sample_timesteps 16 \
      --discount 0.99 \
      --prefix 'plans/kit' > output/diffuser_plan_kit/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

    pids+=($!)
    echo "GPU ${GPU_DEVICES[$i]}에서 시드 $seed로 작업이 시작되었습니다"
    echo "GPU ${GPU_DEVICES[$i]}: ${DATASETS[$i]} 데이터셋 (n_steps=${n_diffusion_steps})"
  done

  # Wait for all background jobs to finish before moving to the next seed
  wait "${pids[@]}"
  echo "시드 $seed에 대한 모든 작업이 완료되었습니다"
done

echo "모든 작업이 완료되었습니다."