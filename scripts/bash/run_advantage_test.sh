#!/usr/bin/env bash
###############################################################################
# scripts/bash/run_unified_d4rl_test.sh
#
#  - Lightweight test runner for `scripts/plan_guided_unified.py` in D4RL mode
#  - Runs a handful of planning seeds on one or more D4RL datasets.
#  - Edit DIFFUSION_PATH / VALUE_PATH to point to your checkpoints before use.
###############################################################################

######################## 사용자 설정 ##########################################
PREFIX="advantage_diffuser"            # 로그 경로 구분용 접두어
PREFIX_PATH="diffusion_plan/${PREFIX}"
LOG_BASE="logs"                 # diffuser 기본 로그 폴더
OUTPUT_DIR="output/diffusion_plan_${PREFIX}"

# 동시에 실행할 최대 작업 수
MAX_CONCURRENT_JOBS=8

# GPU 장치 배열 (여러 개 지정 가능)
declare -a GPU_DEVICES=(0 1 2 3)

# 테스트할 D4RL 데이터셋 목록
declare -a DATASETS=(
  "pen-cloned-v0"
  "kitchen-partial-v0"
)

# 각 DATASET 별 n_sample_timesteps 설정 (DATASETS와 길이 동일)
declare -a N_SAMPLE_TIMESTEPS=(
  16
  16
)

# 고정 하이퍼파라미터
HORIZON=32
N_DIFF_STEPS=16



# Seed 설정 -------------------------------------------------------------------
TRAIN_SEED=(10 20 30)
VALUE_SEED=0
PLAN_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)   # 테스트용 planning seed 3개

# 체크포인트 경로(사용 전에 수정 필요) ---------------------------------------
# Example:
# DIFFUSION_PATH="f:diffusion/flowmatcher_hopper_H32_T16_S10"
# VALUE_PATH="f:values/value_hopper_H32_T16_S10"
DIFFUSION_PATH="f:diffusion/diffusion_peF_repenkit_H32_T16_S${TRAIN_SEED}"
VALUE_PATH="f:advantages/test_3_H32_T16_S${VALUE_SEED}_d0.99"
###############################################################################

# 디렉터리 준비
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_BASE}"

# 루프 실행 -------------------------------------------------------------------
job_counter=0
for idx in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx]}"
  NST="${N_SAMPLE_TIMESTEPS[$idx]}"

  for TRAIN_SEED_VAL in "${TRAIN_SEED[@]}"; do
    for PLAN_SEED in "${PLAN_SEEDS[@]}"; do
      # 동시에 실행되는 작업의 수가 최대치에 도달하면, 작업이 하나 끝날 때까지 기다립니다.
      if (($(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS)); then
        wait -n
      fi

      # GPU를 순환 방식으로 할당합니다.
      GPU_IDX=$((job_counter % ${#GPU_DEVICES[@]}))
      GPU="${GPU_DEVICES[$GPU_IDX]}"

      LOG_FILE="${OUTPUT_DIR}/${DATASET//\//_}_train${TRAIN_SEED_VAL}_plan${PLAN_SEED}.log"

      echo "[실행] GPU ${GPU} | Dataset ${DATASET} | TrainSeed ${TRAIN_SEED_VAL} | PlanSeed ${PLAN_SEED}"
      echo "  로그 -> ${LOG_FILE}"

      # OMP_NUM_THREADS 값을 6으로 유지합니다.
      (OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=${GPU} \
      python scripts/plan_guided_unified.py \
        --dataset "${DATASET}" \
        --logbase "${LOG_BASE}" \
        --benchmark d4rl \
        --save_video \
        --diffusion_loadpath "f:diffusion/diffusion_peF_repenkit_H32_T16_S${TRAIN_SEED_VAL}" \
        --value_loadpath "${VALUE_PATH}" \
        --horizon ${HORIZON} \
        --n_diffusion_steps ${N_DIFF_STEPS} \
        --seed ${PLAN_SEED} \
        --n_sample_timesteps ${NST} \
        --prefix "${PREFIX}/TR${TRAIN_SEED_VAL}_VS${VALUE_SEED}_PS${PLAN_SEED}" \
        --wandb_project "advantage_test" \
        > "${LOG_FILE}" 2>&1) &

      job_counter=$((job_counter + 1))
    done
  done
done

# 백그라운드 작업 완료 대기
echo "모든 작업을 시작했으며, 완료될 때까지 기다립니다..."
wait

echo "모든 테스트가 완료되었습니다." 