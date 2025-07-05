#!/usr/bin/env bash
###############################################################################
# scripts/bash/run_unified_d4rl_test.sh
#
#  - Lightweight test runner for `scripts/plan_guided_unified.py` in D4RL mode
#  - Runs a handful of planning seeds on one or more D4RL datasets.
#  - Edit DIFFUSION_PATH / VALUE_PATH to point to your checkpoints before use.
###############################################################################

######################## 사용자 설정 ##########################################
PREFIX="unified_test"            # 로그 경로 구분용 접두어
PREFIX_PATH="diffusion_plan/${PREFIX}"
LOG_BASE="logs"                 # diffuser 기본 로그 폴더
OUTPUT_DIR="output/diffusion_plan_${PREFIX}"

# GPU 장치 배열 (여러 개 지정 가능)
declare -a GPU_DEVICES=(0 1)

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
TRAIN_SEED=10
VALUE_SEED=10
PLAN_SEEDS=(0 1 2)   # 테스트용 planning seed 3개

# 체크포인트 경로(사용 전에 수정 필요) ---------------------------------------
# Example:
# DIFFUSION_PATH="f:diffusion/flowmatcher_hopper_H32_T16_S10"
# VALUE_PATH="f:values/value_hopper_H32_T16_S10"
DIFFUSION_PATH="f:diffusion/diffusion_peF_repenkit_H32_T16_S${TRAIN_SEED}"
VALUE_PATH="f:values/diffusion_repenkit_H32_T16_S${VALUE_SEED}_d0.99"
###############################################################################

# 디렉터리 준비
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_BASE}"

# 루프 실행 -------------------------------------------------------------------
pids=()
for idx in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx]}"
  GPU="${GPU_DEVICES[$((idx % ${#GPU_DEVICES[@]}))]}"
  NST="${N_SAMPLE_TIMESTEPS[$idx]}"

  for PLAN_SEED in "${PLAN_SEEDS[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/${DATASET//\//_}_plan${PLAN_SEED}.log"

    echo "[실행] GPU ${GPU} | Dataset ${DATASET} | PlanSeed ${PLAN_SEED}"
    echo "  로그 -> ${LOG_FILE}"

    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU} \
    python scripts/plan_guided_unified.py \
      --dataset "${DATASET}" \
      --logbase "${LOG_BASE}" \
      --benchmark d4rl \
      --diffusion_loadpath "${DIFFUSION_PATH}" \
      --value_loadpath "${VALUE_PATH}" \
      --horizon ${HORIZON} \
      --n_diffusion_steps ${N_DIFF_STEPS} \
      --seed ${PLAN_SEED} \
      --n_sample_timesteps ${NST} \
      --prefix "${PREFIX}/TR${TRAIN_SEED}_VS${VALUE_SEED}_PS${PLAN_SEED}" \
      > "${LOG_FILE}" 2>&1 &

    pids+=("$!")
  done

done

# 백그라운드 작업 완료 대기
if [ ${#pids[@]} -gt 0 ]; then
  echo "=== 실행 중인 프로세스 PID 목록 ==="
  printf '%s\n' "${pids[@]}"
  wait "${pids[@]}"
fi

echo "모든 테스트가 완료되었습니다." 