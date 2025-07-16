#!/usr/bin/env bash
###############################################################################
# scripts/bash/run_evaluate_train_test.sh
#
#  - Test runner for `scripts/evaluate_dynamics.py` in D4RL mode.
#  - Runs dynamics evaluation on one or more D4RL datasets.
#  - Edit DIFFUSION_PATH / VALUE_PATH to point to your checkpoints before use.
###############################################################################

######################## 사용자 설정 ##########################################
PREFIX="dynamics_guide"            # 로그 경로 구분용 접두어
PREFIX_PATH="diffusion_plan/${PREFIX}"
LOG_BASE="logs"                 # diffuser 기본 로그 폴더
OUTPUT_DIR="output/diffusion_plan_${PREFIX}"

# GPU 장치 배열 (여러 개 지정 가능)
declare -a GPU_DEVICES=(0 1 2 3)

# 테스트할 D4RL 데이터셋 목록
declare -a DATASETS=(
  #"pen-cloned-v0"
  "kitchen-partial-v0"
)

# 각 DATASET 별 n_sample_timesteps 설정 (DATASETS와 길이 동일)
declare -a N_SAMPLE_TIMESTEPS=(
  #16
  16
)
declare -a SCALE=(
  #0
  #0.001
  #0.01
  #0.1
  0.3
  0.5
  1.0
  2.0
)

# 고정 하이퍼파라미터
HORIZON=32
N_DIFF_STEPS=16



# Seed 설정 -------------------------------------------------------------------
TRAIN_SEED=10
VALUE_SEED=10
PLAN_SEEDS=(1 2 7 11)   # 테스트용 planning seed 15개

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

# 각 GPU에 대해 Xvfb 서버 시작
echo "Starting Xvfb servers for each GPU..."
xvfb_pids=()
for i in "${!GPU_DEVICES[@]}"; do
  GPU_ID="${GPU_DEVICES[$i]}"
  DISPLAY_NUM=$((100 + i))
  # CUDA_VISIBLE_DEVICES를 설정하여 Xvfb가 특정 GPU에서 실행되도록 함
  CUDA_VISIBLE_DEVICES=${GPU_ID} Xvfb :${DISPLAY_NUM} -screen 0 1024x768x24 -ac &
  xvfb_pids+=($!)
  echo "  - Xvfb started for GPU ${GPU_ID} on DISPLAY=:${DISPLAY_NUM} (PID: ${xvfb_pids[$i]})"
done

# 스크립트 종료 시 Xvfb 프로세스 정리
trap "echo 'Stopping Xvfb servers...'; kill ${xvfb_pids[@]}; exit" SIGINT SIGTERM

# 루프 실행 -------------------------------------------------------------------
pids=()
job_idx=0
for idx in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx]}"
  NST="${N_SAMPLE_TIMESTEPS[$idx]}"

  for PLAN_SEED in "${PLAN_SEEDS[@]}"; do
    for SCALE_VAL in "${SCALE[@]}"; do
      gpu_idx=$((job_idx % ${#GPU_DEVICES[@]}))
      GPU="${GPU_DEVICES[$gpu_idx]}"
      DISPLAY_NUM=$((100 + gpu_idx))

      LOG_FILE="${OUTPUT_DIR}/${DATASET//\//_}_plan${PLAN_SEED}_scale${SCALE_VAL}.log"

      echo "[실행] GPU ${GPU} | DISPLAY :${DISPLAY_NUM} | Dataset ${DATASET} | PlanSeed ${PLAN_SEED} | Scale ${SCALE_VAL}"
      echo "  로그 -> ${LOG_FILE}"

        CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU} DISPLAY=:${DISPLAY_NUM} \
        python scripts/evaluate_dynamics.py \
          --batch_size 64 \
          --scale "${SCALE_VAL}" \
          --dataset "${DATASET}" \
          --logbase "${LOG_BASE}" \
          --benchmark d4rl \
          --diffusion_loadpath "${DIFFUSION_PATH}" \
          --value_loadpath "${VALUE_PATH}" \
          --horizon ${HORIZON} \
          --n_diffusion_steps ${N_DIFF_STEPS} \
          --seed ${PLAN_SEED} \
          --n_sample_timesteps ${NST} \
          --prefix "${PREFIX}/TR${TRAIN_SEED}_VS${VALUE_SEED}_PS${PLAN_SEED}_SC${SCALE_VAL}" \
          --save_video \
          > "${LOG_FILE}" 2>&1 &

        pids+=("$!")
        job_idx=$((job_idx + 1))
    done
  done
done

# 백그라운드 작업 완료 대기
if [ ${#pids[@]} -gt 0 ]; then
  echo "=== 실행 중인 프로세스 PID 목록 ==="
  printf '%s\n' "${pids[@]}"
  wait "${pids[@]}"
fi

echo "모든 테스트가 완료되었습니다."
echo "Stopping Xvfb servers..."
kill "${xvfb_pids[@]}" 