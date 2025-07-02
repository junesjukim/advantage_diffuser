#!/usr/bin/env bash
###############################################################################
# scripts/bash/run_advantage_train.sh
#
#  - 여러 D4RL 환경(kitchen, pen 등)에 대해 Advantage Diffusion 모델 학습
#  - 결과는 logs/<dataset>/advantages/<PREFIX>/seed-<N>/ 에 저장
#  - GPU/OMP 설정, nohup 백그라운드 실행, 프로세스 PID 기록
#  - 스크립트 위치가 어디든 상대 경로 문제없이 실행되도록 프로젝트 루트 계산
###############################################################################

######################## 실험별 사용자 설정 ###################################
PREFIX="test"               # 실험 구분용 접두어(폴더 이름에 사용)
N_DIFF_STEPS=16                 # --n_diffusion_steps 하이퍼파라미터
SEEDS=(0 1)                      # 여러 시드 사용 시 예: (20 42 77)
GPU_LIST=(0 1 2 3)                  # 사용하고자 하는 GPU ID 리스트
DATASETS=(                      # 학습할 D4RL 환경 리스트
  "pen-cloned-v0"
  "kitchen-partial-v0"
)
# IQL 학습 결과가 저장된 runs 디렉터리(프로젝트 루트 기준)
# 구조 예)
# iql-pytorch/runs/kitchen-partial-v0-seed1/kitchen-partial-v0/.../model/critic_target_s1000000.pth
# 가장 step 숫자가 큰 파일을 자동으로 사용
IQL_RUNS_DIR="iql-pytorch/runs"
###############################################################################

OMP_THREADS=6                  # OMP_NUM_THREADS 값
LOG_BASE="logs"                # diffuser 기본 로그 폴더

# -----------------------------------------------------------------------------
# 프로젝트 루트 및 스크립트 위치 계산
# -----------------------------------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( realpath "${SCRIPT_DIR}/../../" )"
cd "${PROJECT_ROOT}" || exit 1

# 로그/결과 폴더 생성
mkdir -p "${LOG_BASE}"

# 모든 PID 기록 파일(한눈에 관리용)
PID_SUMMARY="${PROJECT_ROOT}/running_advantage_pids.txt"
> "${PID_SUMMARY}"

# -----------------------------------------------------------------------------
# 함수: 단일 Advantage 학습 잡 실행
# -----------------------------------------------------------------------------
run_job () {
  local DATASET=$1
  local SEED=$2
  local GPU=$3

  # 저장 디렉터리: logs/<dataset>/advantages/<PREFIX>/seed-<SEED>
  local SAVE_DIR="${LOG_BASE}/${DATASET}/advantages/${PREFIX}/seed-${SEED}"
  mkdir -p "${SAVE_DIR}"

  # ------------------------------------
  # IQL 체크포인트 자동 탐색
  # ------------------------------------
  local DATASET_SEED_DIR="${PROJECT_ROOT}/${IQL_RUNS_DIR}/${DATASET}-seed${SEED}"

  if [[ ! -d "${DATASET_SEED_DIR}" ]]; then
      echo "[WARN] ${DATASET_SEED_DIR} 경로가 없습니다. 이전 run 디렉터리를 확인하세요." >&2
  fi

  # critic_target 파일 중 step 숫자가 가장 큰 것 선택
  local Q_CKPT=$(find "${DATASET_SEED_DIR}" -type f -name 'critic*_s*.pth' 2>/dev/null \
                   | sort -V | tail -n 1)
  local V_CKPT=$(find "${DATASET_SEED_DIR}" -type f -name 'value*_s*.pth' 2>/dev/null \
                   | sort -V | tail -n 1)

  if [[ -z "${Q_CKPT}" || -z "${V_CKPT}" ]]; then
      echo "[ERR ] IQL 체크포인트를 찾지 못했습니다: ${DATASET}, seed ${SEED}" >&2
      return
  fi

  # 로그 및 PID 파일
  local LOG_FILE="${SAVE_DIR}/train.log"
  local PID_FILE="${SAVE_DIR}/pid.txt"

  echo "[RUN ] ${DATASET} | seed ${SEED} | GPU ${GPU}"
  echo "[SAVE] ${SAVE_DIR}"

  (
    export OMP_NUM_THREADS=${OMP_THREADS}
    export CUDA_VISIBLE_DEVICES=${GPU}

    nohup python "${PROJECT_ROOT}/scripts/train_advantage_model.py" \
        --dataset "${DATASET}" \
        --q_path "${Q_CKPT}" \
        --v_path "${V_CKPT}" \
        --horizon 32 \
        --n_diffusion_steps "${N_DIFF_STEPS}" \
        --seed "${SEED}" \
        --normalizer 'DebugNormalizer' \
        --prefix "advantages/${PREFIX}" \
        --config config.d4rl \
        > "${LOG_FILE}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "PID ${DATASET} seed-${SEED}: $(cat "${PID_FILE}")" >> "${PID_SUMMARY}"
  )

  echo "[PID ] $(cat "${PID_FILE}") (저장: ${PID_FILE})"
  echo "--------------------------------------------------"
}

# -----------------------------------------------------------------------------
# 메인 실행 루프
# -----------------------------------------------------------------------------
JOB_IDX=0
for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    GPU="${GPU_LIST[$((JOB_IDX % ${#GPU_LIST[@]}))]}"
    run_job "${DATASET}" "${SEED}" "${GPU}"
    ((JOB_IDX++))
  done
done

# -----------------------------------------------------------------------------
# PID 요약 출력
# -----------------------------------------------------------------------------
echo ""
echo "=== 실행 중인 Advantage 학습 프로세스 PID 목록 ==="
cat "${PID_SUMMARY}"
echo "각 PID에 대해 'kill -9 <PID>' 로 간편히 종료할 수 있습니다." 