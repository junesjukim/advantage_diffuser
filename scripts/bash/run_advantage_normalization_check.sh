#!/usr/bin/env bash
###############################################################################
# scripts/bash/advantage_normalization_check.sh
#
# 간단한 정규화 검증 스크립트.
#   1) D4RL 환경 이름과 IQL seed 를 받아서 최신 Q/V 체크포인트를 자동 탐색
#   2) Python 코드로 AdvantageDataset 생성 후
#      Raw vs Z-score 입력에서 Q/V/Advantage 통계를 출력
#
# 사용 예)
#   bash scripts/bash/advantage_normalization_check.sh kitchen-partial-v0 0
###############################################################################

set -euo pipefail

DATASET="${1:-kitchen-partial-v0}"
SEED="${2:-0}"
PROJECT_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}" || { echo "[ERR] Failed to cd to project root: ${PROJECT_ROOT}" >&2; exit 1; }

# IQL 러닝 디렉터리 구조 가정: iql-pytorch/runs/<dataset>-seed<seed>/*/model/*.pth
RUN_ROOT="iql-pytorch/runs/${DATASET}-seed${SEED}"
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "[ERR] IQL run directory not found: ${RUN_ROOT}" >&2; exit 1
fi

Q_PATH=$(find "${RUN_ROOT}" -type f -name 'critic*_s*.pth' | sort -V | tail -n1)
V_PATH=$(find "${RUN_ROOT}" -type f -name 'value*_s*.pth'  | sort -V | tail -n1)

if [[ -z "${Q_PATH}" || -z "${V_PATH}" ]]; then
  echo "[ERR] Could not locate Q/V checkpoints for ${DATASET} seed ${SEED}" >&2; exit 1
fi

echo "[INFO] Dataset      : ${DATASET}"
echo "[INFO] Seed         : ${SEED}"
echo "[INFO] Q checkpoint : ${Q_PATH}"
echo "[INFO] V checkpoint : ${V_PATH}"
echo "--------------------------------------------------"

python - << PY
import torch, numpy as np
from iql_pytorch.critic import Critic, ValueCritic
from diffuser.datasets.sequence import SequenceDataset
from diffuser.datasets.advantage import AdvantageDataset
import os, sys

env_name = "${DATASET}"
q_ckpt   = "${Q_PATH}"
v_ckpt   = "${V_PATH}"

seq = SequenceDataset(env=env_name,
                      horizon=32,
                      normalizer='DebugNormalizer',
                      use_padding=True)
print(f"SequenceDataset loaded | obs_dim {seq.observation_dim}, act_dim {seq.action_dim}")

q = Critic(seq.observation_dim, seq.action_dim)
q.load_state_dict(torch.load(q_ckpt, map_location='cpu'))
q.eval()

v = ValueCritic(seq.observation_dim, 512, 5)
v.load_state_dict(torch.load(v_ckpt, map_location='cpu'))
v.eval()

adv = AdvantageDataset(seq, q, v, device='cpu')
print('\nAdvantageDataset mean/std (first 5 dims)')
print('mean', adv.state_mean[0,:5])
print('std ', adv.state_std [0,:5])

traj = seq[0].trajectories
act = torch.from_numpy(traj[:, :seq.action_dim])
obs = torch.from_numpy(traj[:, seq.action_dim:])
obs_z = (obs - adv.state_mean) / adv.state_std

with torch.no_grad():
    q_raw = q(obs,   act)[0].squeeze(); v_raw = v(obs).squeeze()
    q_z   = q(obs_z, act)[0].squeeze(); v_z   = v(obs_z).squeeze()

print('\nRaw  Q mean', q_raw.mean().item(), 'V mean', v_raw.mean().item(), 'Adv mean', (q_raw-v_raw).mean().item())
print('Z-sc Q mean', q_z.mean().item(), 'V mean', v_z.mean().item(), 'Adv mean', (q_z-v_z).mean().item())
PY
