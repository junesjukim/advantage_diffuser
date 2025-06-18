#!/bin/bash

# 로그 디렉토리 생성
mkdir -p nohup_logs

# 각 실험 스크립트 병렬 실행
echo "Starting _e1 experiment..."
nohup bash _e1_diffusion_peT_train_repenkit.sh > nohup_logs/e1.log 2>&1 &
echo "Starting _e2 experiment..."
nohup bash _e2_diffusion_peF_train_repenkit.sh > nohup_logs/e2.log 2>&1 &
echo "Starting _e3 experiment..."
nohup bash _e3_flowmatching_peT_train_repenkit.sh > nohup_logs/e3.log 2>&1 &
echo "Starting _e4 experiment..."
nohup bash _e4_flowmatching_peF_train_repenkit.sh > nohup_logs/e4.log 2>&1 &
echo "Starting _e5 experiment..."
nohup bash _e5_diffusion_train_value_repenkit.sh > nohup_logs/e5.log 2>&1 &
echo "Starting _e7 experiment..."
nohup bash _e7_flowmatching_train_value_repenkit.sh > nohup_logs/e7.log 2>&1 &

echo "모든 실험이 백그라운드에서 실행되었습니다."