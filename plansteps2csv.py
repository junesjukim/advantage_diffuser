#!/usr/bin/env python3
import os
import json
import csv
import math
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

def read_steps(file_path):
    """
    Attempts to read the JSON file at file_path and return the 'step' value.
    If the file does not exist or the field is missing, returns None.
    """
    if not os.path.exists(file_path):
        print(f"[Warning] File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            step = data.get("step")
            if step is None:
                print(f"[Warning] 'step' not found in {file_path}")
            return step
    except Exception as e:
        print(f"[Error] Could not read {file_path}: {e}")
        return None

def format_stat(avg, std_err):
    """
    Format average and standard error as 'XX.X±Y.Y'
    Values are multiplied by 100 before formatting
    """
    if avg is None or std_err is None:
        return "N/A"
    avg_scaled = avg * 100
    std_err_scaled = std_err * 100
    return f"{avg_scaled:.1f}±{std_err_scaled:.1f}"

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_steps(rollout_data):
    if rollout_data is None:
        return None, None, 0
    
    step = rollout_data
    if step == 999:
        return step, 0, 1
    return step, 0, 0

def main():
    # Set environments
    env1 = "pen-cloned"
    env2 = "kitchen-partial"
    
    # Define the categories with their corresponding file path templates
    categories = {
        # pen-cloned environment (prefix: pen)
        f"{env1}_ST1": f"logs/{env1}-v0/diffusion_plan/pen_H32_T16_S{{seed}}_d0.99_ST1/0/rollout.json",
        f"{env1}_ST2": f"logs/{env1}-v0/diffusion_plan/pen_H32_T16_S{{seed}}_d0.99_ST2/0/rollout.json",
        f"{env1}_ST4": f"logs/{env1}-v0/diffusion_plan/pen_H32_T16_S{{seed}}_d0.99_ST4/0/rollout.json",
        f"{env1}_ST8": f"logs/{env1}-v0/diffusion_plan/pen_H32_T16_S{{seed}}_d0.99_ST8/0/rollout.json",
        f"{env1}_ST16": f"logs/{env1}-v0/diffusion_plan/pen_H32_T16_S{{seed}}_d0.99_ST16/0/rollout.json",
        # kitchen-partial environment (prefix: kit)
        f"{env2}_ST1": f"logs/{env2}-v0/diffusion_plan/kit_H32_T16_S{{seed}}_d0.99_ST1/0/rollout.json",
        f"{env2}_ST2": f"logs/{env2}-v0/diffusion_plan/kit_H32_T16_S{{seed}}_d0.99_ST2/0/rollout.json",
        f"{env2}_ST4": f"logs/{env2}-v0/diffusion_plan/kit_H32_T16_S{{seed}}_d0.99_ST4/0/rollout.json",
        f"{env2}_ST8": f"logs/{env2}-v0/diffusion_plan/kit_H32_T16_S{{seed}}_d0.99_ST8/0/rollout.json",
        f"{env2}_ST16": f"logs/{env2}-v0/diffusion_plan/kit_H32_T16_S{{seed}}_d0.99_ST16/0/rollout.json",
    }
    
    # Dictionary to hold results per seed
    results = {}

    # Loop over seed numbers
    for seed in range(150):
        results[seed] = {}
        for cat_name, path_template in categories.items():
            file_path = path_template.format(seed=seed)
            step = read_steps(file_path)
            if step is not None:
                mean, std_error, count_999 = analyze_steps(step)
                results[seed][cat_name] = {
                    'mean': mean,
                    'std_error': std_error,
                    'count_999': count_999
                }
            else:
                results[seed][cat_name] = None

    # Write the results to a CSV file
    output_csv = "step_analysis.csv"
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        header = ["seed"] + [f"{cat}_{metric}" for cat in categories.keys() 
                           for metric in ['mean', 'std_error', 'count_999']]
        writer.writerow(header)
        
        # Write one row per seed
        for seed in range(150):
            row = [seed]
            for cat in categories.keys():
                if results[seed][cat] is not None:
                    row.extend([
                        results[seed][cat]['mean'],
                        results[seed][cat]['std_error'],
                        results[seed][cat]['count_999']
                    ])
                else:
                    row.extend([None, None, None])
            writer.writerow(row)
    
    print(f"CSV file '{output_csv}' written successfully.")

    # 실험 결과가 있는 디렉토리 경로
    base_dir = Path('logs/plans')
    
    # 결과를 저장할 딕셔너리
    results = defaultdict(lambda: {
        'mean': 0,
        'std_error': 0,
        'count_999': 0,
        'total_episodes': 0
    })
    
    # 모든 실험 디렉토리 순회
    for exp_dir in base_dir.glob('*'):
        if not exp_dir.is_dir():
            continue
            
        # rollout.json 파일 찾기
        rollout_file = exp_dir / 'rollout.json'
        if not rollout_file.exists():
            continue
            
        try:
            step = read_steps(rollout_file)
            if step is not None:
                mean, std_error, count_999 = analyze_steps(step)
                results[exp_dir.name].update({
                    'mean': mean,
                    'std_error': std_error,
                    'count_999': count_999,
                    'total_episodes': 1
                })
        except Exception as e:
            print(f"Error processing {rollout_file}: {e}")
    
if __name__ == '__main__':
    main()
