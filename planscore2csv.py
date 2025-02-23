#!/usr/bin/env python3
import os
import json
import csv
import math
import statistics

def read_score(file_path):
    """
    Attempts to read the JSON file at file_path and return the 'score' field.
    If the file does not exist or the field is missing, returns None.
    """
    if not os.path.exists(file_path):
        print(f"[Warning] File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            score = data.get("score")
            if score is None:
                print(f"[Warning] 'score' not found in {file_path}")
            return score
    except Exception as e:
        print(f"[Error] Could not read {file_path}: {e}")
        return None

def main():
    # Set environment
    env1 = "hopper"
    env2 = "walker2d"
    # Define the four categories with their corresponding file path templates.
    # The {seed} placeholder will be replaced with the seed number.
    categories = {
        "hopper_expert_T1": f"logs/{env1}-medium-expert-v2/plans/H4_T1_S{{seed}}_d0.99/0/rollout.json",
        "hopper_expert_T2": f"logs/{env1}-medium-expert-v2/plans/H4_T2_S{{seed}}_d0.99/0/rollout.json",
        "hopper_expert_T4": f"logs/{env1}-medium-expert-v2/plans/H4_T4_S{{seed}}_d0.99/0/rollout.json",
        "hopper_expert_T8": f"logs/{env1}-medium-expert-v2/plans/H4_T8_S{{seed}}_d0.99/0/rollout.json",
        "hopper_expert_T20": f"logs/{env1}-medium-expert-v2/plans/H4_T20_S{{seed}}_d0.99/0/rollout.json",
        "hopper_replay_T1": f"logs/{env1}-medium-replay-v2/plans/H4_T1_S{{seed}}_d0.99/0/rollout.json",
        "hopper_replay_T2": f"logs/{env1}-medium-replay-v2/plans/H4_T2_S{{seed}}_d0.99/0/rollout.json",
        "hopper_replay_T4": f"logs/{env1}-medium-replay-v2/plans/H4_T4_S{{seed}}_d0.99/0/rollout.json",
        "hopper_replay_T8": f"logs/{env1}-medium-replay-v2/plans/H4_T8_S{{seed}}_d0.99/0/rollout.json",
        "hopper_replay_T20": f"logs/{env1}-medium-replay-v2/plans/H4_T20_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_expert_T1": f"logs/{env2}-medium-expert-v2/plans/H4_T1_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_expert_T2": f"logs/{env2}-medium-expert-v2/plans/H4_T2_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_expert_T4": f"logs/{env2}-medium-expert-v2/plans/H4_T4_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_expert_T8": f"logs/{env2}-medium-expert-v2/plans/H4_T8_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_expert_T20": f"logs/{env2}-medium-expert-v2/plans/H4_T20_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_replay_T1": f"logs/{env2}-medium-replay-v2/plans/H4_T1_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_replay_T2": f"logs/{env2}-medium-replay-v2/plans/H4_T2_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_replay_T4": f"logs/{env2}-medium-replay-v2/plans/H4_T4_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_replay_T8": f"logs/{env2}-medium-replay-v2/plans/H4_T8_S{{seed}}_d0.99/0/rollout.json",
        "walker2d_replay_T20": f"logs/{env2}-medium-replay-v2/plans/H4_T20_S{{seed}}_d0.99/0/rollout.json",
    }
    
    # Dictionary to hold results per seed.
    # Each key is a seed (0 to 149) and the value is another dict mapping category names to scores.
    results = {}

    # Loop over seed numbers.
    for seed in range(150):
        results[seed] = {}
        for cat_name, path_template in categories.items():
            file_path = path_template.format(seed=seed)
            score = read_score(file_path)
            results[seed][cat_name] = score

    # Compute overall averages and standard errors for each category (ignoring None values)
    averages = {}
    std_errors = {}
    for cat in categories.keys():
        valid_scores = [results[seed][cat] for seed in range(150) if results[seed][cat] is not None]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            averages[cat] = avg
            # Compute standard error if more than one score is available.
            if len(valid_scores) > 1:
                stdev = statistics.stdev(valid_scores)
                std_errors[cat] = stdev / math.sqrt(len(valid_scores))
            else:
                std_errors[cat] = 0.0
        else:
            averages[cat] = None
            std_errors[cat] = None

    # Write the results to a CSV file.
    output_csv = "results.csv"
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row: seed number and the category names.
        header = ["seed"] + list(categories.keys())
        writer.writerow(header)
        
        # Write one row per seed.
        for seed in range(150):
            row = [seed] + [results[seed][cat] for cat in categories.keys()]
            writer.writerow(row)
        
        # Append an extra row with overall averages for each category.
        avg_row = ["Average"] + [averages[cat] for cat in categories.keys()]
        writer.writerow(avg_row)
        
        # Append an extra row with standard errors for each category.
        std_err_row = ["StdError"] + [std_errors[cat] for cat in categories.keys()]
        writer.writerow(std_err_row)
    
    print(f"CSV file '{output_csv}' written successfully.")

if __name__ == '__main__':
    main()
