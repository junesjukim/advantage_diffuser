#!/usr/bin/env python3
import csv
import math
import statistics

def main():
    input_csv = "diffuser48.csv"
    output_csv = "diffuser48_with_stderr.csv"

    # Lists to store values from each column (only from data rows)
    expert_T4 = []
    expert_T8 = []
    replay_T4 = []
    replay_T8 = []
    data_rows = []  # to store each valid row

    # Read the CSV file.
    # (We assume that rows with a nonnumeric "seed" label—like "Average"—should be skipped.)
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed_val = row["seed"]
            try:
                # Only process rows where the seed can be converted to an integer
                int(seed_val)
            except ValueError:
                continue  # skip rows like "Average" if present

            try:
                et4 = float(row["expert_T4"])
                et8 = float(row["expert_T8"])
                rt4 = float(row["replay_T4"])
                rt8 = float(row["replay_T8"])
            except Exception as e:
                print(f"Skipping row with seed {seed_val}: {e}")
                continue

            expert_T4.append(et4)
            expert_T8.append(et8)
            replay_T4.append(rt4)
            replay_T8.append(rt8)

            # Save the row (we keep the seed and the original values)
            data_rows.append({
                "seed": seed_val,
                "expert_T4": et4,
                "expert_T8": et8,
                "replay_T4": rt4,
                "replay_T8": rt8
            })

    n = len(expert_T4)
    if n == 0:
        print("No data rows found. Exiting.")
        return

    # Compute averages
    avg_et4 = sum(expert_T4) / n
    avg_et8 = sum(expert_T8) / n
    avg_rt4 = sum(replay_T4) / n
    avg_rt8 = sum(replay_T8) / n

    # Compute standard errors (using sample standard deviation)
    if n > 1:
        se_et4 = statistics.stdev(expert_T4) / math.sqrt(n)
        se_et8 = statistics.stdev(expert_T8) / math.sqrt(n)
        se_rt4 = statistics.stdev(replay_T4) / math.sqrt(n)
        se_rt8 = statistics.stdev(replay_T8) / math.sqrt(n)
    else:
        se_et4 = se_et8 = se_rt4 = se_rt8 = 0.0

    # Write the new CSV file.
    # We rename the columns as follows:
    #   expert_T4  -> expert_T1
    #   expert_T8  -> expert_T2
    #   replay_T4  -> replay_T1
    #   replay_T8  -> replay_T2
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["seed", "expert_T4", "expert_T8", "replay_T4", "replay_T8"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write each data row (copying the values under new names)
        for row in data_rows:
            writer.writerow({
                "seed": row["seed"],
                "expert_T4": row["expert_T4"],
                "expert_T8": row["expert_T8"],
                "replay_T4": row["replay_T4"],
                "replay_T8": row["replay_T8"],
            })

        # Append a row for the averages
        writer.writerow({
            "seed": "Average",
            "expert_T4": avg_et4,
            "expert_T8": avg_et8,
            "replay_T4": avg_rt4,
            "replay_T8": avg_rt8,
        })

        # Append a row for the standard errors
        writer.writerow({
            "seed": "StdError",
            "expert_T4": se_et4,
            "expert_T8": se_et8,
            "replay_T4": se_rt4,
            "replay_T8": se_rt8,
        })

    print(f"Output written to {output_csv}")

if __name__ == "__main__":
    main()
