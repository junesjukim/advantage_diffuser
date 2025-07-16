import gymnasium as gym
import numpy as np
import os
import env.frozen_lake_continuous

def generate_expert_action(state: np.ndarray, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generates an action based on the expert policy defined by the user.
    """
    action = 1.5*rng.standard_normal(2)
    direction_to_goal = goal - state
    norm = np.linalg.norm(direction_to_goal)
    direction_to_goal = direction_to_goal / (norm + 1e-2)
    action += 0.5 * direction_to_goal

    action_norm = np.linalg.norm(action)
    if action_norm > 1.0:
        action = action / action_norm
        
    return action


def generate_dataset(env: gym.Env, num_episodes: int, max_steps_per_episode: int):
    """
    Generates a dataset by running an expert policy in the environment.
    """
    goal = np.array([0.75, 0.75], dtype=np.float32)
    rng = np.random.default_rng(seed=0)

    # Data storage following D4RL format
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []

    successful_episodes = 0

    print(f"Generating dataset for {num_episodes} episodes...")

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=episode_idx)
        
        for step_idx in range(max_steps_per_episode):
            action = generate_expert_action(obs, goal, rng)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Store transition data
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(terminated)
            
            # A timeout occurs if the episode ends due to reaching max_steps
            timeout = (truncated or step_idx == max_steps_per_episode - 1) and not terminated
            timeouts.append(timeout)

            obs = next_obs

            if terminated or timeout:
                break

        # Check if the last reward was >= -0.25
        if reward >= -0.25:
            successful_episodes += 1
        
        if (episode_idx + 1) % 100 == 0:
            print(f"  ... finished episode {episode_idx + 1}/{num_episodes}")


    print(f"\nData generation complete.")
    print(f"Success rate: {successful_episodes}/{num_episodes} episodes had a final reward >= -0.25.")

    # Convert lists to numpy arrays
    dataset = {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_observations': np.array(next_observations, dtype=np.float32),
        'terminals': np.array(terminals, dtype=np.bool_),
        'timeouts': np.array(timeouts, dtype=np.bool_),
    }
    return dataset


if __name__ == "__main__":
    # --- Configuration ---
    NUM_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 100
    DATASET_FILENAME = "/home/junseolee/code/MDFS/frozen_lake_dataset.npz"

    # --- 1. Generate and Save Dataset ---
    env = gym.make('FrozenLakeContinuous-v1')
    
    dataset = generate_dataset(env, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    
    print(f"\nSaving dataset to {DATASET_FILENAME}...")
    np.savez(DATASET_FILENAME, **dataset)
    print("Dataset saved successfully.")

    # --- 2. Load and Verify Dataset ---
    print("\nLoading dataset for verification...")
    if os.path.exists(DATASET_FILENAME):
        loaded_data = np.load(DATASET_FILENAME)
        
        print("Dataset loaded successfully.")
        print("Keys in loaded data:", list(loaded_data.keys()))
        
        total_transitions = len(loaded_data['observations'])
        
        print(f"Total transitions stored: {total_transitions}")
        print("Verification successful!")
    else:
        print(f"Error: Dataset file '{DATASET_FILENAME}' not found.")

    env.close()
