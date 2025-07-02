import torch
from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple

# global namedtuple definitions (picklable by DataLoader)
BatchAdv = namedtuple("BatchAdv", "trajectories conditions values")

class AdvantageDataset(Dataset):
    """
    A wrapper dataset that calculates and provides advantages for a given base dataset.

    This dataset takes a base dataset object (like SequenceDataset for D4RL or 
    OgbenchDataset for OGBench) and pre-computes the advantage A(s,a) = Q(s,a) - V(s) 
    for all state-action pairs using pre-trained Q and V networks.
    """
    def __init__(self, base_dataset, q_network, v_network, device='cuda:0', eps=1e-3, discount=0.99):
        """
        Args:
            base_dataset: An initialized dataset object (e.g., SequenceDataset).
            q_network: A pre-trained Q-network model.
            v_network: A pre-trained V-network model.
            device: The device to perform computations on.
            eps: Epsilon for IQL normalization.
            discount: Discount factor for computing discounted advantages.
        """
        self.base_dataset = base_dataset
        self.q_network = q_network.to(device)
        self.v_network = v_network.to(device)
        self.device = device
        self.discount = discount
        
        # Compute normalization statistics (IQL-style z-score) from base_dataset
        mean, std = self._compute_stats(base_dataset, eps=eps)
        self.state_mean = torch.as_tensor(mean, device=device, dtype=torch.float32)
        self.state_std = torch.as_tensor(std, device=device, dtype=torch.float32)
        
        # The 'values' attribute will store the computed advantages.
        # This aligns with how the original Value-based dataset stored returns,
        # allowing for minimal changes in the training loop.
        print("Pre-computing advantages for the dataset...")
        self.advantages = self._precompute_advantages()
        print("...Advantages pre-computation complete.")

    def _precompute_advantages(self):
        """
        Iterates through the base dataset to compute advantages for all trajectories.
        """
        all_advantages = []
        for i in range(len(self.base_dataset)):
            # Fetches a single trajectory from the base dataset.
            # The base_dataset is expected to return a dictionary or an object
            # with 'observations' and 'actions'.
            trajectory_data = self.base_dataset[i]
            
            # Extract observations and actions depending on base_dataset output format
            if isinstance(trajectory_data, dict) and 'observations' in trajectory_data:
                observations_np = trajectory_data['observations']
                actions_np = trajectory_data['actions']
            else:
                # Assume SequenceDataset Batch: .trajectories with shape [H, A+O]
                traj = trajectory_data.trajectories  # numpy array [H, A+O]
                obs_dim = self.base_dataset.observation_dim
                act_dim = self.base_dataset.action_dim
                actions_np = traj[:, :act_dim]
                observations_np = traj[:, act_dim:]

            # Unnormalize observations if dataset provided normalized values
            if hasattr(self.base_dataset, 'normalizer'):
                try:
                    observations_np = self.base_dataset.normalizer.unnormalize(observations_np, key='observations')
                    actions_np = self.base_dataset.normalizer.unnormalize(actions_np, key='actions')
                except Exception:
                    # If unnormalization fails, assume inputs are already raw
                    pass

            # Convert to torch tensors
            observations = torch.from_numpy(observations_np).to(self.device)
            actions = torch.from_numpy(actions_np).to(self.device)

            # Apply IQL-style z-score normalization (match training stats)
            observations = (observations - self.state_mean) / (self.state_std)

            with torch.no_grad():
                q1, q2 = self.q_network(observations, actions)
                q_values = torch.min(q1, q2)  # [H,1]
                v_values = self.v_network(observations)  # [H,1]
                advantages = (q_values - v_values).cpu().numpy().squeeze(-1)  # shape (H,)

            all_advantages.append(advantages)
        
        # The base_dataset has a list of numpy arrays, we keep the same format
        return all_advantages

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Returns the original data point from the base dataset.
        # The trainer will later access the computed advantages via `dataset.values`.
        # To make it compatible with the batching process, we need to return
        # the same structure as the base dataset.
        data = self.base_dataset[idx]

        # We will convert any incoming sample to BatchAdv to ensure picklability

        if isinstance(data, dict):
            trajectories = np.concatenate([data['actions'], data['observations']], axis=-1)
            conditions = {0: data['observations'][0]}
        else:
            # data is SequenceDataset.Batch
            trajectories = data.trajectories
            conditions = data.conditions

        # compute discounted scalar advantage on the fly
        adv_seq = self.advantages[idx]
        discounts = self.discount ** np.arange(len(adv_seq))
        disc_adv = float((discounts * adv_seq).sum())
        value_scalar = np.array([disc_adv], dtype=np.float32)

        return BatchAdv(trajectories, conditions, value_scalar)

    @property
    def observation_dim(self):
        return self.base_dataset.observation_dim

    @property
    def action_dim(self):
        return self.base_dataset.action_dim
    
    @property
    def normalizer(self):
        return self.base_dataset.normalizer

    def _compute_stats(self, base_dataset, eps=1e-3):
        """Compute mean/std over ALL observations in the dataset, axis=0, matching
        ReplayBuffer.normalize_states. Prefer fast path using `fields.observations` if
        present (SequenceDataset). Fallback to iteration otherwise."""

        if hasattr(base_dataset, 'fields') and hasattr(base_dataset.fields, 'observations'):
            obs = base_dataset.fields.observations  # shape (n_eps, max_len, obs_dim)
            obs_flat = obs.reshape(-1, obs.shape[-1])
            mean = obs_flat.mean(axis=0, keepdims=True)
            std = obs_flat.std(axis=0, keepdims=True) + eps
            return mean, std

        # Fallback: iterate through dataset (slower)
        obs_list = []
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if isinstance(item, dict):
                obs_np = item['observations']
            else:
                traj = item.trajectories
                act_dim = base_dataset.action_dim
                obs_np = traj[:, act_dim:]
            obs_list.append(obs_np)

        obs_concat = np.concatenate(obs_list, axis=0)
        mean = obs_concat.mean(axis=0, keepdims=True)
        std = obs_concat.std(axis=0, keepdims=True) + eps
        return mean, std
