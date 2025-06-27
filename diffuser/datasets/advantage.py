import torch
from torch.utils.data import Dataset

class AdvantageDataset(Dataset):
    """
    A wrapper dataset that calculates and provides advantages for a given base dataset.

    This dataset takes a base dataset object (like SequenceDataset for D4RL or 
    OgbenchDataset for OGBench) and pre-computes the advantage A(s,a) = Q(s,a) - V(s) 
    for all state-action pairs using pre-trained Q and V networks.
    """
    def __init__(self, base_dataset, q_network, v_network, device='cuda:0'):
        """
        Args:
            base_dataset: An initialized dataset object (e.g., SequenceDataset).
            q_network: A pre-trained Q-network model.
            v_network: A pre-trained V-network model.
            device: The device to perform computations on.
        """
        self.base_dataset = base_dataset
        self.q_network = q_network.to(device)
        self.v_network = v_network.to(device)
        self.device = device
        
        # The 'values' attribute will store the computed advantages.
        # This aligns with how the original Value-based dataset stored returns,
        # allowing for minimal changes in the training loop.
        print("Pre-computing advantages for the dataset...")
        self.values = self._precompute_advantages()
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
            
            # Extract observations and actions
            # Assumes the data is a dictionary, common in diffuser project.
            observations = torch.from_numpy(trajectory_data['observations']).to(self.device)
            actions = torch.from_numpy(trajectory_data['actions']).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(observations, actions)
                v_values = self.v_network(observations)
                advantages = q_values - v_values
            
            all_advantages.append(advantages.cpu().numpy())
        
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

        # The computed advantages are stored as a list of arrays. We need to 
        # add the corresponding advantage to the returned dictionary.
        # The trainer in diffuser expects a 'values' key in the batch.
        data['values'] = self.values[idx]

        return data

    @property
    def observation_dim(self):
        return self.base_dataset.observation_dim

    @property
    def action_dim(self):
        return self.base_dataset.action_dim
    
    @property
    def normalizer(self):
        return self.base_dataset.normalizer
