from collections import namedtuple
import numpy as np
import torch
from .sequence import SequenceDataset, Batch, ValueBatch
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

# lazy import
import ogbench

class OGBenchGoalDataset(SequenceDataset):
    def __init__(
        self,
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        seed=None,
        discount=0.99,
        normed=False
    ):
        self.env, self.train_dataset, self.val_dataset = ogbench.make_env_and_datasets(
            env_name,
            render_mode='rgb_array'
        )
        
        self.observation_dim = self.train_dataset['observations'].shape[-1]
        self.action_dim = self.train_dataset['actions'].shape[-1]
        self.max_path_length = max_path_length
        self.horizon = horizon
        self.use_padding = use_padding
        
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False # This will be set to True later if normed is True
        
        self.episodes = self._split_episodes()
        
        self.fields = ReplayBuffer(
            max_n_episodes,
            max_path_length,
            termination_penalty
        )
        
        for episode in self.episodes:
            self.fields.add_path(episode)
        self.fields.finalize()
        
        self.normalizer = DatasetNormalizer(
            self.fields,
            normalizer,
            path_lengths=self.fields['path_lengths']
        )
        
        self.indices = self.make_indices(
            self.fields.path_lengths,
            horizon
        )
        
        self.n_episodes = self.fields.n_episodes
        self.path_lengths = self.fields.path_lengths
        
        self.normalize()
        
        print(self.fields)
        
        # Now that self.indices is initialized, we can call _get_bounds()
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True
        
    def _split_episodes(self):
        episodes = []
        terminals = self.train_dataset['terminals']
        start_idx = 0
        
        for i in range(len(terminals)):
            if terminals[i]:
                episode = {
                    'observations': self.train_dataset['observations'][start_idx:i+1],
                    'actions': self.train_dataset['actions'][start_idx:i+1],
                    'rewards': self.train_dataset['rewards'][start_idx:i+1],
                    'terminals': self.train_dataset['terminals'][start_idx:i+1],
                    'next_observations': self.train_dataset['next_observations'][start_idx:i+1],
                    'masks': self.train_dataset['masks'][start_idx:i+1],
                    'timeouts': np.zeros_like(self.train_dataset['terminals'][start_idx:i+1])
                }
                episodes.append(episode)
                start_idx = i + 1
                
        return episodes
    
    def get_conditions(self, observations):
        return {
            0: observations[0],
            self.horizon - 1: observations[-1]
        }
    
    def normalize(self, keys=['observations', 'actions']):
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    
    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        
        total = len(self.indices)
        for i in range(total):
            if i % 100 == 0:
                print(f'\r[ datasets/sequence ] Getting value dataset bounds... {i}/{total}', end='', flush=True)
            
            value = self._compute_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        
        print(f'\r[ datasets/sequence ] Getting value dataset bounds... {total}/{total} ✓')
        return vmin, vmax

    def normalize_value(self, value):
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        normed = normed * 2 - 1
        return normed

    def _compute_value(self, idx):
        raise NotImplementedError("This should be implemented by subclasses")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        return Batch(trajectories, conditions)


class OGBenchValueDataset(OGBenchGoalDataset):
    def __init__(
        self,
        env_name,
        horizon,
        normalizer,
        preprocess_fns,
        max_path_length,
        max_n_episodes,
        termination_penalty,
        use_padding,
        seed=None,
        discount=0.99,
        normed=False,
        q_network=None,
        v_network=None,
        device='cuda'
    ):
        self.q_network = q_network
        self.v_network = v_network
        self.device = device
        
        super().__init__(
            env_name=env_name,
            horizon=horizon,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            termination_penalty=termination_penalty,
            use_padding=use_padding,
            seed=seed,
            discount=discount,
            normed=normed
        )
        
    def to(self, device):
        self.device = device
        if self.q_network is not None:
            self.q_network = self.q_network.to(device)
        if self.v_network is not None:
            self.v_network = self.v_network.to(device)
        return self
    
    def _compute_value(self, idx):
        path_ind, start, end = self.indices[idx]
        
        observations = self.fields.observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            act_tensor = torch.FloatTensor(actions).to(self.device)
            
            q1, q2 = self.q_network(obs_tensor, act_tensor)
            q_values = torch.min(q1, q2)
            
            v_values = self.v_network(obs_tensor)
            
            advantages = q_values - v_values
            
            discounts = torch.FloatTensor(self.discounts[:len(advantages)]).to(self.device)
            advantage_sum = (discounts * advantages).sum()
            
            return advantage_sum.cpu().item()
    
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        
        value = self._compute_value(idx)
        
        if self.normed:
            value = self.normalize_value(value)
        
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch 