from collections import namedtuple
import numpy as np
import torch
from .sequence import SequenceDataset, Batch, ValueBatch
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

# lazy import
import ogbench

class OgbenchDataset(SequenceDataset):
    '''
    https://github.com/og-bench/og-bench?tab=readme-ov-file
    '''

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
        returns_scale=1.0,
        normed=False
    ):
        self.env_name = env_name
        self.env, self.dataset, self.sequence_dataset = ogbench.make_env_and_datasets(self.env_name, render_mode='rgb_array')

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, self.env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.observation_dim = self.dataset['observations'].shape[1]
        self.action_dim = self.dataset['actions'].shape[1]
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
        terminals = self.dataset['terminals']
        start_idx = 0
        
        for i in range(len(terminals)):
            if terminals[i]:
                episode = {
                    'observations': self.dataset['observations'][start_idx:i+1],
                    'actions': self.dataset['actions'][start_idx:i+1],
                    'rewards': self.dataset['rewards'][start_idx:i+1],
                    'terminals': self.dataset['terminals'][start_idx:i+1],
                    'next_observations': self.dataset['next_observations'][start_idx:i+1],
                    'masks': self.dataset['masks'][start_idx:i+1],
                    'timeouts': np.zeros_like(self.dataset['terminals'][start_idx:i+1])
                }
                episodes.append(episode)
                start_idx = i + 1
                
        return episodes
    
    def get_conditions(self, observations):
        '''
            conditions are supplied normalized
        '''
        return {0: observations[0]}
    
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

# NOTE: The OGBenchValueDataset class below is now obsolete and has been removed.
# Its functionality is replaced by the more generic 'diffuser/datasets/advantage.py:AdvantageDataset'. 