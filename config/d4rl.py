# This file contains settings for D4RL datasets.
from diffuser.datasets.d4rl import SequenceDataset

# Specify the Dataset class to be used.
DatasetClass = SequenceDataset

# D4RL environments use a renderer during training for visualization.
use_renderer = True
