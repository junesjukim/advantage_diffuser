# This file contains settings for D4RL datasets.
from diffuser.datasets.sequence import SequenceDataset

# Specify the Dataset class to be used.
DatasetClass = SequenceDataset

# D4RL environments use a renderer during training for visualization.
use_renderer = True

# -----------------------------------------------------------------------------#
# Provide a `base` dict with a 'values' entry so that utils.setup.Parser.read_config
# can parse this config module without raising AttributeError. We reuse the
# values configuration from `config.locomotion` as a reasonable default.
# -----------------------------------------------------------------------------#

from copy import deepcopy

try:
    from config.locomotion import base as locomotion_base
    base = {
        "values": deepcopy(locomotion_base["values"]),
    }
except Exception:
    # Fallback: minimal empty config - sufficient for train_advantage_model.py
    print("No config.locomotion found, using minimal empty config")
    base = {"values": {}}
