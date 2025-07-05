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
    # D4RL 설정은 기본적으로 locomotion 베이스를 그대로 사용합니다.
    # (diffusion / values / plan 모두 포함)
    base = deepcopy(locomotion_base)
except Exception:
    # locomotion 설정을 찾지 못한 경우, 최소한 values / plan 섹션만 제공
    print("No config.locomotion found, falling back to minimal config (values / plan)")
    base = {
        "values": {},
        "plan": {},
    }
