# This file contains settings for OGBench datasets.
from diffuser.datasets.ogbench import OgbenchDataset
from copy import deepcopy

# Specify the Dataset class to be used.
DatasetClass = OgbenchDataset

# OGBench training does not require a renderer.
use_renderer = False

try:
    from config.locomotion import base as locomotion_base
    # OGBench 또한 기본 파라미터를 locomotion 베이스로부터 상속
    base = deepcopy(locomotion_base)
except Exception:
    print("No config.locomotion found, using minimal empty config for ogbench")
    base = {
        "values": {},
        "plan": {},
    }
