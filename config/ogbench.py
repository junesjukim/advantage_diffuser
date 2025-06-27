# This file contains settings for OGBench datasets.
from diffuser.datasets.ogbench import OgbenchDataset

# Specify the Dataset class to be used.
DatasetClass = OgbenchDataset

# OGBench training does not require a renderer.
use_renderer = False
