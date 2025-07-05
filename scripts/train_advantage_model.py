import sys
import os
import importlib

# Add parent directory to path to import from diffuser
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
import torch

# IQL and diffuser models import
try:
    from iql_pytorch.critic import Critic, ValueCritic
except ImportError as e:
    print(f"Error: Failed to import 'iql_pytorch.critic'. Make sure the iql_pytorch library is in your PYTHONPATH.")
    print(f"Original Error: {e}")
    sys.exit(1)

from diffuser.models import ValueFunction
from diffuser.datasets.advantage import AdvantageDataset # Use the new generic advantage dataset
from diffuser.models.diffusion import AdvantageValueDiffusion

# Optional wandb import for logging
try:
    import wandb
except ImportError:
    wandb = None

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = None # For auto-detection
    # --- Value learning specific parameters ---
    horizon: int = 32
    batch_size: int = 256
    learning_rate: float = 2e-4
    normalizer: str = 'DebugNormalizer'
    normed: bool = True
    use_padding: bool = True
    max_path_length: int = 1000
    # --- Paths to pre-trained IQL networks (NOW REQUIRED ARGUMENTS) ---
    q_path: str = "path/to/your/critic.pth"
    v_path: str = "path/to/your/value.pth"

def get_config_path(args):
    """
    Determines the configuration file path.
    1. If --config is provided, use it.
    2. Otherwise, auto-detect based on --dataset name.
    """
    if args.config is not None:
        print(f"[INFO] Using user-specified config: {args.config}")
        return args.config

    dataset_name = args.dataset.lower()
    if 'scene-play' in dataset_name or 'metaworld' in dataset_name or 'ogbench' in dataset_name:
        print(f"[INFO] Auto-detected OGBench dataset for '{args.dataset}'. Using 'config.ogbench'.")
        return 'config.ogbench'
    elif 'hopper' in dataset_name or 'walker' in dataset_name or 'antmaze' in dataset_name or 'kitchen' in dataset_name or 'pen' in dataset_name or 'hammer' in dataset_name or 'door' in dataset_name or 'relocate' in dataset_name:
        print(f"[INFO] Auto-detected D4RL dataset for '{args.dataset}'. Using 'config.d4rl'.")
        return 'config.d4rl'
    
    raise ValueError(
        f"Could not auto-detect config for dataset '{args.dataset}'. "
        "Please specify it manually with the --config flag (e.g., --config config.d4rl)."
    )

args = Parser().parse_args('values')

# Load configuration module
config_path = get_config_path(args)
config_module = importlib.import_module(config_path)

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

# Load base dataset using the auto-detected config
DatasetClass = config_module.DatasetClass
dataset_kwargs = {
    'horizon': args.horizon,
    'normalizer': args.normalizer,
    'preprocess_fns': args.preprocess_fns,
    'use_padding': args.use_padding,
    'max_path_length': args.max_path_length,
}
if "ogbench" in config_path:
    dataset_kwargs['env_name'] = args.dataset
else: # d4rl
    dataset_kwargs['env'] = args.dataset

base_dataset_config = utils.Config(DatasetClass, **dataset_kwargs)
base_dataset = base_dataset_config()

# Load pre-trained IQL networks
# These must be provided via command line arguments
print(f"Loading Q-network from: {args.q_path}")
print(f"Loading V-network from: {args.v_path}")

# Dynamically get state and action dimensions from the base dataset
state_dim = base_dataset.observation_dim
action_dim = base_dataset.action_dim

q_network = Critic(state_dim, action_dim).to(args.device)
v_network = ValueCritic(state_dim, 512, 5).to(args.device)  # match IQL training config
q_network.load_state_dict(torch.load(args.q_path, map_location=args.device))
v_network.load_state_dict(torch.load(args.v_path, map_location=args.device))
q_network.eval()
v_network.eval()

# Create the final AdvantageDataset by wrapping the base dataset (avoid pickling)
dataset = AdvantageDataset(
    base_dataset=base_dataset,
    q_network=q_network,
    v_network=v_network,
    device=args.device,
)

# save dataset config (lightweight) manually if desired
# utils.Config will pickle but we avoid unpicklable objects; here we only store parameters
dataset_cfg_simple = utils.Config(
    AdvantageDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    base_dataset='SequenceDataset',
    q_path=args.q_path,
    v_path=args.v_path,
    device=args.device,
    overwrite=True if hasattr(utils.Config, 'overwrite') else False,
)

# Obtain dims
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

# Model for learning advantage (same as value function model)
model_config = utils.Config(
    ValueFunction,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

# Diffusion model for advantage
diffusion_config = utils.Config(
    AdvantageValueDiffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

# Trainer
# First, sanitize the args dict so that all values are pickle-friendly (e.g., primitives or strings)
_simple_types = (int, float, bool, str, type(None))
wandb_cfg_safe = {k: (v if isinstance(v, _simple_types) else str(v)) for k, v in vars(args).items()}

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    # WandB options
    use_wandb=(wandb is not None),
    wandb_project='advantage_diffuser',
    wandb_run_name=f'{args.dataset}-{args.prefix}',
    wandb_config=wandb_cfg_safe,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, None) # No renderer needed

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')
print(f"Initial loss: {loss.item()}")

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

print("Training finished.")

# Gracefully finish wandb run if enabled
if 'trainer' in globals() and hasattr(trainer, 'wandb') and trainer.wandb is not None:
    trainer.wandb.finish() 