# -----------------------------------------------------------------------------#
# -------------------------- conda env test -----------------------------------#
# -----------------------------------------------------------------------------#
import os
import importlib

import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "hopper-medium-expert-v2"
    config: str = None # Changed default to None for auto-detection


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


args = Parser().parse_args("diffusion")

# Set model mode (e.g., 'diffusion' or 'values')
set_model_mode(args.prefix)

# Load configuration module
config_path = get_config_path(args)
config_module = importlib.import_module(config_path)

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

# Get Dataset class and kwargs from config
DatasetClass = config_module.DatasetClass
dataset_kwargs = {
    'horizon': args.horizon,
    'normalizer': args.normalizer,
    'preprocess_fns': args.preprocess_fns,
    'use_padding': args.use_padding,
    'max_path_length': args.max_path_length,
}

# Add environment-specific kwargs
if "ogbench" in config_path:
    dataset_kwargs['env_name'] = args.dataset
else: # d4rl
    dataset_kwargs['env'] = args.dataset

dataset_config = utils.Config(
    DatasetClass,
    savepath=(args.savepath, "dataset_config.pkl"),
    **dataset_kwargs,
)

# Setup renderer if specified in config
renderer = None
if getattr(config_module, 'use_renderer', False):
    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, "render_config.pkl"),
        env=args.dataset,
    )
    renderer = render_config()

dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
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
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=args.n_steps_per_epoch)
