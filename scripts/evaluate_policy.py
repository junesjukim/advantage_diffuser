# -----------------------------------------------------------------------------#
# -------------------------- conda env test -----------------------------------#
# -----------------------------------------------------------------------------#
import os
import re
import wandb
import sys
import importlib
import torch
import numpy as np
import imageio
import subprocess
import time

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
from diffuser.models.values import ValueFunction
from diffuser.guides.core import ValueGuide
from diffuser.sampling.policies import GuidedPolicy

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # --- General ---
    dataset: str = "walker2d-medium-replay-v2"
    config: str = None # For auto-detection
    # --- Evaluation ---
    n_runs: int = 10
    max_episode_length: int = 1000
    # --- Guidance ---
    guidance_type: str = "value" # 'value' or 'advantage'
    scale: float = 1.0 # Renamed from guide_scale for consistency
    # --- Sampling ---
    n_samples: int = 64 # for planning
    n_guide_steps: int = 2
    t_stopgrad: int = 2
    scale_grad_by_std: bool = True
    # --- Visualization & Logging ---
    save_video: bool = False
    vis_freq: int = 100
    max_render: int = 8
    use_wandb: bool = False
    wandb_project: str = "diffuser_evaluation"


def get_config_path(args):
    if args.config: return args.config
    dataset_name = args.dataset.lower()
    if 'scene-play' in dataset_name or 'metaworld' in dataset_name or 'ogbench' in dataset_name:
        return 'config.ogbench'
    return 'config.d4rl'

def setup_virtual_display():
    if 'DISPLAY' not in os.environ:
        print("Setting up virtual display...")
        subprocess.run(['pkill', 'Xvfb'], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
        os.environ['DISPLAY'] = ':100.0'
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        time.sleep(2)

args = Parser().parse_args('plan')
set_model_mode(args.prefix)
config_path = get_config_path(args)
config_module = importlib.import_module(config_path)
is_ogbench = "ogbench" in config_path

if is_ogbench:
    setup_virtual_display()

# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

# Load diffusion model (policy)
print(f"Loading diffusion model from: {args.diffusion_loadpath} at epoch {args.diffusion_epoch}")
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# Load guidance model (value or advantage)
print(f"Loading guidance model ({args.guidance_type}) from: {args.value_loadpath} at epoch {args.value_epoch}")
guidance_model_path = os.path.join(args.loadbase, args.dataset, args.value_loadpath, f'state_{args.value_epoch}.pt')
print(f"Loading guidance checkpoint from: {guidance_model_path}")

guidance_model = ValueFunction(
    horizon=args.horizon,
    transition_dim=dataset.observation_dim + dataset.action_dim,
    cond_dim=dataset.observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
).to(args.device)

checkpoint = torch.load(guidance_model_path, map_location=args.device)
guidance_model.load_state_dict(checkpoint['ema']) # Use EMA state for better performance
guidance_model.eval()

# -----------------------------------------------------------------------------#
# ---------------------------- setup policy & logger --------------------------#
# -----------------------------------------------------------------------------#

guide = ValueGuide(guidance_model, normalizer=dataset.normalizer, verbose=False)

policy_config = utils.Config(
    GuidedPolicy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
policy = policy_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)
logger = logger_config()

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

if is_ogbench:
    import ogbench
    env, _, _ = ogbench.make_env_and_datasets(
        args.dataset,
        render_mode='rgb_array' if args.save_video else 'none',
        width=640, height=480
    )
else:
    env = dataset.env

total_rewards = []
total_scores = []
success_count = 0

for run_id in range(args.n_runs):
    print(f"\n{'='*50}\n[ Run {run_id + 1} / {args.n_runs} ]\n{'='*50}")

    if is_ogbench:
        obs, info = env.reset(options={'task_id': args.task_id, 'render_goal': True})
        goal = info['goal']
    else:
        obs = env.reset()

    rollout = [obs.copy()]
    total_reward = 0
    done = False
    step = 0

    video_writer = None
    if args.save_video:
        video_path = os.path.join(args.savepath, f"run_{run_id + 1}.mp4")
        video_writer = imageio.get_writer(video_path, fps=30)

    while not done and step < args.max_episode_length:
        conditions = {0: obs}
        if is_ogbench:
            conditions[diffusion.horizon - 1] = goal

        action, samples = policy(conditions, batch_size=args.n_samples, verbose=args.verbose)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if not is_ogbench:
            score = env.get_normalized_score(total_reward)
            print(f"t: {step:4d} | r: {reward:8.2f} | R: {total_reward:8.2f} | score: {score:.4f}", flush=True)
        
        rollout.append(obs.copy())
        
        if video_writer:
            if is_ogbench:
                frame = env.render()
            else: # d4rl
                frame = logger.render_rollout(rollout, fps=80)
            video_writer.append_data(frame)

        step += 1

    # End of episode
    if video_writer:
        video_writer.close()
        print(f"Saved video to {video_path}")

    total_rewards.append(total_reward)

    if is_ogbench:
        is_success = info.get('success', False)
        if is_success:
            success_count += 1
        current_success_rate = (success_count / (run_id + 1)) * 100
        print(f"Run {run_id + 1}: {'Success' if is_success else 'Failure'} | Current Success Rate: {current_success_rate:.2f}%")
    else:
        final_score = env.get_normalized_score(total_reward)
        total_scores.append(final_score)
        print(f"Run {run_id + 1}: Final Score: {final_score:.4f}")

# Final results
print(f"\n\n{'='*50}\n[ Final Evaluation Results ]\n{'='*50}")
print(f"Total runs: {args.n_runs}")
if is_ogbench:
    final_success_rate = (success_count / args.n_runs) * 100 if args.n_runs > 0 else 0
    print(f"Successful runs: {success_count}")
    print(f"Final Success Rate: {final_success_rate:.2f}%")
else:
    mean_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    print(f"Average Score: {mean_score:.4f} ± {std_score:.4f}")

if 'DISPLAY' in os.environ and ':100' in os.environ['DISPLAY']:
    subprocess.run(['pkill', 'Xvfb'], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
