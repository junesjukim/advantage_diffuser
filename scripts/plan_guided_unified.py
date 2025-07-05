# -----------------------------------------------------------------------------#
# ---------------------- Unified Guided Planning Script -----------------------#
# -----------------------------------------------------------------------------#
import os
import re
import time
import subprocess
import imageio
import wandb

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
import ogbench

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    """Argument parser extended for unified planning across benchmarks."""

    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion"
    wandb_project: str = "diffuser_unified_planning"

    # unified extras
    benchmark: str = "d4rl"  # 'd4rl' | 'ogbench'
    save_video: bool = False


args = Parser().parse_args("plan")

# -----------------------------------------------------------------------------#
# ----------------------------- wandb initialisation --------------------------#
# -----------------------------------------------------------------------------#
train_seed_match = re.search(r'_S(\d+)', args.diffusion_loadpath)
train_seed = int(train_seed_match.group(1)) if train_seed_match else -1

value_seed_match = re.search(r'_S(\d+)', args.value_loadpath)
value_seed = int(value_seed_match.group(1)) if value_seed_match else -1

plan_seed = args.seed

# Initialize wandb
wandb.init(
    project=args.wandb_project,
    config=vars(args),
    name=f"{args.dataset.replace('-v0', '')}_tr{train_seed}_vs{value_seed}_ps{plan_seed}",
    group=f"TR{train_seed}_VS{value_seed}_{args.dataset.replace('-v0', '')}",
    tags=[args.dataset.replace('-v0', ''), f"train_seed_{train_seed}", f"value_seed_{value_seed}", f"plan_seed_{plan_seed}"],
    reinit=True,
)
wandb.config.update({
    "train_seed": train_seed,
    "value_seed": value_seed,
    "plan_seed": plan_seed
})







# -----------------------------------------------------------------------------#
# ------------------------------ helper functions -----------------------------#
# -----------------------------------------------------------------------------#

def setup_headless_display():
    """Launch Xvfb if DISPLAY is not available (for off-screen rendering)."""
    if 'DISPLAY' not in os.environ:
        subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
        os.environ['DISPLAY'] = ':100.0'
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'


def run_episode(env, policy, args, logger=None, diffusion_exp=None, value_exp=None):
    """Execute a single episode and return collected metrics."""

    # gym / gymnasium reset compatibility
    reset_out = env.reset()
    observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    info = reset_out[2] if isinstance(reset_out, tuple) and len(reset_out) > 2 else {}

    # video setup
    video_writer = None
    if args.save_video:
        setup_headless_display()
        video_path = os.path.join(args.savepath, 'episode.mp4')
        video_writer = imageio.get_writer(video_path, fps=30)

    total_reward = 0.0

    for t in range(args.max_episode_length):
        if t % 10 == 0:
            print(args.savepath, flush=True)

        # current observation only
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        # env step (gym / gymnasium compatible)
        step_out = env.step(action)
        if len(step_out) == 5: # gymnasium API
            observation, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else: # classic gym API
            observation, reward, done, info = step_out

        total_reward += reward
        try:
            score = env.get_normalized_score(total_reward)
        except AttributeError:
            score = total_reward

        print(f"t: {t} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f}", flush=True)

        log_dict = {
            'step_reward': reward,
            'total_reward': total_reward,
        }
        # diffuser 모드일 때만 세부 값 로깅
        if args.benchmark != 'ogbench':
            log_dict.update({
                'score': score,
                'value_mean': samples.values.mean().item() if hasattr(samples, 'values') else 0,
            })

        wandb.log(log_dict, step=t)

        if video_writer:
            video_writer.append_data(env.render())

        if done:
            break

    if video_writer:
        video_writer.close()
        wandb.log({'episode_video': wandb.Video(video_path, fps=30)})

    if logger and diffusion_exp and value_exp:
        logger.finish(t, score, total_reward, done, diffusion_exp, value_exp)

    metrics = {
        'total_reward': total_reward,
        'score': score,
        'episode_length': t + 1,
        'is_terminal': done,
    }
    if 'success' in info:
        metrics['success'] = info['success']

    # ogbench의 단일 성공(boolean)도 별도 기록
    if 'success' in metrics:
        wandb.log({'success': metrics['success']})

    return metrics


# -----------------------------------------------------------------------------#
# ----------------------------- loading components ---------------------------#
# -----------------------------------------------------------------------------#

set_model_mode(args.prefix)

print(f"========== n_sample_timesteps: {args.n_sample_timesteps} ==========", flush=True)

diffusion_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
)

value_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.value_loadpath,
    epoch=args.value_epoch,
    seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
)

# compatibility check
utils.check_compatibility(diffusion_experiment, value_experiment)

# alias handles
diffusion = diffusion_experiment.ema

dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# guide & policy (plan_guided 방식)
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)
policy_config = utils.Config(
    args.policy,
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
logger = logger_config()
policy = policy_config()

# optional ogbench env override
if args.benchmark == 'ogbench':
    assert ogbench is not None, "ogbench package not available"
    env, _, _ = ogbench.make_env_and_datasets(
        args.dataset,
        render_mode='rgb_array' if args.save_video else 'none',
        width=640,
        height=480
    )
else:
    env = dataset.env

# device alignment
device = next(diffusion.parameters()).device
print(f"model device: {device}")
for m in (diffusion, value_function, guide):
    m.to(device)

# -----------------------------------------------------------------------------#
# --------------------------------- run & log ---------------------------------#
# -----------------------------------------------------------------------------#

metrics = run_episode(env, policy, args, logger, diffusion_experiment, value_experiment)

wandb.summary.update(metrics)
wandb.finish() 