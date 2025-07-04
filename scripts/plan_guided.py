# -----------------------------------------------------------------------------#
# -------------------------- conda env test -----------------------------------#
# -----------------------------------------------------------------------------#
import os
import re
import wandb

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion"
    wandb_project: str = "diffuser_research-planning-repenkit"


args = Parser().parse_args("plan")

# Extract seeds for wandb logging
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


# setting model mode flowmatching or diffusion
set_model_mode(args.prefix)


# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#
## load diffusion model and value function from disk
print(
    f"========== n_sample_timesteps: {args.n_sample_timesteps} ==========", flush=True
)
diffusion_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
)
print("=" * 30 + str(args.n_sample_timesteps) + "=" * 30)
value_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.value_loadpath,
    epoch=args.value_epoch,
    seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
)


## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer


## initialize value guide
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

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

# 디바이스 확인 및 설정
device = next(diffusion.parameters()).device
print(f"모델 디바이스: {device}")

# 모든 모델을 동일한 디바이스로 이동
diffusion = diffusion.to(device)
value_function = value_function.to(device)
guide = guide.to(device)

total_reward = 0
for t in range(args.max_episode_length):
    if t % 10 == 0:
        print(args.savepath, flush=True)

    ## save state for rendering only
    # state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}

    action, samples = policy(
        conditions, batch_size=args.batch_size, verbose=args.verbose
    )

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
        f"values: {samples.values} | scale: {args.scale}",
        flush=True,
    )

    # wandb logging
    wandb.log({
        'step_reward': reward,
        'total_reward': total_reward,
        'score': score,
        'value_mean': samples.values.mean().item(),
    }, step=t)

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    # logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)

# wandb finish
wandb.summary['final_score'] = score
wandb.summary['final_total_reward'] = total_reward
wandb.summary['episode_length'] = t + 1
wandb.summary['is_terminal'] = terminal
wandb.finish()
