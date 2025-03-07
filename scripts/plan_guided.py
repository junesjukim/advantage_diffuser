import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#-------------------------- conda env test -----------------------------------#
#-----------------------------------------------------------------------------#

import os

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
    # Add support for multiple diffusion steps during planning
    planning_steps: str = None  # Format: "16,8,4,2,1" for multiple steps

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
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

# Parse planning steps if provided
planning_steps = []
if args.planning_steps:
    planning_steps = [int(steps) for steps in args.planning_steps.split(',')]
else:
    # If not provided, use the n_diffusion_steps from args
    planning_steps = [args.n_diffusion_steps]

print(f"Planning with diffusion steps: {planning_steps}")

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# Function to run planning with a specific number of diffusion steps
def run_planning_with_steps(n_steps):
    # Adjust diffusion model to use the specified number of steps
    original_n_timesteps = diffusion.n_timesteps
    diffusion.adjust_diffusion_steps(n_steps)
    
    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(args.max_episode_length):

        if t % 10 == 0: 
            print(f"{args.savepath} - Steps: {n_steps}", flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f'Steps: {n_steps} | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values} | scale: {args.scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        logger.log(t, samples, state, rollout, suffix=f"_steps_{n_steps}")

        if terminal:
            break

        observation = next_observation

    # Reset diffusion model to original number of steps
    diffusion.reset_diffusion_steps()
    
    return t, score, total_reward, terminal

# Run planning for each specified number of diffusion steps
results = {}
for steps in planning_steps:
    print(f"\n{'='*50}\nRunning planning with {steps} diffusion steps\n{'='*50}")
    t, score, total_reward, terminal = run_planning_with_steps(steps)
    
    # Save results for this configuration
    results[steps] = {
        'timesteps': t,
        'score': score,
        'total_reward': total_reward,
        'terminal': terminal
    }
    
    # Write results to json file
    logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment, 
                 suffix=f"_steps_{steps}")

# Print summary of results
print("\n\n===== PLANNING RESULTS SUMMARY =====")
for steps, result in results.items():
    print(f"Diffusion Steps: {steps}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Total Reward: {result['total_reward']:.2f}")
    print(f"  Terminal: {result['terminal']}")
    print(f"  Timesteps: {result['timesteps']}")
    print("-" * 30)
