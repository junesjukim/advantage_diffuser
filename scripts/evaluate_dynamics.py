# -----------------------------------------------------------------------------#
# ---------------------- Dynamics Evaluation Script -----------------------#
# -----------------------------------------------------------------------------#
import os
import re
import time
import subprocess
import wandb
import gym
import numpy as np
import torch
import cv2

# -----------------------------------------------------------------------------#
# ---------------------- headless EGL / Xvfb setup ----------------------------#
# -----------------------------------------------------------------------------#
# dm_control‐based environments (e.g. kitchen) require a valid OpenGL context.
# We enforce EGL backend **before** MuJoCo is imported anywhere.

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'  # use headless EGL rendering

# Launch a virtual X-server when DISPLAY is absent (common on headless servers)
if 'DISPLAY' not in os.environ:
    subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
    os.environ['DISPLAY'] = ':100.0'

# -----------------------------------------------------------------------------#

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
import ogbench
import imageio

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    """Argument parser for dynamics evaluation."""

    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion"
    wandb_project: str = "diffuser_dynamics_scale_test"

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

scale = args.scale
# Initialize wandb
wandb.init(
    project=args.wandb_project,
    config=vars(args),
    name=f"{args.dataset.replace('-v0', '')}_tr{train_seed}_vs{value_seed}_ps{plan_seed}_sc{scale}",
    group=f"TR{train_seed}_VS{value_seed}_{args.dataset.replace('-v0' , '')}_SC{scale}",
    tags=[args.dataset.replace('-v0', ''), f"train_seed_{train_seed}", f"value_seed_{value_seed}", f"plan_seed_{plan_seed}", f"scale_{scale}"],
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
        os.environ['MUJOCO_EGL_NO_GLX'] = '1'
        os.environ['MUJOCO_EGL_NO_X11'] = '1'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

def run_episode(env, policy, args, logger, diffusion_exp, value_exp):
    """Execute a single episode, evaluate dynamics, and return metrics."""

    # gym / gymnasium reset compatibility
    reset_out = env.reset()
    observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    info = reset_out[2] if isinstance(reset_out, tuple) and len(reset_out) > 2 else {}

    total_reward = 0.0
    all_dynamics_data = []
    all_proportions_data = [] # For custom wandb chart

    # video setup
    video_writer = None
    if args.save_video:
        setup_headless_display()
        video_path = os.path.join(args.savepath, 'episode.mp4')
        video_writer = imageio.get_writer(video_path, fps=30)

    dataset = diffusion_exp.dataset
    obs_dim = dataset.observation_dim

    for t in range(args.max_episode_length):
        if t % 10 == 0:
            print(args.savepath, flush=True)

        # current observation only
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
        # env step (gym / gymnasium compatible)
        step_out = env.step(action)
        if len(step_out) == 5: # gymnasium API
            real_next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else: # classic gym API
            real_next_obs, reward, done, info = step_out

        # Dynamics evaluation
        # samples는 이미 가장 가치가 높은 궤적이 인덱스 0에 오도록 정렬되어 있습니다.
        # 따라서 첫 번째 궤적(`[0]`)의 다음 observation(`[1]`)을 예측값으로 사용합니다.
        predicted_next_obs = samples.observations[0, 1]
        dynamics_error_vector = (predicted_next_obs - real_next_obs)
        dynamics_error_norm = np.linalg.norm(dynamics_error_vector)
        
        # 각 차원별 에러 기여도 계산
        total_squared_error = dynamics_error_norm**2
        squared_error_per_dim = dynamics_error_vector**2
        error_proportion_per_dim = squared_error_per_dim / (total_squared_error + 1e-8)

        print(f"observations at step {t}:")
        print("np.linalg.norm(dynamics_error_vector): ", np.linalg.norm(dynamics_error_vector))
        print("np.linalg.norm(predicted_next_obs): ", np.linalg.norm(predicted_next_obs))
        print("np.linalg.norm(real_next_obs): ", np.linalg.norm(real_next_obs))

        # 상위 5개 에러 기여도 차원 및 값 출력
        top5_indices = np.argsort(error_proportion_per_dim)[-5:][::-1]
        print(f"Top 5 error proportions at step {t}:")
        for i in top5_indices:
            print(f"  - Dim {i:02d}: {error_proportion_per_dim[i]:.4f}")

        dynamics_data = {
            'pred_s_next': predicted_next_obs,
            'real_s_next': real_next_obs,
            'error_vector': dynamics_error_vector,
            'error_norm': dynamics_error_norm,
        }
        all_dynamics_data.append(dynamics_data)
        
        observation = real_next_obs
        total_reward += reward

        try:
            score = env.get_normalized_score(total_reward)
        except AttributeError:
            score = total_reward

        print(f"t: {t} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | dyn_err: {dynamics_error_norm:.4f}", flush=True)

        log_dict = {
            'step_reward': reward,
            'total_reward': total_reward,
            'dynamics_error_norm': dynamics_error_norm,
        }
        # Log dynamics error for each dimension
        for i in range(obs_dim):
            log_dict[f'dyn_err_dim_{i}'] = float(dynamics_error_vector[i])
            log_dict[f'dyn_err_proportion_dim_{i}'] = float(error_proportion_per_dim[i])
            all_proportions_data.append([t, i, error_proportion_per_dim[i]])
            
        # diffuser 모드일 때만 세부 값 로깅
        if args.benchmark != 'ogbench':
            log_dict.update({
                'score': score,
                'value_mean': samples.values.mean().item() if hasattr(samples, 'values') else 0,
            })

        wandb.log(log_dict, step=t)

        if video_writer:
            # MuJoCoRenderer 를 이용해 현재 observation 으로부터 이미지를 생성
            try:
                frame = None
                if args.dataset in ['kitchen-complete-v0', 'kitchen-partial-v0']:
                    # MuJoCo 직접 렌더링: width=640, height=480, 원하는 카메라 이름 사용
                    cid = 0                               # fixed 카메라
                    env.sim.model.cam_fovy[cid] = 80.0    # 시야각을 70도로 확대
                    env.sim.model.cam_pos [cid][2] = 2.2
                    frame = env.sim.render(1920, 2560, camera_id = cid)
                elif args.dataset in ['pen-cloned-v0', 'pen-human-v0', 'pen-expert-v0']:
                    frame = env.sim.render(2560, 1920, camera_name = 'fixed')
                    frame = frame[::-1]
                else:
                    frame = env.render()
                
                if frame is not None:
                    # OpenCV는 BGR을 사용하므로 RGB -> BGR 변환 후 텍스트 추가
                    frame = frame.copy() # read-only buffer일 수 있으므로 복사
                    text = f"t = {t}"
                    font_scale = 2
                    font_thickness = 3
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
                    
                    # 텍스트 위치 (우측 상단)
                    text_x = frame.shape[1] - text_size[0] - 30
                    text_y = text_size[1] + 30
                    
                    # 텍스트 추가
                    cv2.putText(frame, text, (text_x, text_y), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    video_writer.append_data(frame)

            except Exception as e:
                print(f"[warning] renderer.render failed: {e}")

        if done:
            break

    # Log custom line chart for error proportions in chunks
    if all_proportions_data:
        chunk_size = 10
        # Create a dictionary to hold data for each chunk, structured for line_series
        # {start_dim: {dim: ([timesteps], [contributions])}}
        chunked_data = {}

        # Distribute data into the new structure
        for timestep, dim, contribution in all_proportions_data:
            chunk_key = (dim // chunk_size) * chunk_size
            if chunk_key not in chunked_data:
                chunked_data[chunk_key] = {}

            dim_key = f"dim_{dim:02d}"
            if dim_key not in chunked_data[chunk_key]:
                chunked_data[chunk_key][dim_key] = ([], []) # (timesteps, contributions)
            
            chunked_data[chunk_key][dim_key][0].append(timestep)
            chunked_data[chunk_key][dim_key][1].append(contribution)

        # Log a chart for each chunk using line_series
        for start_dim, data_chunk in chunked_data.items():
            if not data_chunk:
                continue

            # Extract keys, xs, and ys from the processed data
            keys = sorted(data_chunk.keys())
            xs = [data_chunk[key][0] for key in keys]
            ys = [data_chunk[key][1] for key in keys]
            
            end_dim = min(start_dim + chunk_size, obs_dim)
            chart_title = f"Error Contribution (Dims {start_dim}-{end_dim-1})"
            
            wandb.log({
                f"error_contribution_dims_{start_dim}_{end_dim-1}": wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title=chart_title,
                    xname="Timestep"
                )
            })

    # Save all dynamics data
    if len(all_dynamics_data) > 0:
        save_path = os.path.join(args.savepath, 'dynamics_data.npz')
        np.savez(
            save_path,
            pred_s_next=np.stack([d['pred_s_next'] for d in all_dynamics_data]),
            real_s_next=np.stack([d['real_s_next'] for d in all_dynamics_data]),
            error=np.stack([d['error_vector'] for d in all_dynamics_data]),
            error_norm=np.array([d['error_norm'] for d in all_dynamics_data]),
        )
        wandb.save(save_path)

    if video_writer:
        video_writer.close()
        wandb.log({'episode_video': wandb.Video(video_path, fps=30, format="mp4")})

    if logger and diffusion_exp and value_exp:
        logger.finish(t, score, total_reward, done, diffusion_exp, value_exp)

    metrics = {
        'total_reward': total_reward,
        'score': score,
        'episode_length': t + 1,
        'is_terminal': done,
        'avg_dynamics_error': np.mean([d['error_norm'] for d in all_dynamics_data]) if len(all_dynamics_data) > 0 else 0,
        'std_dynamics_error': np.std([d['error_norm'] for d in all_dynamics_data]) if len(all_dynamics_data) > 0 else 0,
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
        render_mode='rgb_array' if args.save_video else None, # no rendering for dynamics evaluation
        width=640,
        height=480
    )
else:
    env = dataset.env

# Check for observation dimension mismatch
check_env = gym.make(args.dataset)
if check_env.observation_space.shape[0] != dataset.observation_dim:
    print(f"\n[FATAL] Observation dimension mismatch detected!")
    print(f"  - Dataset from command line ('{args.dataset}'): {check_env.observation_space.shape[0]}")
    print(f"  - Dataset from loaded model ('{dataset.env_name}'): {dataset.observation_dim}")
    print(f"Please ensure the loaded checkpoint in your shell script matches the evaluation dataset.\n")
    exit(1)

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