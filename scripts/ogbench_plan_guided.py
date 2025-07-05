import sys
import os
import numpy as np
import torch
import imageio
import subprocess
import time
import pickle
import wandb
from diffuser.sampling.functions import n_step_guided_p_sample

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
from diffuser.models.values import ValueFunction  # ValueFunction 모델 임포트
from diffuser.guides.core import ValueGuide      # ValueGuide 임포트
from diffuser.sampling.policies import GuidedPolicy
import ogbench

# Xvfb를 사용하여 가상 디스플레이 설정
if 'DISPLAY' not in os.environ:
    subprocess.run(['pkill', 'Xvfb'], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
    os.environ['DISPLAY'] = ':100.0'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['EGL_DEVICE_ID'] = '0' # 사용 가능한 GPU ID로 설정
    time.sleep(2)  # Xvfb가 시작될 때까지 대기

class Parser(utils.Parser):
    task_id: int = 2
    dataset: str = 'scene-play-singletask-task2-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    n_diffusion_steps: int = 100
    batch_size: int = 32
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000
    # sampling specific parameters
    n_samples: int = 32
    guide_scale: float = 1.0
    verbose: bool = True
    # 추가된 설정
    n_runs: int = 1000
    save_video: bool = False
    use_wandb: bool = False

args = Parser().parse_args('plan')
args.dataset = f'scene-play-singletask-task{args.task_id}-v0'
set_model_mode(args.prefix)

# 'scripts' 디렉토리를 제외하고 'logs'를 바로 참조하도록 수정
args.loadbase = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

if args.use_wandb:
    wandb.init(
        project="diffuser_ogbench_planning_eval_adv",
        name=f"task{args.task_id}-{os.path.basename(args.diffusion_loadpath)}-{args.diffusion_epoch}-adv_guide",
        config=vars(args)
    )

# 환경 및 데이터셋 초기화
env, _, _ = ogbench.make_env_and_datasets(
    args.dataset,
    render_mode='rgb_array' if args.save_video else 'none', # 비디오 저장 안 할 시 렌더링 최소화
    width=640,
    height=480
)

# Diffusion 모델 로드
print(f"Loading diffusion model from: {args.diffusion_loadpath} at epoch {args.diffusion_epoch}", flush=True)
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps
)
diffusion = diffusion_experiment.ema
renderer = diffusion_experiment.renderer # 렌더러는 diffusion model에서 가져옴
dataset = diffusion_experiment.dataset # 데이터셋(정규화 정보 포함)도 여기서 가져옴

# Advantage 기반 ValueFunction 모델 로드
value_loadpath = os.path.join(args.loadbase, args.dataset, args.value_loadpath)
value_model_path = os.path.join(value_loadpath, f'value_model_step_{args.value_epoch}.pth')
print(f"Loading advantage value model from: {value_model_path}", flush=True)

# ValueFunction 모델 초기화
value_function = ValueFunction(
    horizon=args.horizon,
    transition_dim=dataset.observation_dim + dataset.action_dim,
    cond_dim=dataset.observation_dim,
    dim_mults=args.dim_mults,
    device=args.device
).to(args.device)

# 저장된 state_dict 로드
state_dict = torch.load(value_model_path, map_location=args.device)
value_function.load_state_dict(state_dict['model'])
value_function.eval()


# ValueGuide 초기화 (diffuser.guides.core.ValueGuide 사용)
guide = ValueGuide(value_function, normalizer=dataset.normalizer, verbose=False)

# Guided policy 초기화
policy_config = utils.Config(
    GuidedPolicy, # diffuser.sampling.policies.GuidedPolicy
    guide=guide,
    scale=args.guide_scale, # args.scale 대신 args.guide_scale 사용
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
    sample_fn=n_step_guided_p_sample,
)

policy = policy_config()

# 평가 루프
success_count = 0
for run_id in range(args.n_runs):
    # 환경 초기화
    obs, info = env.reset(
        options=dict(
            task_id=args.task_id,
            render_goal=True,
        )
    )
    
    goal = info['goal']
    
    save_current_video = args.save_video and (run_id + 1) % 50 == 0
    if save_current_video:
        # 비디오 저장 설정
        video_path = os.path.join(args.savepath, f'task_{args.task_id}_run_{run_id + 1}_adv_guided.mp4')
        video_writer = imageio.get_writer(video_path, fps=30)
    
    max_steps = 1000  # 최대 step 제한
    done = False
    step = 0
    while not done and step < max_steps:
        # 현재 상태와 최종 목표를 조건으로 사용
        conditions = {
            0: obs,
            diffusion.horizon - 1: goal,
        }
        
        # Value guided policy로 액션 생성
        action, samples = policy(
            conditions,
            batch_size=args.n_samples,
            verbose=args.verbose
        )
        
        # 환경에서 액션 실행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if step % 100 == 0:
            print(f"\n{'='*50}")
            print(f"[Task {args.task_id}, Run {run_id+1}/{args.n_runs}] Step: {step}")
            print(f"{'='*50}\n")
        
        if save_current_video:
            # 프레임 저장
            frame = env.render()
            video_writer.append_data(frame)
        step += 1
    
    if save_current_video:
        video_writer.close()
        print(f"Saved video for run {run_id + 1} to {video_path}")

    if info['success']:
        success_count += 1
    
    current_success_rate = (success_count / (run_id + 1)) * 100
    log_message = f"Task {args.task_id}, Run {run_id+1}/{args.n_runs} | Success: {info['success']} | Current success rate: {current_success_rate:.2f}%"
    print(log_message)
    if args.use_wandb:
        wandb.log({
            "run": run_id + 1,
            "success_rate": current_success_rate,
            "success_count": success_count,
        })

final_success_rate = (success_count / args.n_runs) * 100 if args.n_runs > 0 else 0
print(f"\n{'='*50}")
print(f"Final Evaluation Results for Task {args.task_id}")
print(f"Total runs: {args.n_runs}")
print(f"Successful runs: {success_count}")
print(f"Success rate: {final_success_rate:.2f}%")
print(f"{'='*50}")

if args.use_wandb:
    wandb.log({
        "final_success_rate": final_success_rate,
        "total_successful_runs": success_count
    })
    wandb.finish()

# Xvfb 프로세스 종료
subprocess.run(['pkill', 'Xvfb'], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) 