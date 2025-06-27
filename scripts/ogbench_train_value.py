import sys
import os

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
import torch
import ogbench

# IQL 및 diffuser 모델 임포트
try:
    from iql_pytorch.critic import Critic, ValueCritic
except ImportError as e:
    print(f"Error: Failed to import 'iql_pytorch.critic'. Make sure the iql_pytorch library is in your PYTHONPATH.")
    print(e)
    sys.exit(1)

from diffuser.models.values import ValueFunction
from diffuser.datasets.ogbench import OGBenchValueDataset
from diffuser.models.diffusion import AdvantageValueDiffusion # 새로 만든 클래스 임포트

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'scene-play-singletask-task3-v0'
    config: str = 'config.locomotion'
    # Value 학습을 위한 파라미터들
    horizon: int = 64
    batch_size: int = 32
    learning_rate: float = 2e-4
    normalizer: str = 'LimitsNormalizer'
    normed: bool = True
    use_padding: bool = True
    max_path_length: int = 1000
    q_path: str = '/home/junseolee/intern/diffuser_ogbench/iql_pytorch/runs/ogbench_runs/task3/test-task3-06-14-08-38-bs256-s2-t3.0-e0.7/model/critic_target_s1000000.pth'
    v_path: str = '/home/junseolee/intern/diffuser_ogbench/iql_pytorch/runs/ogbench_runs/task3/test-task3-06-14-08-38-bs256-s2-t3.0-e0.7/model/value_s1000000.pth'

args = Parser().parse_args('values')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# IQL 네트워크 로드 (dataset 생성에 필요)
env, _, _ = ogbench.make_env_and_datasets(args.dataset, render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

q_network = Critic(state_dim, action_dim).to(args.device)
v_network = ValueCritic(state_dim, 512, 5).to(args.device)
q_network.load_state_dict(torch.load(args.q_path, map_location=args.device))
v_network.load_state_dict(torch.load(args.v_path, map_location=args.device))
q_network.eval()
v_network.eval()

# 데이터셋 설정
dataset_config = utils.Config(
    OGBenchValueDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env_name=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    max_n_episodes=1000,
    termination_penalty=0, # d4rl과 달리 0으로 설정
    discount=args.discount,
    normed=args.normed,
    # OGBenchValueDataset에 필요한 추가 인자
    q_network=q_network,
    v_network=v_network,
    device=args.device
)

dataset = dataset_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# 모델 설정 (ValueFunction 사용)
model_config = utils.Config(
    ValueFunction,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

# Diffusion 설정 (AdvantageValueDiffusion 사용)
diffusion_config = utils.Config(
    AdvantageValueDiffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type, # MSE loss 등을 사용
    device=args.device,
)

# Trainer 설정
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
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, None) # renderer는 필요 없음

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')
print(f"Initial loss: {loss.item()}")

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

print("Training finished.") 