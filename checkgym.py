import d4rl
import gym

# 모든 등록된 환경 출력
for env_spec in gym.envs.registry.all():
    print(env_spec.id)