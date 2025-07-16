import gym
import d4rl
import numpy as np

# gym 환경을 생성합니다.
env = gym.make('kitchen-partial-v0')

# D4RL로부터 데이터셋을 불러옵니다.
dataset = env.get_dataset()

# 'observations' 키를 사용하여 관측 데이터를 가져옵니다.
observations = dataset['observations']

# 각 차원(dimension)의 평균, 최소, 최대를 계산합니다.
mean_obs = np.mean(observations, axis=0)
min_obs = np.min(observations, axis=0)
max_obs = np.max(observations, axis=0)

# 결과를 출력합니다.
print("Observation Mean:")
print(mean_obs)

print("\nObservation Min:")
print(min_obs)

print("\nObservation Max:")
print(max_obs)


# 벡터들의 차원을 출력합니다.
print("\nDimension of the mean vector:")
print(mean_obs.shape)
print("\nDimension of the min vector:")
print(min_obs.shape)
print("\nDimension of the max vector:")
print(max_obs.shape)

# Franka Kitchen 환경의 전체 60차원 Observation에 대한 설명입니다.
#
# 벡터 구조: [qpos (30D), qvel (30D)]
#
# 패딩의 원인 및 위치:
# 주전자(kettle)는 위치(qpos)가 7차원이지만, 물리적 속도(qvel)는 6차원입니다.
# D4RL은 qpos/qvel의 차원을 30으로 맞추기 위해, 29차원인 qvel 벡터 전체의
# 마지막에 0을 추가(패딩)합니다.
# 그 결과, 패딩은 전체 60차원 벡터의 가장 마지막(59번 인덱스)에 위치하게 됩니다.
full_descriptions = [
    # qpos (dims 0-29)
    "qpos 0: robot:panda0_joint1 angle",
    "qpos 1: robot:panda0_joint2 angle",
    "qpos 2: robot:panda0_joint3 angle",
    "qpos 3: robot:panda0_joint4 angle",
    "qpos 4: robot:panda0_joint5 angle",
    "qpos 5: robot:panda0_joint6 angle",
    "qpos 6: robot:panda0_joint7 angle",
    "qpos 7: robot:r_gripper_finger_joint position",
    "qpos 8: robot:l_gripper_finger_joint position",
    "qpos 9: bottom right burner knob rotation",
    "qpos 10: bottom right burner opening",
    "qpos 11: bottom left burner knob rotation",
    "qpos 12: bottom left burner opening",
    "qpos 13: top right burner knob rotation",
    "qpos 14: top right burner opening",
    "qpos 15: top left burner knob rotation",
    "qpos 16: top left burner opening",
    "qpos 17: overhead light switch angle",
    "qpos 18: overhead light joint opening",
    "qpos 19: slide cabinet joint translation",
    "qpos 20: left hinge cabinet joint rotation",
    "qpos 21: right hinge cabinet joint rotation",
    "qpos 22: microwave door joint rotation",
    "qpos 23: kettle x coordinate",
    "qpos 24: kettle y coordinate",
    "qpos 25: kettle z coordinate",
    "qpos 26: kettle x quaternion",
    "qpos 27: kettle y quaternion",
    "qpos 28: kettle z quaternion",
    "qpos 29: kettle w quaternion",
    # qvel (dims 30-59)
    "qvel 0: robot:panda0_joint1 angular velocity",
    "qvel 1: robot:panda0_joint2 angular velocity",
    "qvel 2: robot:panda0_joint3 angular velocity",
    "qvel 3: robot:panda0_joint4 angular velocity",
    "qvel 4: robot:panda0_joint5 angular velocity",
    "qvel 5: robot:panda0_joint6 angular velocity",
    "qvel 6: robot:panda0_joint7 angular velocity",
    "qvel 7: robot:r_gripper_finger_joint linear velocity",
    "qvel 8: robot:l_gripper_finger_joint linear velocity",
    "qvel 9: bottom right burner knob angular velocity",
    "qvel 10: bottom right burner opening linear velocity",
    "qvel 11: bottom left burner knob angular velocity",
    "qvel 12: bottom left burner opening linear velocity",
    "qvel 13: top right burner knob angular velocity",
    "qvel 14: top right burner opening linear velocity",
    "qvel 15: top left burner knob angular velocity",
    "qvel 16: top left burner opening linear velocity",
    "qvel 17: overhead light switch angular velocity",
    "qvel 18: overhead light opening angular velocity",
    "qvel 19: slide cabinet joint linear velocity",
    "qvel 20: left hinge cabinet angular velocity",
    "qvel 21: right hinge cabinet angular velocity",
    "qvel 22: microwave door angular velocity",
    "qvel 23: kettle x linear velocity",
    "qvel 24: kettle y linear velocity",
    "qvel 25: kettle z linear velocity",
    "qvel 26: kettle x axis angular velocity",
    "qvel 27: kettle y axis angular velocity",
    "qvel 28: kettle z axis angular velocity",
    "qvel 29: padding (to make qvel 30-dim)"
]

# 불러온 데이터셋의 차원 수만큼 통계와 설명을 함께 출력합니다.
print("\n--- Dimension-wise Statistics and Descriptions ---")
num_dims = mean_obs.shape[0]
for i in range(num_dims):
    print(f"dim {i+1:2d} | mean {mean_obs[i]:.2e} | min {min_obs[i]:.2e} | max {max_obs[i]:.2e} | {full_descriptions[i]}")
