import pybullet_envs
from pybullet_envs.minitaur.envs.minitaur_gym_env import MinitaurGymEnv
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
import gym
# env = MinitaurGymEnv(render=True)
env = HalfCheetahBulletEnv(render=True)
env.reset()
while True:
    a = env.action_space.sample()
    # env.configure()
    env.step(a)