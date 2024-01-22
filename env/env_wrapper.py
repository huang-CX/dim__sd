import gym
import numpy as np
from gym import spaces
import env
from env.exhx5_walkingmodule_env import Exhx5WalkEnv


def invert_task_id(id):
    if id is None:
        id = 99
    floor = id // 7
    mod = id % 7
    x_shift = mod / 10.
    if 1 <= id <= 20:
        interval = .7
        return np.array([interval, x_shift, floor * 0.1])
    elif 21 <= id <= 27:
        interval = .8
        return np.array([interval, x_shift, .0])
    elif 28 <= id <= 34:
        interval = .9
        return np.array([interval, x_shift, .0])
    else:
        return np.array([.7, .0, .0])


class MetaWrapper(gym.Wrapper):
    """
    Wrap the env to RNN-style env
    Concatenate the [s_t, a_{t-1}, r_{t-1}] as the new s
    """

    def __init__(self, unwrapped_env, test=False):
        super().__init__(unwrapped_env)
        low = np.concatenate((self.env.observation_space.low, self.env.action_space.low, (-np.inf,)), axis=-1)
        high = np.concatenate((self.env.observation_space.high, self.env.action_space.high, (np.inf,)), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        # self.random = random
        self.test = test
        # if self.random is not None:
        #     self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        #     self.env.change_dynamic()

    def reset(self):
        init_observation = self.env.reset()
        init_action = np.random.uniform(self.action_space.low, self.action_space.high)
        init_reward = np.array([0.], dtype=np.float)
        observation = np.concatenate((init_observation, init_action, init_reward))
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, action, np.asarray([reward], dtype=np.float)), axis=-1)
        return observation, reward, done, info


class RandomNoContext(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            self.env.change_dynamic()
        if self.test:
            self.env.change_adapt_dynamic()
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.random and not self.test:
            self.env.change_dynamic()
        return observation


class RandomNoContextFrictionWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()
        # self.xyz_shift = invert_task_id(self.random)
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        if hasattr(self.env, 'get_dynamic'):
            self.dynamic_para = self.env.get_dynamic()
        else:
            self.dynamic_para = [1.8238, 0.6, 1, 0.8]

    def reset(self):
        observation = self.env.reset()
        friction = 0.1 + self.seed * 0.01
        self.env.change_adapt_friction(friction=friction)
        return observation


class RandomNoContextVelocityWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        period_time = 0.7 + self.seed * 0.01
        self.env.change_adapt_velocity(period_time)
        return observation


class RandomNoContextDynamicWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 40000 + self.seed * 2000
            damping = 4 + self.seed
        else:
            stiffness = 120000 + (self.seed - 5) * 2000
            damping = 12 + self.seed - 5
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        return observation


class RandomNoContextDampingWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 80000
            damping = 4 + self.seed
        else:
            stiffness = 80000
            damping = 12 + self.seed - 5
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        return observation


class RandomNoContextStiffnessWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 40000 + self.seed * 2000
            damping = 10
        else:
            stiffness = 120000 + (self.seed - 5) * 2000
            damping = 10
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        return observation


class RandomNoContextCoMWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        return observation


class RandomNoContextIntervalWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            # self.env.change_dynamic()

    def reset(self):
        observation = self.env.reset()
        return observation


class RandomWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
            self.env.change_dynamic()
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.random and not self.test:
            self.env.change_dynamic()
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomDynamicWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 40000 + self.seed * 2000
            damping = 4 + self.seed
        else:
            stiffness = 120000 + (self.seed - 5) * 2000
            damping = 12 + self.seed - 5
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomDampingWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 80000
            damping = 4 + self.seed
        else:
            stiffness = 80000
            damping = 12 + self.seed - 5
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomStiffnessWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.seed < 5:
            stiffness = 40000 + self.seed * 2000
            damping = 10
        else:
            stiffness = 120000 + (self.seed - 5) * 2000
            damping = 10
        self.env.change_adapt_dynamic(stiffness=stiffness, damping=damping)
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomFrictionWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        if self.test:
            friction = 0.1 + self.seed * 0.01
        else:
            friction = np.random.uniform(0.2, 0.8)
        self.env.change_adapt_friction(friction=friction)
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomCoMWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomIntervalWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info


class RandomVelocityWrapper(gym.Wrapper):
    """
    Wrap the env to random tasks
    """

    def __init__(self, unwrapped_env, random: int = None, test=False, seed=0):
        super().__init__(unwrapped_env)
        self.random = random
        self.seed = seed
        self.test = test
        self.env.seed(seed)
        low = np.concatenate((self.env.observation_space.low, np.zeros((8, ))), axis=-1)
        high = np.concatenate((self.env.observation_space.high, np.ones((8, ))), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high)
        if self.random is not None:
            self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
        self.xyz_shift = invert_task_id(self.random)
        # mass, friction, damping, stiffness and normalization
        self.dynamic_para = self.env.get_dynamic()

    def reset(self):
        observation = self.env.reset()
        period_time = 0.7 + self.seed * 0.01
        self.env.change_adapt_velocity(period_time)
        self.dynamic_para = self.env.get_dynamic()
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate((observation, self.dynamic_para, self.xyz_shift), axis=-1)
        return observation, reward, done, info
# class UniversalWrapper(gym.Wrapper):
#     """
#     Wrap the env to RNN-style env
#     Concatenate the [s_t, a_{t-1}, r_{t-1}] as the new s
#     """
#
#     def __init__(self, unwrapped_env, random: int = None, test=False):
#         super().__init__(unwrapped_env)
#         low = np.concatenate((self.env.observation_space.low, self.env.action_space.low,
#                               (-np.inf,), np.zeros((7, ))), axis=-1)
#         high = np.concatenate((self.env.observation_space.high, self.env.action_space.high,
#                                (np.inf,), np.zeros((7, ))), axis=-1)
#         self.observation_space = spaces.Box(low=low, high=high)
#         self.random = random
#         self.test = test
#         if self.random is not None:
#             self.env.urdf_path = "env/robot_urdf/exhx5_w_b_16_" + str(self.random) + ".urdf"
#             self.env.change_dynamic()
#         self.xyz_shift = invert_task_id(self.random)
#         self.base, self.left_leg, self.right_leg = self.env.get_dynamic()
#         # mass, friction, damping, stiffness and normalization
#         self.dynamic_para = self.base[0], self.left_leg[1], self.left_leg[8] / 10, self.left_leg[9] / 1e6
#
#     def reset(self):
#         init_observation = self.env.reset()
#         init_action = np.random.uniform(self.action_space.low, self.action_space.high)
#         init_reward = np.array([0.], dtype=np.float)
#         observation = np.concatenate((init_observation, init_action, init_reward,
#                                       self.dynamic_para, self.xyz_shift), axis=-1)
#         return observation
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         observation = np.concatenate((observation, action, np.asarray([reward], dtype=np.float),
#                                       self.dynamic_para, self.xyz_shift), axis=-1)
#         return observation, reward, done, info


if __name__ == "__main__":
    import os
    os.chdir("../")
    walk_env = Exhx5WalkEnv(render=True)
    meta_env = RandomWrapper(MetaWrapper(walk_env), random=1)
    meta_env.seed(0)
    meta_env.reset()
    while True:
        act = np.array([0.03, 0.02, 0.01, 0.015, 0.2])
        meta_env.step(act)
