#!/usr/bin/env python3
import gym
import numpy as np
import env

from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from env.env_wrapper import MetaWrapper, RandomIntervalWrapper, RandomWrapper, RandomDynamicWrapper, RandomFrictionWrapper, \
    RandomVelocityWrapper, RandomNoContext, RandomCoMWrapper, RandomNoContextCoMWrapper, RandomNoContextIntervalWrapper, \
    RandomIntervalWrapper
from env.exhx5_walkingmodule_env import Exhx5WalkEnv


def make_random_env(task=Exhx5WalkEnv, wrapper: gym.wrappers = RandomWrapper, train_task_num=3, test_task_num=2, seed=0):
    np.random.seed(seed)
    train_task_list = np.arange(train_task_num)

    print(f"train task ids {train_task_list}")
    # print(f"test task ids {test_task_list}")
    train_envs = SubprocVectorEnv(
        [lambda n=index, x=i: wrapper(MetaWrapper(task()), random=x, seed=seed + n)
         for index, i in enumerate(train_task_list)]
    )
    # test_envs = SubprocVectorEnv(
    #     [lambda n=index, x=i: RandomWrapper(MetaWrapper(task()), random=x, test=True, seed=seed + n,)
    #      for index, i in enumerate(test_task_list)]
    # )
    test_envs = train_envs
    return train_envs, test_envs


def make_meta_env(task=Exhx5WalkEnv, wrapper: gym.wrappers = RandomWrapper, train_task_num=3, test_task_num=2, seed=0):
    np.random.seed(seed)
    task_list = np.arange(35)  # 35 for the total urdfs
    train_task_list = np.random.choice(task_list, train_task_num, replace=True)
    print(f"train task ids {train_task_list}")
    # print(f"test task ids {test_task_list}")
    train_envs = SubprocVectorEnv(
        [lambda n=index, x=i: wrapper(MetaWrapper(task()), random=x, seed=seed + n)
         for index, i in enumerate(train_task_list)]
    )
    test_envs = train_envs
    return train_envs, test_envs


def make_random_adapt_env(task=Exhx5WalkEnv, wrapper: gym.wrappers = RandomFrictionWrapper, adapt_task_num=3, seed=0):
    np.random.seed(seed)
    if wrapper is RandomCoMWrapper or wrapper is RandomNoContextCoMWrapper:
        task_list = [0, 21, 28]
        adapt_task_list = np.random.choice(task_list, 10)
        adapt_envs = SubprocVectorEnv(
            [lambda n=index, x=i: wrapper(MetaWrapper(task()), random=x, test=True, seed=n, )
             for index, i in enumerate(adapt_task_list)]
        )
    elif wrapper is RandomIntervalWrapper or wrapper is RandomNoContextIntervalWrapper:
        task_list = np.arange(28, 35)
        adapt_task_list = np.random.choice(task_list, 10)
        adapt_envs = SubprocVectorEnv(
            [lambda n=index, x=i: wrapper(MetaWrapper(task()), random=x, test=True, seed=n, )
             for index, i in enumerate(adapt_task_list)]
        )
    else:
        adapt_task_list = 10 * np.ones(shape=(adapt_task_num, ))
        adapt_envs = SubprocVectorEnv(
            [lambda n=index: wrapper(MetaWrapper(task()), random=10, test=True, seed=n,)
             for index in range(adapt_task_num)]
        )
    print(f"adapt task ids {adapt_task_list}")
    adapt_test_envs = adapt_envs
    return adapt_envs, adapt_test_envs


if __name__ == '__main__':
    train_envs, test_envs = make_random_env()
    train_envs.reset()
    while True:
        action_bias = [0.03, 0.025, 0.015, 0.01, 0.2]
        action = np.array([action_bias, action_bias, action_bias])
        obs, _, _, _ = train_envs.step(action)
