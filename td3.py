#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, Recurrent
from tianshou.utils.net.continuous import Actor, Critic

from env.exhx5_walkingmodule_env import Exhx5WalkEnv
from env.env_wrapper import RandomWrapper, MetaWrapper
from env.make_random_env import make_random_env
from model.net import MQLRecurrentCritic, MQLRecurrentActorProb, MQLRecurrentActor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Exhx5WalkMod-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=30)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--recurrent', type=int, default=True)
    parser.add_argument('--context-hidden-size', type=int, default=64)
    parser.add_argument('--context-size', type=int, default=8)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--history-len', type=int, default=8)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_td3(args=get_args()):
    if args.recurrent:
        env_wrapper = MetaWrapper
    else:
        env_wrapper = RandomWrapper
    env = env_wrapper(Exhx5WalkEnv(render=args.watch))
    if args.watch:
        train_envs = env
        test_envs = env
    else:
        train_envs, test_envs = make_random_env(task=Exhx5WalkEnv, wrapper=env_wrapper,
                                                train_task_num=args.training_num,
                                                test_task_num=args.test_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    if args.recurrent:
        actor = MQLRecurrentActor(layer_num=args.layer_num, state_shape=args.state_shape,
                                  action_shape=args.action_shape, context_hidden_size=args.context_hidden_size,
                                  context_size=args.context_size,
                                  hidden_sizes=args.hidden_sizes,
                                  max_action=args.max_action,
                                  device=args.device).to(args.device)
        # net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        # actor = Actor(
        #     net_a, args.action_shape, max_action=args.max_action, device=args.device, hidden_sizes=args.hidden_sizes
        # ).to(args.device)
        critic1 = MQLRecurrentCritic(layer_num=args.layer_num, state_shape=args.state_shape,
                                     action_shape=args.action_shape,
                                     device=args.device, context_hidden_size=args.context_hidden_size,
                                     context_size=args.context_size,
                                     hidden_sizes=args.hidden_sizes).to(args.device)
        critic2 = MQLRecurrentCritic(layer_num=args.layer_num, state_shape=args.state_shape,
                                     action_shape=args.action_shape,
                                     device=args.device, context_hidden_size=args.context_hidden_size,
                                     context_size=args.context_size,
                                     hidden_sizes=args.hidden_sizes).to(args.device)
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs), stack_num=args.history_len)
        else:
            buffer = ReplayBuffer(args.buffer_size, stack_num=args.history_len)
    else:
        net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        actor = Actor(
            net_a, args.action_shape, max_action=args.max_action, device=args.device
        ).to(args.device)

        net_c1 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        )
        net_c2 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        )
        critic1 = Critic(net_c1, device=args.device).to(args.device)
        critic2 = Critic(net_c2, device=args.device).to(args.device)
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(args.buffer_size)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_td3'
    log_path = os.path.join(args.logdir, args.task, 'td3', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_fn=save_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_td3()
