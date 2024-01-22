#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from env.exhx5_walkingmodule_env import Exhx5WalkEnv
from env.exhx5_gazebo_env import Exhx5GazeboEnv
from env.env_wrapper import RandomWrapper
from env.make_random_env import make_random_env

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Exhx5WalkMod-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=4096)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=2048)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    # ppo special
    parser.add_argument('--rew-norm', type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default='/home/zhou/exhx5_pybullet/log/Exhx5WalkMod-v0/ppo/seed_1_0106_214108-Exhx5WalkMod_v0_ppo/policy.pth')
    parser.add_argument(
        '--watch',
        default=True,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_ppo(args=get_args()):
    # env = Exhx5GazeboEnv()
    # train_envs = env
    # test_envs = env
    # args.training_num = 1
    # args.test_num = 10
    # args.state_shape = env.observation_space.shape or env.observation_space.n
    # args.action_shape = env.action_space.shape or env.action_space.n
    # args.max_action = env.action_space.high[0]
    # print("Observations shape:", args.state_shape)
    # print("Actions shape:", args.action_shape)
    # print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    p_env = Exhx5WalkEnv()
    g_env = Exhx5GazeboEnv()
    args.state_shape = p_env.observation_space.shape or p_env.observation_space.n
    args.action_shape = p_env.action_space.shape or p_env.action_space.n
    args.max_action = p_env.action_space.high[0]
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,
        device=args.device
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr
    )

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=p_env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )

    # load a previous policy
    policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer = ReplayBuffer(args.buffer_size)
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # test_collector = Collector(policy, test_envs)
    # # log
    # t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_ppo'
    # log_path = os.path.join(args.logdir, args.task, 'ppo', log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger = TensorboardLogger(writer, update_interval=100, train_interval=100)

    # def save_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    #
    # if not args.watch:
    #     # trainer
    #     result = onpolicy_trainer(
    #         policy,
    #         train_collector,
    #         test_collector,
    #         args.epoch,
    #         args.step_per_epoch,
    #         args.repeat_per_collect,
    #         args.test_num,
    #         args.batch_size,
    #         step_per_collect=args.step_per_collect,
    #         save_fn=save_fn,
    #         logger=logger,
    #         test_in_train=False
    #     )
    #     pprint.pprint(result)

    g_env.reset()
    p_env.reset()
    length = 200
    states = np.ones((length, 20)) * np.array([0, 0, 0, 0, 0, 0, 0.25, 0, 0.75,
                                               0, -0.05, -0.5, 1.25, 0.75, -0.1, 0.05, 0.5, -1.25, -0.25, 0.05])
    step = 0
    state_id = 10
    states[:, state_id] = np.linspace(-0.2, 0.1, length)
    steps = np.arange(0, length)
    actions, _ = policy.actor(torch.tensor(states, dtype=torch.float32).reshape(-1, 20))
    actions = actions[0].detach().cpu().numpy()
    actions = policy.map_action(actions)
    plt.figure(1)
    for i in range(0, 5):
        plt.subplot(2, 3, i+1)
        plt.plot(steps, actions[:, i], label=f'actions{i}')
    plt.legend()
    # plt.figure(2)
    # for i in range(0, 20):
    #     plt.subplot(4, 5, i+1)
    #     plt.plot(steps, states_g[:, i], label=f'state_gazebo{i}')
    #     plt.plot(steps, states_p[:, i], label=f'state_pybullet{i}')
    # plt.legend()
    plt.show()
    # [-1,1],[-1,1],[-1,1],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,1],[-2.5,2.5],[0,1.5],[-2.5,2.5]
    # [-0.2,0.1],[-0.75,-0.25],[1,1.5],[0.5,1.0],[-0.2,0.05],[-0.1,0.2],[0.25,0.75],[-1.5,-1],[-1,0.5],[-0.1,0.2]


if __name__ == '__main__':
    test_ppo()