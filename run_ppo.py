import gc
import csv
import datetime
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from threading import Thread
from env.gazebo_env.robot_env import *

import argparse
import numpy as np
import torch
import gym
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import SACPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from env.exhx5_walkingmodule_env import Exhx5WalkEnv
from env.env_wrapper import RandomWrapper

gc.enable()
rospy.init_node('x5_controller')
env = ROBOT()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Exhx5WalkMod-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=4096)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=30000)
    parser.add_argument('--step-per-collect', type=int, default=2048)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
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
    parser.add_argument('--resume-path', type=str,
                        default='log/Exhx5WalkMod-v0/ppo/seed_1_1219_215226-Exhx5WalkMod_v0_ppo/policy.pth')
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_sac(args=get_args()):
    gym_env = gym.make(args.task)
    args.state_shape = gym_env.observation_space.shape or gym_env.observation_space.n
    args.action_shape = gym_env.action_space.shape or gym_env.action_space.n
    args.max_action = gym_env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(gym_env.action_space.low), np.max(gym_env.action_space.high))
    # train_envs = env
    train_envs = SubprocVectorEnv(
        [lambda: Exhx5WalkEnv(render=False) for _ in range(args.training_num)]
    )
    # test_envs = env
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
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
        action_space=gym_env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    policy.eval()
    enable_walk_module.publish("walking_module")

    reward_test = 0.0

    for episode in range(1, args.test_num):

        reward_sum = 0.0
        # initial state_sequence
        robot_state.reset_state()
        env.reset()
        # env.reset()

        # reset walking parameters, random initial position
        env.reset_walking_para()
        apply_para.publish(walking_para)
        rospy.sleep(0.5)
        send_command.publish("start")

        # read initial state
        robot_state.set_robot_state()
        state_t = robot_state.pybullet_state
        # state_t = env.state_normalization(state_t)

        for steps in range(1, 500):

            action, _ = policy.actor(torch.tensor(state_t, dtype=torch.float32).reshape(1, 20))
            action = action[0].detach().cpu().numpy()[0]
            action = policy.map_action(action)
            print(action)
            # action = [0.01966546, 0.01,       0.01843299, 0.01702707, 0.20592713]
            # state_t1, _, _, _, = gym_env.step(action)
            env.take_action(action)
            reward, reward_vec, done = env.compute_rew(robot_state.reference)
            reward_sum += reward
            # update state
            state_t1 = robot_state.pybullet_state
            # state_t1 = env.state_normalization(state_t1)
            state_t = state_t1
            if done:
                send_command.publish("stop")
                rospy.sleep(1.0)
                robot_state.done = False
                env.reset_action()
                break
        print("episode", episode, "reward", reward_sum)
        reward_test += reward_sum
    print("average reward", reward_test / args.test_num)


def listener():
    print("Listener")
    call = CallBackData()
    rospy.Subscriber("/r_ank_roll_link_contact_sensor_state", ContactsState, call.callback_r_contact)
    rospy.Subscriber("/l_ank_roll_link_contact_sensor_state", ContactsState, call.callback_l_contact)
    rospy.Subscriber("/odom/body", Odometry, call.callback_odom)
    rospy.Subscriber("/imu", Imu, call.callback_imu)
    rospy.Subscriber("/exhx5/joint_states", JointState, call.callbackJointStates)


def main():
    # listen
    thread = Thread(target=listener(), )
    thread.start()
    # publish
    test_sac()


if __name__ == '__main__':
    main()
