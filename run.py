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
from tianshou.policy import SACPolicy
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str,
                        default='/home/zhou/exhx5_pybullet/log/Exhx5WalkMod-v0/sac/seed_0_1220_192339-Exhx5WalkMod_v0_sac/policy.pth')
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_sac(args=get_args()):
    gym_env = RandomWrapper(Exhx5WalkEnv())
    train_envs = gym_env
    test_envs = gym_env
    args.state_shape = gym_env.observation_space.shape or gym_env.observation_space.n
    args.action_shape = gym_env.action_space.shape or gym_env.action_space.n
    args.max_action = gym_env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(gym_env.action_space.low), np.max(gym_env.action_space.high))
    # train_envs = gym.make(args.task)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
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
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(gym_env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=gym_env.action_space
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
        # state_t = gym_env.reset()
        # print(gym_env.reset(), state_t)

        for steps in range(1, 500):

            action, _ = policy.actor(torch.tensor(state_t, dtype=torch.float32).reshape(1, 20))
            action = action[0].detach().cpu().numpy()[0]
            action = policy.map_action(action)
            # print(state_t)
            # print(time.time())
            # action = [0.01966546, 0.01,       0.01843299, 0.01702707, 0.20592713]
            # state_t1, reward, done, _, = gym_env.step(action)
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
