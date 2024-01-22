import pybullet
import gym
import torch
import pybullet_data
import numpy as np
import math
import time
import gc
import csv
import datetime
import rospy
import rospy
import time

from abc import ABC
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from threading import Thread
from gym import spaces

from std_srvs.srv import Empty
from std_msgs.msg import Float64, String
from gazebo_msgs.srv import SetModelConfiguration
from op3_walking_module_msgs.msg import WalkingParam
import matplotlib.pyplot as plt

# x_move_amplitude, z_move_amplitude, x_swap_amplitude, z_swap_amplitude, dsp_ratio
ACTION_BIAS = np.array([0.03, 0.025, 0.015, 0.01, 0.2])
ACTION_SCALE = np.array([0.02, 0.015, 0.01, 0.01, 0.1])
SEQUENCE_LEN = 3
BATCH_SIZE = 128
STATE_DIM = 23
ACTION_DIM = 5
REWARD_DIM = 5
PI = 3.1415926
POSITION = (1.0, 0.0, 0.0)
JOINT_LOW = -3.14
JOINT_HIGH = 3.14
USE_CUDA = True
seed = 56
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODE_STEP = 2000


class RobotState(object):
    def __init__(self):
        # state.position
        self.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.reference = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.velocity
        self.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.effort
        self.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.robot_orientation
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.last_orientation = [0.0, 0.0, 0.0, 1.0]
        self.euler = [0.0, 0.0, 0.0]
        # state.imu
        self.imu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.robot_position
        self.body_height = 0.213649069412
        self.body_x = 0.0
        self.body_y = 0.0
        self.last_body_height = 0.213649069412
        self.last_body_x = 0.0
        self.last_body_y = 0.0
        # state.contact
        self.r_contact = 1
        self.l_contact = 1
        self.last_r_contact = 1
        self.last_l_contact = 1
        self.l_contact_force = [0.0, 0.0, 0.0]
        self.r_contact_force = [0.0, 0.0, 0.0]
        # foot position
        self.l_position = [0.0, 0.0, 0.0]
        self.r_position = [0.0, 0.0, 0.0]
        # gait
        self.gait_orientation = [0.0, 0.0, 0.0, 1.0]
        self.gait_step = 0
        self.gait_start_position = [0.0, 0.0]
        self.support_mode = 1
        self.last_support_mode = 1
        self.double_support_time = 0
        self.r_foot_support_time = 0
        self.l_foot_support_time = 0
        self.without_contact_time = 0

        self.position_contact = [self.body_x, self.body_y, self.body_height, self.l_contact, self.r_contact]
        # len=35, position=[0:10], imu=[10:16], orientation=[16:20], euler=[20:23]
        self.state = self.position + self.imu + self.orientation + self.euler
        self.full_state = self.position + self.imu + self.orientation + self.euler + self.position_contact + \
                          self.l_contact_force + self.r_contact_force + self.l_position + self.r_position
        self.pybullet_state = [self.imu[0], self.imu[1], self.imu[2]] + self.orientation + self.euler + \
                              [self.position[3], self.position[2], self.position[4], self.position[0], self.position[1],
                               self.position[8], self.position[7], self.position[9], self.position[5],
                               self.position[6]]

        # reward
        self.latest_reward = 0.0
        self.best_reward = -10000.0
        self.worst_reward = 1000
        self.episode = 0
        self.avg_reward = 0.0

        # other
        self.last_time = 0.0
        self.fall = 0
        self.done = False
        self.count_of_motionless = 0

    def set_robot_state(self):
        # set state
        # t = time.time()
        # self.position_contact = [self.body_x, self.body_y, self.body_height, self.l_contact, self.r_contact]
        # # len=43, position=[0:12], velocity=[12: 24], effort=[24:36], orientation=[36:40],
        # # len=28, position=[0:12], orientation=[12:16], contact=[17:19], euler = [19:22], imu=[22:28]
        # self.state = self.position + self.imu + self.orientation + self.euler
        # self.full_state = self.position + self.imu + self.orientation + self.euler + self.position_contact + \
        #                   self.l_contact_force + self.r_contact_force + self.l_position + self.r_position
        # self.pybullet_state = [self.body_x, self.body_y, self.body_height] + self.orientation + \
        #                       [(self.body_x - self.last_body_x) / (t - self.last_time),
        #                        (self.body_y - self.last_body_y) / (t - self.last_time),
        #                        (self.body_height - self.last_body_height) / (t - self.last_time)] + \
        #                       [self.position[2], self.position[3], self.position[4], self.position[0],
        #                       self.position[1],
        #                        self.position[7], self.position[8], self.position[9], self.position[5],
        #                        self.position[6]]
        self.pybullet_state = [self.imu[0], self.imu[1], self.imu[2]] + self.orientation + self.euler + \
                              [self.position[3], self.position[2], self.position[4], self.position[0], self.position[1],
                               self.position[8], self.position[7], self.position[9], self.position[5],
                               self.position[6]]

    def reset_state(self):
        # state.position
        self.position = [0.7150, -0.0635, -0.6169, -0.0635, 1.3319, -0.7150, 0.0635, 0.6169, 0.0635, -1.3319]
        self.reference = [0.7150, -0.0635, -0.6169, -0.0635, 1.3319, -0.7150, 0.0635, 0.6169, 0.0635, -1.3319]
        # state.velocity
        self.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.effort
        self.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.robot_orientation
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.last_orientation = [0.0, 0.0, 0.0, 1.0]
        self.euler = [0.0, 0.0, 0.0]
        # state.imu
        self.imu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # state.robot_position
        self.body_height = 0.213649069412
        self.body_x = 0.0
        self.body_y = 0.0
        self.last_body_height = 0.213649069412
        self.last_body_x = 0.0
        self.last_body_y = 0.0
        # state.contact
        self.r_contact = 1
        self.l_contact = 1
        self.last_r_contact = 1
        self.last_l_contact = 1
        self.l_contact_force = [0.0, 0.0, 0.0]
        self.r_contact_force = [0.0, 0.0, 0.0]
        # foot position
        self.l_position = [0.0, 0.0, 0.0]
        self.r_position = [0.0, 0.0, 0.0]
        # gait
        self.gait_orientation = [0.0, 0.0, 0.0, 1.0]
        self.gait_step = 0
        self.gait_start_position = [0.0, 0.0]
        self.support_mode = 1
        self.last_support_mode = 1
        self.double_support_time = 0
        self.r_foot_support_time = 0
        self.l_foot_support_time = 0
        self.without_contact_time = 0


robot_state = RobotState()
walking_para = WalkingParam()
pub_l_ank_pitch = rospy.Publisher('/exhx5/l_ank_pitch_position/command', Float64, queue_size=10)
pub_l_ank_roll = rospy.Publisher('/exhx5/l_ank_roll_position/command', Float64, queue_size=10)
pub_l_hip_pitch = rospy.Publisher('/exhx5/l_hip_pitch_position/command', Float64, queue_size=10)
pub_l_hip_roll = rospy.Publisher('/exhx5/l_hip_roll_position/command', Float64, queue_size=10)
pub_l_hip_yaw = rospy.Publisher('/exhx5/l_hip_yaw_position/command', Float64, queue_size=10)
pub_l_knee = rospy.Publisher('/exhx5/l_knee_position/command', Float64, queue_size=10)

pub_r_ank_pitch = rospy.Publisher('/exhx5/r_ank_pitch_position/command', Float64, queue_size=10)
pub_r_ank_roll = rospy.Publisher('/exhx5/r_ank_roll_position/command', Float64, queue_size=10)
pub_r_hip_pitch = rospy.Publisher('/exhx5/r_hip_pitch_position/command', Float64, queue_size=10)
pub_r_hip_roll = rospy.Publisher('/exhx5/r_hip_roll_position/command', Float64, queue_size=10)
pub_r_hip_yaw = rospy.Publisher('/exhx5/r_hip_yaw_position/command', Float64, queue_size=10)
pub_r_knee = rospy.Publisher('/exhx5/r_knee_position/command', Float64, queue_size=10)

reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

enable_walk_module = rospy.Publisher('/robotis/enable_ctrl_module', String, queue_size=10)
apply_para = rospy.Publisher('/robotis/walking/set_params', WalkingParam, queue_size=10)
send_command = rospy.Publisher('/robotis/walking/command', String, queue_size=10)


class Exhx5GazeboEnv(gym.Env, ABC):
    def __init__(self):
        # robot state
        rospy.init_node('x5_controller')
        self.rate = rospy.Rate(125)
        self.last_t = rospy.Time.now()
        self._thread = Thread(target=listener(), )
        self._thread.start()
        time.sleep(1)
        enable_walk_module.publish("walking_module")
        time.sleep(1)
        low = np.array([-np.inf] * 10 + [JOINT_LOW] * 10)
        high = np.array([np.inf] * 10 + [JOINT_HIGH] * 10)
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=ACTION_BIAS - ACTION_SCALE, high=ACTION_BIAS + ACTION_SCALE)
        self.episode_step = 0

    def step(self, action):
        _, state_current_info = self.__get_observation()
        self.__apply_action(action)
        state_next, state_next_info = self.__get_observation()
        base_current_pos = state_current_info['base_position']
        base_next_pos = state_next_info['base_position']
        euler_current = state_current_info['base_euler']
        euler_next = state_next_info['base_euler']
        done = False
        fall_reward = 0.0

        # step_reward
        step_vector = [(base_next_pos[0] - base_current_pos[0]), (base_next_pos[1] - base_current_pos[1])]
        # step_vector = [(robot_state.body_x - robot_state.last_body_x)]
        step_vector = np.asarray(step_vector)
        step_len_x = np.linalg.norm(step_vector[0])
        step_len = np.linalg.norm(step_vector)

        if (base_next_pos[0] - base_current_pos[0]) > 0:
            step_reward = min(step_len_x * 100, 3)
        else:
            step_reward = -min(abs(step_len_x * 100), 3)

        # height reward
        robot_height = base_next_pos[2]
        if robot_height > 0.18:
            height_reward = 1.0
        else:
            height_reward = 1.0 - 100 * abs(0.18 - robot_height)

        height_reward *= 0.2

        # orientation reward

        if abs(euler_next[0]) < 5. * PI / 180.:
            orientation_reward_r = 0.5
        else:
            orientation_reward_r = 0.8 - min(abs(euler_next[0]) / 10, 1.5)

        if abs(euler_next[1]) < 5. * PI / 180.:
            orientation_reward_p = 0.5
        else:
            orientation_reward_p = 0.8 - min(abs(euler_next[1]) / 10, 1.5)

        if abs(euler_next[2]) < 5. * PI / 180.:
            orientation_reward_y = 0.5
        else:
            orientation_reward_y = 0.8 - min(abs(euler_next[2]) / 10, 1.5)
        orientation_reward = orientation_reward_r + orientation_reward_p + orientation_reward_y
        orientation_reward = 0.5 * orientation_reward
        # effort reward
        effort_reward = 3.0 - 0.25 * np.sum(abs(np.asarray(state_next_info["effort"])))
        effort_reward = np.clip(effort_reward, -0.5, 0.5)

        # fall reward
        if robot_height <= 0.145 or robot_height >= 2.05:
            fall_reward = -50
            done = True
            # print("height irregular", robot_height)
        # if abs(euler_next[0]) >= 20. * PI / 180. or abs(euler_next[1]) >= 20. * PI / 180. or abs(
        #         euler_next[2]) >= 30. * PI / 180.:
        #     fall_reward = -50
            # done = True
            # print("tilt")
        # print(f"step:{step_reward}, fall:{fall_reward}, height:{height_reward}, effort:{effort_reward}")
        if self.episode_step >= MAX_EPISODE_STEP:
            done = True
        reward = step_reward + effort_reward + height_reward + fall_reward + orientation_reward
        step_info = {
            'step_reward': step_reward,
            'effort_reward': effort_reward,
            'fall_reward': fall_reward,
            'orientation_reward': orientation_reward,
            'height_reward': height_reward
        }
        self.episode_step += 1
        return state_next, reward, done, step_info

    def reset(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pause()
        except (rospy.ServiceException) as e:
            print("rospause failed!'")
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_simulation()
        except(rospy.ServiceException) as e:
            print("reset_world failed!")
        rospy.wait_for_service('gazebo/set_model_configuration')
        try:
            reset_joints("exhx5", "robot_description", ["l_ank_pitch", "l_ank_roll", "l_hip_pitch", "l_hip_roll",
                                                        "l_knee", "r_ank_pitch", "r_ank_roll", "r_hip_pitch",
                                                        "r_hip_roll", "r_knee"],
                         [0.7725, -0.0429, -0.5874, -0.0429, 1.3598, -0.7725, 0.0429, 0.5874, 0.0429, -1.3598])
        except (rospy.ServiceException) as e:
            print("reset_joints failed!")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        robot_state.reset_state()
        self.__reset_action()
        self.__reset_walking_para()
        self.episode_step = 0
        apply_para.publish(walking_para)
        time.sleep(0.5)
        send_command.publish("start")
        time.sleep(0.5)
        state, _ = self.__get_observation()
        return state

    def __reset_action(self):
        # reset robot action
        pub_l_ank_pitch.publish(0.7725)
        pub_l_ank_roll.publish(-0.0429)
        pub_l_hip_pitch.publish(-0.5874)
        pub_l_hip_roll.publish(-0.0429)
        pub_l_knee.publish(1.3598)
        pub_r_ank_pitch.publish(-0.7725)
        pub_r_ank_roll.publish(0.0429)
        pub_r_hip_pitch.publish(0.5874)
        pub_r_hip_roll.publish(0.0429)
        pub_r_knee.publish(-1.3598)
        self.rate.sleep()
        # time.sleep(0.008)

    def __apply_action(self, trj_para):
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     unpause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")
        walking_para.x_move_amplitude = trj_para[0]
        walking_para.z_move_amplitude = trj_para[1]
        walking_para.y_swap_amplitude = trj_para[2]
        walking_para.z_swap_amplitude = trj_para[3]
        walking_para.dsp_ratio = trj_para[4]
        # walking_para.period_time = trj_para[5]
        apply_para.publish(walking_para)
        # sleep_time = self.compute_sleep(trj_para[5], trj_para[4])
        wait_num = int(0.933 / 8 / 0.008)
        for _ in range(wait_num):
            self.rate.sleep()
        # time.sleep(0.933 / 8.)
        t = rospy.Time.now()
        # print(t - self.last_t)
        self.last_t = t
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     pause()
        # except (rospy.ServiceException) as e:
        #     print("rospause failed!'")

    def __reset_walking_para(self):
        walking_para.init_x_offset = -0.01
        walking_para.init_y_offset = 0.01
        walking_para.init_z_offset = 0.025
        walking_para.init_roll_offset = 0.0
        walking_para.init_pitch_offset = 0.0
        walking_para.init_yaw_offset = 0.0
        walking_para.period_time = 0.933
        walking_para.dsp_ratio = 0.15
        walking_para.step_fb_ratio = 0.30
        walking_para.x_move_amplitude = 0.00
        walking_para.y_move_amplitude = 0.0
        walking_para.z_move_amplitude = 0.02
        walking_para.angle_move_amplitude = 0.0
        walking_para.move_aim_on = False
        walking_para.balance_enable = False
        walking_para.balance_hip_roll_gain = 0.35
        walking_para.balance_knee_gain = 0.30
        walking_para.balance_ankle_roll_gain = 0.7
        walking_para.balance_ankle_pitch_gain = 0.9
        walking_para.y_swap_amplitude = 0.018
        walking_para.z_swap_amplitude = 0.006
        walking_para.arm_swing_gain = 0.20
        walking_para.pelvis_offset = 0.0872664600611
        walking_para.hip_pitch_offset = 0.0
        walking_para.p_gain = 0
        walking_para.i_gain = 0
        walking_para.d_gain = 0

    def close(self):
        pass

    def __get_observation(self):
        robot_state.set_robot_state()
        observation = robot_state.pybullet_state
        state_info = {"base_position": [robot_state.body_x, robot_state.body_y, robot_state.body_height],
                      "base_orientation": robot_state.orientation,
                      "effort": robot_state.effort,
                      "base_euler": robot_state.euler}
        return np.array(observation, dtype=np.float), state_info

    def get_dynamic(self):
        mass = 1.8238
        friction = 0.6
        damping = 10
        stiffness = 80000
        return [mass, friction, damping / 10., stiffness / 1e5, 0.933]


class CallBackData(object):
    def __init__(self):
        self.init = 0

    def callbackJointStates(self, data):
        if len(data.velocity) != 0:
            # callback position
            data_position = list(data.position)
            robot_state.position = data_position
            # callback effort
            data_effort = list(data.effort)
            robot_state.effort = data_effort
            # callback effort
            data_velocity = list(data.velocity)
            robot_state.velocity = data_velocity
        else:
            robot_state.reset_state()
        # robot_state.set_robot_state()

    def callback_odom(self, data):
        robot_state.body_height = data.pose.pose.position.z
        robot_state.body_x = data.pose.pose.position.x
        robot_state.body_y = data.pose.pose.position.y
        # robot_state.set_robot_state()

    def callback_r_contact(self, data):
        if len(data.states) >= 10:
            robot_state.r_contact = 1
        else:
            robot_state.r_contact = 0
        # robot_state.set_robot_state()

    def callback_l_contact(self, data):
        if len(data.states) >= 10:
            robot_state.l_contact = 1
        else:
            robot_state.l_contact = 0
        # robot_state.set_robot_state()

    def callback_imu(self, imu_data):
        robot_state.imu[0] = imu_data.angular_velocity.x
        robot_state.imu[1] = imu_data.angular_velocity.y
        robot_state.imu[2] = imu_data.angular_velocity.z
        robot_state.imu[3] = imu_data.linear_acceleration.x
        robot_state.imu[4] = imu_data.linear_acceleration.y
        robot_state.imu[5] = imu_data.linear_acceleration.z
        robot_state.orientation[0] = imu_data.orientation.x
        robot_state.orientation[1] = imu_data.orientation.y
        robot_state.orientation[2] = imu_data.orientation.z
        robot_state.orientation[3] = imu_data.orientation.w
        # euler
        euler = pybullet.getEulerFromQuaternion(robot_state.orientation)
        robot_state.euler = list(euler)
        # robot_state.set_robot_state()


def listener():
    print("Listener")
    call = CallBackData()
    rospy.Subscriber("/r_ank_roll_link_contact_sensor_state", ContactsState, call.callback_r_contact)
    rospy.Subscriber("/l_ank_roll_link_contact_sensor_state", ContactsState, call.callback_l_contact)
    rospy.Subscriber("/odom/body", Odometry, call.callback_odom)
    rospy.Subscriber("/imu", Imu, call.callback_imu)
    rospy.Subscriber("/exhx5/joint_states", JointState, call.callbackJointStates)


if __name__ == '__main__':
    env = Exhx5GazeboEnv()
    print(env.observation_space.low)
    print(env.observation_space.high)
    env.reset()
    t = 0
    states = np.zeros((50, 20))
    step = 0
    steps = np.arange(0, 50)
    while step < 50:
        # print(f"observation:{observation}")
        act = np.random.uniform(ACTION_BIAS - ACTION_SCALE, ACTION_BIAS + ACTION_SCALE)
        # if t > 100:
        #     act = np.array([0.03, 0.025, 0.015, 0.01, 0.2])
        # else:
        #     act = np.array([0., 0.025, 0.015, 0.01, 0.2])
        act = ACTION_BIAS
        state, _, _, _ = env.step(act)
        states[step] = state
        step += 1
        # print(f"observation:{obs}, reward:{r}, info:{info}")

    for i in range(20):
        plt.plot(steps, states[:, i], label=f'state{i}')
    plt.legend()
    plt.show()