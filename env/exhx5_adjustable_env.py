from abc import ABC
import pybullet
from pybullet_utils.bullet_client import BulletClient
import time
import gym
import pybullet_data
import numpy as np
import math
import torch
from env.walkingmodule import WalkingModule
from gym import spaces

USE_REAL_TIME = 0
# x_move_amplitude, z_move_amplitude, x_swap_amplitude, z_swap_amplitude, dsp_ratio
ACTION_BIAS = np.array([0.03, 0.025, 0.015, 0.01, 0.2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
ACTION_SCALE = np.array([0.02, 0.015, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll
INIT_JOINT = np.array([-0.0429, -0.5874, 1.3598, 0.7725, -0.0429, 0.0429, 0.5874, -1.3598, -0.7725, 0.0429])
START_POSITION = [0, 0, 0.18]
START_ORIENTATION = pybullet.getQuaternionFromEuler([0, 0, 0])
OBSERVATION_LOW = -3.14
OBSERVATION_HIGH = 3.14
Kp = 0.3
Kd = 1.6
PERIOD_TIME = 0.933
CONTROL_FREQUENCY = 60.
dt = 1. / 240.
PI = math.pi


class Exhx5AdjEnv(gym.Env, ABC):
    def __init__(self, render=False):
        self._render = render
        self._p = BulletClient(connection_mode=(pybullet.GUI if self._render else pybullet.DIRECT))
        # self._physics_client_id = self._p.connect(self._p.GUI if self._render else self._p.DIRECT)
        self._physics_client_id = 0
        # self.timeStep =
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.8)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self._p.loadURDF("plane.urdf")
        self.robot = self._p.loadURDF("/home/zhou/exhx5_pybullet/exhx5_w_b_16.urdf", START_POSITION, START_ORIENTATION)

        self.joints_num = self._p.getNumJoints(bodyUniqueId=self.robot)
        self.joint_indices = range(self.joints_num)
        self.use_joint_indices = []
        self.controlMode = self._p.POSITION_CONTROL
        for j in self.joint_indices:
            info = self._p.getJointInfo(self.robot, j, )
            joint_name = info[1].decode("ascii")
            if info[2] != self._p.JOINT_REVOLUTE:
                continue
            self.use_joint_indices.append(j)
            # self._p.enableJointForceTorqueSensor(self.robot, j, True)
            lower, upper = (info[8], info[9])
        self.maxControlForces = np.ones_like(self.use_joint_indices) * 1

        self.walkingModule = WalkingModule()
        self.walkingModule.init_walking_param(init_x_offset=-0.01,
                                              init_y_offset=0.01,
                                              init_z_offset=0.025,
                                              init_roll_offset=0.0,
                                              init_pitch_offset=0.0,
                                              init_yaw_offset=0.0,
                                              period_time=PERIOD_TIME,
                                              dsp_ratio=0.2,
                                              step_fb_ratio=0.30,
                                              x_move_amplitude=0.03,
                                              y_move_amplitude=0.0,
                                              z_move_amplitude=0.02,
                                              angle_move_amplitude=0.0,
                                              move_aim_on=False,
                                              balance_enable=False,
                                              balance_hip_roll_gain=0.35,
                                              balance_knee_gain=0.30,
                                              balance_ankle_roll_gain=0.7,
                                              balance_ankle_pitch_gain=0.9,
                                              y_swap_amplitude=0.018,
                                              z_swap_amplitude=0.006,
                                              arm_swing_gain=0.20,
                                              pelvis_offset=5.0,
                                              hip_pitch_offset=0.0)
        low = np.array([-np.inf] * 10 + [OBSERVATION_LOW] * len(self.use_joint_indices))
        high = np.array([np.inf] * 10 + [OBSERVATION_HIGH] * len(self.use_joint_indices))
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=ACTION_BIAS - ACTION_SCALE, high=ACTION_BIAS + ACTION_SCALE)
        self.t = 0
        print('Adjustable Walking Module Env')

    def reset(self):
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.8, )
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self._p.loadURDF("plane.urdf")
        self.robot = self._p.loadURDF("/home/zhou/exhx5_pybullet/exhx5_w_b_16.urdf", START_POSITION, START_ORIENTATION)
        for index, i in enumerate(self.use_joint_indices):
            self._p.resetJointState(self.robot,
                                    jointIndex=i,
                                    targetValue=INIT_JOINT[index])
            # self._p.setJointMotorControl2(bodyIndex=self.robot,
            #                         ,
            #                         jointIndex=i,
            #                         controlMode=self._p.POSITION_CONTROL,
            #                         targetPosition=INIT_JOINT[index],
            #                         targetVelocity=0,
            #                         force=4,
            #                         positionGain=Kp,
            #                         velocityGain=Kd)
        # for _ in range(100):
        self._p.stepSimulation()
        # time.sleep(1.)
        state, _ = self.__get_observation()
        return state

    def step(self, action):
        state_current, state_current_info = self.__get_observation()
        self.__apply_action(action)
        state_next, state_next_info = self.__get_observation()
        euler_current = pybullet.getEulerFromQuaternion([state_current[3], state_current[4], state_current[5], state_current[6]])
        euler_next = pybullet.getEulerFromQuaternion([state_next[3], state_next[4], state_next[5], state_next[6]])
        done = False
        fall_reward = 0.0

        # step_reward
        step_vector = [(state_next[0] - state_current[0]), (state_next[1] - state_current[1])]
        # step_vector = [(robot_state.body_x - robot_state.last_body_x)]
        step_vector = np.asarray(step_vector)
        step_len_x = np.linalg.norm(step_vector[0])
        step_len = np.linalg.norm(step_vector)

        if (state_next[0] - state_current[0]) > 0:
            step_reward = min(step_len_x * 100, 3)
        else:
            step_reward = -min(abs(step_len_x * 100), 3)

        # height reward
        robot_height = state_next[2]
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
        effort_reward = 3.0 - 0.25 * np.sum(abs(np.asarray(state_next_info['joint_torque'])))
        effort_reward = np.clip(effort_reward, -0.5, 0.5)

        # fall reward
        if robot_height <= 0.165 or robot_height >= 2.05:
            fall_reward = -50
            done = True
        if abs(euler_next[0]) >= 20. * PI / 180. or abs(euler_next[1]) >= 20. * PI / 180. or abs(euler_next[2]) >= 30. * PI / 180.:
            fall_reward = -50
            done = True
        # print(f"step:{step_reward}, fall:{fall_reward}, height:{height_reward}, effort:{effort_reward}")
        reward = step_reward + effort_reward + height_reward + fall_reward + orientation_reward
        step_info = {
            'step_reward': step_reward,
            'effort_reward': effort_reward,
            'fall_reward': fall_reward,
            'orientation_reward': orientation_reward,
            'height_reward': height_reward
        }
        return state_next, reward, done, step_info

    def __apply_action(self, action):
        # assert len(action) == 5, 'Wrong action dimension! '
        self.walkingModule.update_walking_param(x_move_amplitude=action[0],
                                                z_move_amplitude=action[1],
                                                y_swap_amplitude=action[2],
                                                z_swap_amplitude=action[3],
                                                dsp_ratio=action[4])
        iter_num = int(np.ceil((PERIOD_TIME / CONTROL_FREQUENCY) / dt))
        for _ in range(iter_num):
            joint_positions = self.walkingModule.compute_motor_angles(self.t)
            for index, i in enumerate(self.use_joint_indices):
                self._p.setJointMotorControl2(bodyIndex=self.robot,
                                              jointIndex=i,
                                              controlMode=self._p.POSITION_CONTROL,
                                              targetPosition=joint_positions[index] + action[index + 5],
                                              targetVelocity=0,
                                              force=4,
                                              positionGain=Kp,
                                              velocityGain=Kd)
            self.t = self.t + dt
            if self.t > PERIOD_TIME:
                self.t = 0.
            self._p.stepSimulation()
            if self._render:
                time.sleep(dt)
            pos, _ = self._p.getBasePositionAndOrientation(self.robot)
            self._p.resetDebugVisualizerCamera(cameraDistance=3,
                                               cameraYaw=50,
                                               cameraPitch=-35,
                                               cameraTargetPosition=pos)

    def __get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        base_pos, base_ori = self._p.getBasePositionAndOrientation(self.robot)
        euler = self._p.getEulerFromQuaternion(base_ori)
        linear_vel, angular_vel = self._p.getBaseVelocity(self.robot)
        joint_states = self._p.getJointStates(self.robot,
                                              jointIndices=self.use_joint_indices)
        joints_pos = []
        joints_vel = []
        joints_force = []
        joints_torque = []
        for joint in joint_states:
            joint_pos, joint_vel, joint_force, joint_torque = joint
            joints_pos.append(joint_pos)
            joints_vel.append(joint_vel)
            joints_force.append(joint_force)
            joints_torque.append(joint_torque)
        state_info = {'joint_force': joints_force,
                      'joint_torque': joints_torque}
        return np.concatenate([angular_vel, base_ori, euler, joints_pos]), state_info

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    env = Exhx5AdjEnv(render=True)
    print(env.observation_space.low)
    print(env.observation_space.high)
    env.reset()
    while True:
        # print(f"observation:{observation}")
        act = np.random.uniform(ACTION_BIAS - ACTION_SCALE, ACTION_BIAS + ACTION_SCALE)
        # act = ACTION_BIAS
        obs, r, d, info = env.step(act)
        # print(f"observation:{obs}, reward:{r}, info:{info}")
