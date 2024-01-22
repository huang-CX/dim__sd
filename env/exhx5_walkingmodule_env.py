from abc import ABC
import pybullet
from pybullet_utils.bullet_client import BulletClient
import gym
import pybullet_data
import numpy as np
import math
import time
from env.walkingmodule import WalkingModule
from gym import spaces
from gym.utils import seeding
import sys

# x_move_amplitude, z_move_amplitude, x_swap_amplitude, z_swap_amplitude, dsp_ratio
ACTION_BIAS = np.array([0.03, 0.025, 0.015, 0.01, 0.2])
ACTION_SCALE = np.array([0.02, 0.015, 0.01, 0.01, 0.1])
# l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll
INIT_JOINT = np.array([-0.0429, -0.5874, 1.3598, 0.7725, -0.0429, 0.0429, 0.5874, -1.3598, -0.7725, 0.0429])
START_POSITION = [0, 0, 0.18]
START_ORIENTATION = pybullet.getQuaternionFromEuler([0, 0, 0])
JOINT_LOW = -3.14
JOINT_HIGH = 3.14
CONTROL_FREQUENCY = 8.
dt = 1. / 240.
PI = math.pi
MAX_EPISODE_STEP = 2000
LATENCY = 1


class Exhx5WalkEnv(gym.Env, ABC):
    """
    walking module env class
    action space: the parameters of the walking module
    """

    def __init__(self, render=False):

        self._render = render
        self.p = BulletClient(connection_mode=(pybullet.GUI if self._render else pybullet.DIRECT))
        # self._physics_client_id = self.p.connect(self.p.GUI if self._render else self.p.DIRECT)
        self.np_random = None
        self.seed()
        self._physics_client_id = 0
        self.urdf_path = "env/robot_urdf/exhx5_w_b_16.urdf"
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.8)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.p.loadURDF("plane.urdf")
        self.robot = self.p.loadURDF(self.urdf_path, START_POSITION, START_ORIENTATION)

        self.set_dynamic_init_param()
        self.period_time = 0.933
        self.kp = 0.3
        self.kd = 1.6

        self.joints_num = self.p.getNumJoints(bodyUniqueId=self.robot)
        self.joint_indices = range(self.joints_num)
        self.use_joint_indices = []
        self.controlMode = self.p.POSITION_CONTROL
        for j in self.joint_indices:
            info = self.p.getJointInfo(self.robot, j, )
            joint_name = info[1].decode("ascii")
            if info[2] != self.p.JOINT_REVOLUTE:
                continue
            self.use_joint_indices.append(j)
            # self.p.enableJointForceTorqueSensor(self.robot, j, True)
            lower, upper = (info[8], info[9])
        self.maxControlForces = np.ones_like(self.use_joint_indices) * 1

        self.walkingModule = WalkingModule()
        self.walkingModule.init_walking_param(init_x_offset=-0.01,
                                              init_y_offset=0.01,
                                              init_z_offset=0.025,
                                              init_roll_offset=0.0,
                                              init_pitch_offset=0.0,
                                              init_yaw_offset=0.0,
                                              period_time=self.period_time,
                                              dsp_ratio=0.2,
                                              step_fb_ratio=0.30,
                                              x_move_amplitude=0.0,
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
        low = np.array([-np.inf] * 10 + [JOINT_LOW] * len(self.use_joint_indices))
        high = np.array([np.inf] * 10 + [JOINT_HIGH] * len(self.use_joint_indices))
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=ACTION_BIAS - ACTION_SCALE, high=ACTION_BIAS + ACTION_SCALE)
        self.t = 0
        self.episode_step = 0

    def reset(self):
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.8)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.p.loadURDF("plane.urdf")
        self.robot = self.p.loadURDF(self.urdf_path, START_POSITION, START_ORIENTATION)
        self.set_dynamic_init_param()
        for index, i in enumerate(self.use_joint_indices):
            self.p.resetJointState(self.robot,
                                   jointIndex=i,
                                   targetValue=INIT_JOINT[index])
            # self.p.setJointMotorControl2(bodyIndex=self.robot,
            #                         ,
            #                         jointIndex=i,
            #                         controlMode=self.p.POSITION_CONTROL,
            #                         targetPosition=INIT_JOINT[index],
            #                         targetVelocity=0,
            #                         force=4,
            #                         positionGain=Kp,
            #                         velocityGain=Kd)
        # for _ in range(100):
        self.p.stepSimulation()
        # time.sleep(1./240.)
        state, _ = self.__get_observation()
        self.episode_step = 0
        return state

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
        effort_reward = 3.0 - 0.25 * np.sum(abs(np.asarray(state_next_info['joint_torque'])))
        effort_reward = np.clip(effort_reward, -0.5, 0.5)

        # fall reward
        if robot_height <= 0.165 or robot_height >= 2.05:
            fall_reward = -50
            done = True
        if abs(euler_next[0]) >= 20. * PI / 180. or abs(euler_next[1]) >= 20. * PI / 180. or abs(
                euler_next[2]) >= 30. * PI / 180.:
            fall_reward = -50
            done = True
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

    def __apply_action(self, action):
        # assert len(action) == 5, 'Wrong action dimension! '
        self.walkingModule.update_walking_param(x_move_amplitude=action[0],
                                                z_move_amplitude=action[1],
                                                y_swap_amplitude=action[2],
                                                z_swap_amplitude=action[3],
                                                dsp_ratio=action[4],
                                                period_time=self.period_time)
        iter_num = int(np.ceil((self.period_time / CONTROL_FREQUENCY) / dt))
        iter_num += LATENCY * np.random.randint(iter_num / 8, iter_num / 4)  # simulate gazebo latency
        for _ in range(iter_num):
            joint_positions = self.walkingModule.compute_motor_angles(self.t)
            for index, i in enumerate(self.use_joint_indices):
                self.p.setJointMotorControl2(bodyIndex=self.robot,
                                             jointIndex=i,
                                             controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=joint_positions[index] + np.random.normal(0, 0.01),
                                             targetVelocity=0,
                                             force=4,
                                             positionGain=self.kp,
                                             velocityGain=self.kd)
            self.t = self.t + dt
            if self.t > self.period_time:
                self.t = 0.
            self.p.stepSimulation()
            if self._render:
                time.sleep(dt)
            pos, _ = self.p.getBasePositionAndOrientation(self.robot)
            self.p.resetDebugVisualizerCamera(cameraDistance=2,
                                              cameraYaw=90,
                                              cameraPitch=-20,
                                              cameraTargetPosition=pos)

    def __get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        base_euler = self.p.getEulerFromQuaternion(base_ori)
        linear_vel, angular_vel = self.p.getBaseVelocity(self.robot)
        joint_states = self.p.getJointStates(self.robot,
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
        state_info = {'base_position': base_pos,
                      'base_oritation': base_ori,
                      'base_euler': base_euler,
                      'joint_force': joints_force,
                      'joint_torque': joints_torque}
        # print(angular_vel)
        return np.concatenate([angular_vel, base_ori, base_euler, joints_pos]), state_info

    def close(self):
        if self._physics_client_id >= 0:
            self.p.disconnect()
        self._physics_client_id = -1

    def render(self, mode="human"):
        # _, _, px, _, _ = self.p.getCameraImage(320, 240)
        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        # return np.array(px, dtype=np.uint8)
        pass

    def seed(self, seed=None):
        self.np_random, np_seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def set_dynamic_init_param(self):
        self.p.changeDynamics(self.plane, -1, lateralFriction=0.5)
        for i in range(13):
            self.p.changeDynamics(self.robot, i, lateralFriction=0.2)
        self.p.changeDynamics(self.robot, 6, contactStiffness=80000, contactDamping=10, lateralFriction=0.6)
        self.p.changeDynamics(self.robot, 12, contactStiffness=80000, contactDamping=10, lateralFriction=0.6)
        self.period_time = 0.933

    def change_dynamic(self):
        self.p.changeDynamics(self.plane, -1, lateralFriction=0.5)
        stiffness = np.random.uniform(40000, 120000)
        damping = np.random.uniform(6, 12)
        friction = np.random.uniform(0.2, 0.8)
        mass_head = np.random.uniform(1.8238 - 0.1, 1.8238 + 0.1)
        # self.kp = np.random.uniform(0.2, 0.4)
        # self.kd = np.random.uniform(1., 2.)
        self.period_time = np.random.uniform(0.8, 0.95)
        self.p.changeDynamics(self.robot, -1, mass=mass_head)
        self.p.changeDynamics(self.robot, 6, contactStiffness=stiffness, contactDamping=damping,
                              lateralFriction=friction)
        self.p.changeDynamics(self.robot, 12, contactStiffness=stiffness, contactDamping=damping,
                              lateralFriction=friction)
        # print('dynamic', stiffness, damping, friction, mass_head)

    def change_adapt_dynamic(self, stiffness, damping):
        self.p.changeDynamics(self.plane, -1, lateralFriction=0.5)
        self.p.changeDynamics(self.robot, 6, contactStiffness=stiffness, contactDamping=damping,)
        self.p.changeDynamics(self.robot, 12, contactStiffness=stiffness, contactDamping=damping,)
        # print('dynamic', stiffness, damping, friction, mass_head)

    def change_adapt_friction(self, friction):
        self.p.changeDynamics(self.plane, -1, lateralFriction=0.5)
        self.p.changeDynamics(self.robot, 6, lateralFriction=friction)
        self.p.changeDynamics(self.robot, 12, lateralFriction=friction)
        # print('dynamic', stiffness, damping, friction, mass_head)

    def change_adapt_velocity(self, period_time):
        self.p.changeDynamics(self.plane, -1, lateralFriction=0.5)
        self.period_time = period_time

    def get_dynamic(self):
        base = self.p.getDynamicsInfo(self.robot, -1)
        left_leg = self.p.getDynamicsInfo(self.robot, 6)
        right_leg = self.p.getDynamicsInfo(self.robot, 6)
        return base[0], left_leg[1], left_leg[8] / 10, left_leg[9] / 1e5, self.period_time
        # mass = 1.8238
        # friction = 0.6
        # damping = 10
        # stiffness = 80000
        # mass = 0.
        # friction = 0.
        # damping = 0.
        # stiffness = 0.
        # return [mass, friction, damping / 10., stiffness / 1e5]


if __name__ == "__main__":
    env = Exhx5WalkEnv(render=True)
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
