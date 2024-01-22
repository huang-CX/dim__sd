from abc import ABC

import pybullet as p
import time
import gym
import pybullet_data
import numpy as np
from gym import spaces
from pybullet_examples.pdControllerStable import PDControllerStable


class Exhx5Env(gym.Env, ABC):
    """
    basic env
    action space: joint positions
    """
    def __init__(self, render=True):
        self._render = render
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        # self.timeStep =
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        self.startPos = [0, 0, 0.25]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("robot_urdf/exhx5_w_b_16.urdf", self.startPos, self.startOrientation)

        self.jointsNum = p.getNumJoints(bodyUniqueId=self.robot)
        self.jointIndices = range(self.jointsNum)
        self.useJointIndices = []
        self.pd = p.loadPlugin("pdControlPlugin")
        self.controlMode = p.POSITION_CONTROL
        for j in self.jointIndices:
            info = p.getJointInfo(self.robot, j, physicsClientId=self._physics_client_id)
            jointName = info[1].decode("ascii")
            if info[2] != p.JOINT_REVOLUTE:
                continue
            self.useJointIndices.append(j)
            lower, upper = (info[8], info[9])
            print(f"jointName:{jointName}, posLow:{lower}, posHigh:{upper}")
        self.maxControlForces = np.ones_like(self.useJointIndices) * 1
        print(f"useJointIndices:{self.useJointIndices}")
        self.sPD = PDControllerStable(p)

        low = np.array([-np.inf] * 10 + [-1.67] * len(self.useJointIndices))
        high = np.array([np.inf] * 10 + [1.67] * len(self.useJointIndices))
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(np.array([-3.14] * len(self.useJointIndices)),
                                       np.array([3.14] * len(self.useJointIndices)))
        self.t = 0
        self.useSimulation = 1

    def __apply_action(self, action):
        target_position = action
        target_velocity = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if self.useSimulation:
            p.setJointMotorControlArray(self.robot,
                                        jointIndices=self.useJointIndices,
                                        targetPositions=target_position,
                                        targetVelocities=target_velocity,
                                        controlMode=self.controlMode,
                                        positionGains=[0.3] * 10,
                                        velocityGains=[1.6] * 10,
                                        forces=[4] * 10)
        else:
            for index, j in enumerate(self.useJointIndices):
                p.resetJointState(self.robot, j, target_position[index])

    def __get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        linea_vel, angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self._physics_client_id)
        joint_states = p.getJointStates(self.robot,
                                        physicsClientId=self._physics_client_id,
                                        jointIndices=self.useJointIndices)
        joints_pos = []
        joints_vel = []
        for joint in joint_states:
            joint_pos, joint_vel, _, _ = joint
            joints_pos.append(joint_pos)
            joints_vel.append(joint_vel)
        return np.concatenate([base_pos, base_ori, linea_vel, joints_pos])

    def reset(self):
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("robot_urdf/exhx5_w_b_16.urdf", self.startPos, self.startOrientation)
        p.stepSimulation()
        return self.__get_observation()

    def step(self, action):
        # targetPosition = np.random.uniform(-0.1, 0.1, (len(self.useJointIndices), ))
        dt = 1. / 240.
        state_current = self.__get_observation()
        efford_weight = 0.001
        step_weight = 1.
        self.__apply_action(action)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        time.sleep(dt)
        state_next = self.__get_observation()
        done = False
        fall_reward = 0
        if state_next[2] <= 0.15:
            fall_reward = -10
            done = True
        step_reward = step_weight * (state_next[0] - state_current[0]) / dt
        efford_reward = efford_weight * np.linalg.norm(action)**2
        reward = step_reward + efford_reward + fall_reward
        info = {
            'step_reward': step_reward,
            'efford_reward': efford_reward,
            'fall_reward': fall_reward
        }
        return state_next, reward, done, info

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1


if __name__ == "__main__":
    env = Exhx5Env()
    print(env.observation_space.low)
    print(env.observation_space.high)
    observation = env.reset()
    while True:
        action = np.random.uniform(-1.67, 1.67, size=(10,))
        observation, reward, done, info = env.step(action)
        # print(f"observation:{observation}, reward:{reward}, info:{info}")
