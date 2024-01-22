import numpy
import time
import math


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
        t = time.time()
        self.position_contact = [self.body_x, self.body_y, self.body_height, self.l_contact, self.r_contact]
        # len=43, position=[0:12], velocity=[12: 24], effort=[24:36], orientation=[36:40],
        # len=28, position=[0:12], orientation=[12:16], contact=[17:19], euler = [19:22], imu=[22:28]
        self.state = self.position + self.imu + self.orientation + self.euler
        self.full_state = self.position + self.imu + self.orientation + self.euler + self.position_contact + \
                          self.l_contact_force + self.r_contact_force + self.l_position + self.r_position
        # self.pybullet_state = [self.body_x, self.body_y, self.body_height] + self.orientation + \
        #                       [(self.body_x - self.last_body_x) / (t - self.last_time),
        #                        (self.body_y - self.last_body_y) / (t - self.last_time),
        #                        (self.body_height - self.last_body_height) / (t - self.last_time)] + \
        #                       [self.position[2], self.position[3], self.position[4], self.position[0], self.position[1],
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
