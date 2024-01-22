import numpy as np
import math
import pybullet as p

pi = math.pi


def w_sin(time, period, period_shift, mag, mag_shift):
    return mag * math.sin(2 * pi * time / period - period_shift) + mag_shift


class Pose3D:
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 roll: float = 0.0,
                 pitch: float = 0.0,
                 yaw: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return "x:{}, y:{}, z:{}, roll:{}, pitch:{}, yaw:{}".format(self.x, self.y, self.z, self.roll, self.pitch,
                                                                    self.yaw)


class WalkingParams:
    def __init__(self):
        self.init_x_offset = -0.000
        self.init_y_offset = 0.000
        self.init_z_offset = -0.000

        self.init_roll_offset = 0.0
        self.init_pitch_offset = math.radians(0.0)
        self.init_yaw_offset = math.radians(0.0)
        self.hip_pitch_offset = math.radians(0.0)

        self.period_time = 600 * 0.001
        self.dsp_ratio = 0.1
        self.step_fb_ratio = 0.28

        self.x_move_amplitude = 0.0
        self.y_move_amplitude = 0.0
        self.z_move_amplitude = 0.020
        self.angle_move_amplitude = 0.0

        self.balance_enable = False
        self.balance_hip_roll_gain = 0.5
        self.balance_knee_gain = 0.3
        self.balance_ankle_roll_gain = 1.0
        self.balance_ankle_pitch_gain = 0.9
        self.y_swap_amplitude = 0.020
        self.z_swap_amplitude = 0.005
        self.pelvis_offset = math.radians(3.0)
        self.arm_swing_gain = 1.5
        self.move_aim_on = False

        self.body_swing_y = 0
        self.body_swing_z = 0

    def init_params(self, **kwargs):
        self.init_x_offset = kwargs['init_x_offset']
        self.init_y_offset = kwargs['init_y_offset']
        self.init_z_offset = kwargs['init_z_offset']

        self.init_roll_offset = kwargs['init_roll_offset']
        self.init_pitch_offset = kwargs['init_pitch_offset']
        self.init_yaw_offset = kwargs['init_yaw_offset']
        self.hip_pitch_offset = kwargs['hip_pitch_offset']

        self.period_time = kwargs['period_time']
        self.dsp_ratio = kwargs['dsp_ratio']
        self.step_fb_ratio = kwargs['step_fb_ratio']

        self.x_move_amplitude = kwargs['x_move_amplitude']
        self.y_move_amplitude = kwargs['y_move_amplitude']
        self.z_move_amplitude = kwargs['z_move_amplitude']
        self.angle_move_amplitude = kwargs['angle_move_amplitude']

        self.balance_enable = kwargs['balance_enable']
        self.balance_hip_roll_gain = kwargs['balance_hip_roll_gain']
        self.balance_knee_gain = kwargs['balance_knee_gain']
        self.balance_ankle_roll_gain = kwargs['balance_ankle_roll_gain']
        self.balance_ankle_pitch_gain = kwargs['balance_ankle_pitch_gain']
        self.y_swap_amplitude = kwargs['y_swap_amplitude']
        self.z_swap_amplitude = kwargs['z_swap_amplitude']
        self.pelvis_offset = kwargs['pelvis_offset']
        self.arm_swing_gain = kwargs['arm_swing_gain']

        # self.body_swing_y = kwargs['body_swing_y']
        # self.body_swing_z = kwargs['body_swing_z']

        self.move_aim_on = kwargs['move_aim_on']
        # self.ctrl_running_ = kwargs['ctrl_running_']
        # self.real_running_ = kwargs['real_running_']

    def update_params(self, **kwargs):
        if 'period_time' in kwargs.keys():
            self.period_time = kwargs['period_time']
        self.dsp_ratio = kwargs['dsp_ratio']

        self.x_move_amplitude = kwargs['x_move_amplitude']
        self.z_move_amplitude = kwargs['z_move_amplitude']

        self.y_swap_amplitude = kwargs['y_swap_amplitude']
        self.z_swap_amplitude = kwargs['z_swap_amplitude']


class WalkingModule:
    def __init__(self):
        # robot parameters
        self.thigh_length = 75.9 * 0.001
        self.calf_length = 74.6 * 0.001
        self.ankle_length = 20.2 * 0.001
        self.leg_side_offset_m_ = 70.0 * 0.001
        self.leg_length = self.thigh_length + self.calf_length + self.ankle_length

        # time parameters
        self.period_time_ = 0.0
        self.dsp_ratio_ = 0.0
        self.ssp_ratio_ = 0.0
        self.walking_param_ = WalkingParams()
        self.x_swap_period_time_ = 0.0
        self.x_move_period_time_ = 0.0
        self.y_swap_period_time_ = 0.0
        self.y_move_period_time_ = 0.0
        self.z_swap_period_time_ = 0.0
        self.z_move_period_time_ = 0.0
        self.a_move_period_time_ = 0.0
        self.ssp_time_ = 0.0
        self.l_ssp_start_time_ = 0.0
        self.l_ssp_end_time_ = 0.0
        self.r_ssp_start_time_ = 0.0
        self.r_ssp_end_time_ = 0.0
        self.phase1_time_ = 0.0
        self.phase2_time_ = 0.0
        self.phase3_time_ = 0.0
        self.pelvis_offset_ = 0.0
        self.pelvis_swing_ = 0.0

        # move parameters
        self.x_move_amplitude_ = 0.0
        self.x_swap_amplitude_ = 0.0
        self.y_move_amplitude_ = 0.0
        self.y_swap_amplitude_ = 0.0
        self.y_move_amplitude_shift_ = 0.0
        self.z_move_amplitude_ = 0.0
        self.z_swap_amplitude_ = 0.0
        self.z_move_amplitude_shift_ = 0.0
        self.z_swap_amplitude_shift_ = 0.0
        self.a_move_amplitude_ = 0.0
        self.a_move_amplitude_shift_ = 0.0
        self.previous_x_move_amplitude_ = 0.0
        self.swap = Pose3D()  # x, y, self.z, roll, pitch, yaw
        self.left_leg_move = Pose3D()  # x, y, self.z, roll, pitch, yaw
        self.right_leg_move = Pose3D()  # x, y, self.z, roll, pitch, yaw

        # pose parameters
        self.x_offset_ = 0.0
        self.y_offset_ = 0.0
        self.z_offset_ = 0.0
        self.r_offset_ = 0.0
        self.p_offset_ = 0.0
        self.a_offset_ = 0.0
        self.hit_pitch_offset_ = 0.0

        # some fixed parameters
        self.x_swap_phase_shift_ = pi
        self.x_swap_amplitude_shift_ = 0
        self.x_move_phase_shift_ = pi / 2
        self.x_move_amplitude_shift_ = 0
        self.y_swap_phase_shift_ = 0
        self.y_swap_amplitude_shift_ = 0
        self.y_move_phase_shift_ = pi / 2
        self.z_swap_phase_shift_ = pi * 3 / 2
        self.z_move_phase_shift_ = pi / 2
        self.a_move_phase_shift_ = pi / 2

        self.pelvis_offset_l = 0
        self.pelvis_offset_r = 0

        self.ctrl_running_ = False
        # initialize parameters
        self.update_time_param()
        self.update_move_param()
        self.update_pose_param()

    def enable_walking_module(self):
        self.ctrl_running_ = True

    def compute_motor_angles(self, time_, **kwargs):
        # self.walking_param_.update_params(**kwargs)
        self.update_move_param()
        self.update_time_param()
        self.update_pose_param()
        if time_ == 0.:
            self.previous_x_move_amplitude_ = self.x_move_amplitude_ * 0.5

        # endpoint
        self.swap.x = w_sin(time_, self.x_swap_period_time_,
                            self.x_swap_phase_shift_,
                            self.x_swap_amplitude_,
                            self.x_swap_amplitude_shift_)
        self.swap.y = w_sin(time_, self.y_swap_period_time_,
                            self.y_swap_phase_shift_,
                            self.y_swap_amplitude_,
                            self.y_swap_amplitude_shift_)
        self.swap.z = w_sin(time_,
                            self.z_swap_period_time_,
                            self.z_swap_phase_shift_,
                            self.z_swap_amplitude_,
                            self.z_swap_amplitude_shift_)
        self.swap.roll = 0.0
        self.swap.pitch = 0.0
        self.swap.yaw = 0.0

        if time_ <= self.l_ssp_start_time_:
            self.left_leg_move.x = w_sin(self.l_ssp_start_time_, self.x_move_period_time_,
                                         self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                         self.x_move_amplitude_,
                                         self.x_move_amplitude_shift_)
            self.left_leg_move.y = w_sin(self.l_ssp_start_time_, self.y_move_period_time_,
                                         self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                         self.y_move_amplitude_,
                                         self.y_move_amplitude_shift_)
            self.left_leg_move.z = w_sin(self.l_ssp_start_time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.z_move_amplitude_,
                                         self.z_move_amplitude_shift_)
            self.left_leg_move.yaw = w_sin(self.l_ssp_start_time_, self.a_move_period_time_,
                                           self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                           self.a_move_amplitude_, self.a_move_amplitude_shift_)
            self.right_leg_move.x = w_sin(self.l_ssp_start_time_, self.x_move_period_time_,
                                          self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                          -self.x_move_amplitude_, -self.x_move_amplitude_shift_)
            self.right_leg_move.y = w_sin(self.l_ssp_start_time_, self.y_move_period_time_,
                                          self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                          -self.y_move_amplitude_, -self.y_move_amplitude_shift_)
            self.right_leg_move.z = w_sin(self.r_ssp_start_time_, self.z_move_period_time_,
                                          self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                          self.z_move_amplitude_,
                                          self.z_move_amplitude_shift_)
            self.right_leg_move.yaw = w_sin(self.l_ssp_start_time_, self.a_move_period_time_,
                                            self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                            -self.a_move_amplitude_, -self.a_move_amplitude_shift_)
            self.pelvis_offset_l = 0
            self.pelvis_offset_r = 0
        elif time_ <= self.l_ssp_end_time_:
            self.left_leg_move.x = w_sin(time_, self.x_move_period_time_,
                                         self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                         self.x_move_amplitude_,
                                         self.x_move_amplitude_shift_)
            self.left_leg_move.y = w_sin(time_, self.y_move_period_time_,
                                         self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                         self.y_move_amplitude_,
                                         self.y_move_amplitude_shift_)
            self.left_leg_move.z = w_sin(time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.z_move_amplitude_,
                                         self.z_move_amplitude_shift_)
            self.left_leg_move.yaw = w_sin(time_, self.a_move_period_time_,
                                           self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                           self.a_move_amplitude_, self.a_move_amplitude_shift_)
            self.right_leg_move.x = w_sin(time_, self.x_move_period_time_,
                                          self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                          -self.x_move_amplitude_, -self.x_move_amplitude_shift_)
            self.right_leg_move.y = w_sin(time_, self.y_move_period_time_,
                                          self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                          -self.y_move_amplitude_, -self.y_move_amplitude_shift_)
            self.right_leg_move.z = w_sin(self.r_ssp_start_time_, self.z_move_period_time_,
                                          self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                          self.z_move_amplitude_,
                                          self.z_move_amplitude_shift_)
            self.right_leg_move.yaw = w_sin(time_, self.a_move_period_time_,
                                            self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                            -self.a_move_amplitude_, -self.a_move_amplitude_shift_)
            self.pelvis_offset_l = w_sin(time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.pelvis_swing_ / 2,
                                         self.pelvis_swing_ / 2)
            self.pelvis_offset_r = 0.25 * w_sin(time_, self.z_move_period_time_,
                                                self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                                -self.pelvis_offset_ / 2, -self.pelvis_offset_ / 2)
        elif time_ <= self.r_ssp_start_time_:
            self.left_leg_move.x = w_sin(self.l_ssp_end_time_, self.x_move_period_time_,
                                         self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                         self.x_move_amplitude_,
                                         self.x_move_amplitude_shift_)
            self.left_leg_move.y = w_sin(self.l_ssp_end_time_, self.y_move_period_time_,
                                         self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                         self.y_move_amplitude_,
                                         self.y_move_amplitude_shift_)
            self.left_leg_move.z = w_sin(self.l_ssp_end_time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.z_move_amplitude_,
                                         self.z_move_amplitude_shift_)
            self.left_leg_move.yaw = w_sin(self.l_ssp_end_time_, self.a_move_period_time_,
                                           self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                           self.a_move_amplitude_, self.a_move_amplitude_shift_)
            self.right_leg_move.x = w_sin(self.l_ssp_end_time_, self.x_move_period_time_,
                                          self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.l_ssp_start_time_,
                                          -self.x_move_amplitude_, -self.x_move_amplitude_shift_)
            self.right_leg_move.y = w_sin(self.l_ssp_end_time_, self.y_move_period_time_,
                                          self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.l_ssp_start_time_,
                                          -self.y_move_amplitude_, -self.y_move_amplitude_shift_)
            self.right_leg_move.z = w_sin(self.r_ssp_start_time_, self.z_move_period_time_,
                                          self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                          self.z_move_amplitude_,
                                          self.z_move_amplitude_shift_)
            self.right_leg_move.yaw = w_sin(self.l_ssp_end_time_, self.a_move_period_time_,
                                            self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.l_ssp_start_time_,
                                            -self.a_move_amplitude_, -self.a_move_amplitude_shift_)
            self.pelvis_offset_l = 0
            self.pelvis_offset_r = 0
        elif time_ <= self.r_ssp_end_time_:
            self.left_leg_move.x = w_sin(time_, self.x_move_period_time_,
                                         self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.r_ssp_start_time_ + pi,
                                         self.x_move_amplitude_, self.x_move_amplitude_shift_)
            self.left_leg_move.y = w_sin(time_, self.y_move_period_time_,
                                         self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.r_ssp_start_time_ + pi,
                                         self.y_move_amplitude_, self.y_move_amplitude_shift_)
            self.left_leg_move.z = w_sin(self.l_ssp_end_time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.z_move_amplitude_,
                                         self.z_move_amplitude_shift_)
            self.left_leg_move.yaw = w_sin(time_, self.a_move_period_time_,
                                           self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.r_ssp_start_time_ + pi,
                                           self.a_move_amplitude_, self.a_move_amplitude_shift_)
            self.right_leg_move.x = w_sin(time_, self.x_move_period_time_,
                                          self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.r_ssp_start_time_ + pi,
                                          -self.x_move_amplitude_, -self.x_move_amplitude_shift_)
            self.right_leg_move.y = w_sin(time_, self.y_move_period_time_,
                                          self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.r_ssp_start_time_ + pi,
                                          -self.y_move_amplitude_, -self.y_move_amplitude_shift_)
            self.right_leg_move.z = w_sin(time_, self.z_move_period_time_,
                                          self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                          self.z_move_amplitude_,
                                          self.z_move_amplitude_shift_)
            self.right_leg_move.yaw = w_sin(time_, self.a_move_period_time_,
                                            self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.r_ssp_start_time_ + pi,
                                            -self.a_move_amplitude_, -self.a_move_amplitude_shift_)
            self.pelvis_offset_l = 0.25 * w_sin(time_, self.z_move_period_time_,
                                                self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                                self.pelvis_offset_ / 2,
                                                self.pelvis_offset_ / 2)
            self.pelvis_offset_r = w_sin(time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                         -self.pelvis_swing_ / 2,
                                         -self.pelvis_swing_ / 2)
        else:
            self.left_leg_move.x = w_sin(self.r_ssp_end_time_, self.x_move_period_time_,
                                         self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.r_ssp_start_time_ + pi,
                                         self.x_move_amplitude_, self.x_move_amplitude_shift_)
            self.left_leg_move.y = w_sin(self.r_ssp_end_time_, self.y_move_period_time_,
                                         self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.r_ssp_start_time_ + pi,
                                         self.y_move_amplitude_, self.y_move_amplitude_shift_)
            self.left_leg_move.z = w_sin(self.l_ssp_end_time_, self.z_move_period_time_,
                                         self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.l_ssp_start_time_,
                                         self.z_move_amplitude_,
                                         self.z_move_amplitude_shift_)
            self.left_leg_move.yaw = w_sin(self.r_ssp_end_time_, self.a_move_period_time_,
                                           self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.r_ssp_start_time_ + pi,
                                           self.a_move_amplitude_, self.a_move_amplitude_shift_)
            self.right_leg_move.x = w_sin(self.r_ssp_end_time_, self.x_move_period_time_,
                                          self.x_move_phase_shift_ + 2 * pi / self.x_move_period_time_ * self.r_ssp_start_time_ + pi,
                                          -self.x_move_amplitude_, -self.x_move_amplitude_shift_)
            self.right_leg_move.y = w_sin(self.r_ssp_end_time_, self.y_move_period_time_,
                                          self.y_move_phase_shift_ + 2 * pi / self.y_move_period_time_ * self.r_ssp_start_time_ + pi,
                                          -self.y_move_amplitude_, -self.y_move_amplitude_shift_)
            self.right_leg_move.z = w_sin(self.r_ssp_end_time_, self.z_move_period_time_,
                                          self.z_move_phase_shift_ + 2 * pi / self.z_move_period_time_ * self.r_ssp_start_time_,
                                          self.z_move_amplitude_,
                                          self.z_move_amplitude_shift_)
            self.right_leg_move.yaw = w_sin(self.r_ssp_end_time_, self.a_move_period_time_,
                                            self.a_move_phase_shift_ + 2 * pi / self.a_move_period_time_ * self.r_ssp_start_time_ + pi,
                                            -self.a_move_amplitude_, -self.a_move_amplitude_shift_)
        self.pelvis_offset_l = 0
        self.pelvis_offset_r = 0

        self.left_leg_move.roll = 0
        self.left_leg_move.pitch = 0
        self.right_leg_move.roll = 0
        self.right_leg_move.pitch = 0

        right_leg_pose = Pose3D()
        left_leg_pose = Pose3D()
        right_leg_pose.x = self.swap.x + self.right_leg_move.x + self.x_offset_
        right_leg_pose.y = self.swap.y + self.right_leg_move.y - self.y_offset_ / 2
        right_leg_pose.z = self.swap.z + self.right_leg_move.z + self.z_offset_ - self.leg_length
        right_leg_pose.roll = self.swap.roll + self.right_leg_move.roll - self.r_offset_ / 2
        right_leg_pose.pitch = self.swap.pitch + self.right_leg_move.pitch + self.p_offset_
        right_leg_pose.yaw = self.swap.yaw + self.right_leg_move.yaw - self.a_offset_ / 2
        left_leg_pose.x = self.swap.x + self.left_leg_move.x + self.x_offset_
        left_leg_pose.y = self.swap.y + self.left_leg_move.y + self.y_offset_ / 2
        left_leg_pose.z = self.swap.z + self.left_leg_move.z + self.z_offset_ - self.leg_length
        left_leg_pose.roll = self.swap.roll + self.left_leg_move.roll + self.r_offset_ / 2
        left_leg_pose.pitch = self.swap.pitch + self.left_leg_move.pitch + self.p_offset_
        left_leg_pose.yaw = self.swap.yaw + self.left_leg_move.yaw + self.a_offset_ / 2

        right_leg_joints = self.compute_ik_for_right_leg(right_leg_pose)
        left_leg_joints = self.compute_ik_for_left_leg(left_leg_pose)
        return list(right_leg_joints[1:]) + list(left_leg_joints[1:])

    def compute_inverse_kinetics(self, end_point_pose):
        joints_position = [0., 0., 0., 0., 0., 0.]
        ad_pos = [end_point_pose.x, end_point_pose.y, end_point_pose.z]
        ad_ori = p.getQuaternionFromEuler([end_point_pose.roll, end_point_pose.pitch, end_point_pose.yaw])

        # Get knee
        trans_ad = p.getMatrixFromQuaternion(ad_ori)
        vec = np.array([0., 0., 0.])
        vec[0] = trans_ad[0 * 3 + 2] * self.ankle_length + ad_pos[0]
        vec[1] = trans_ad[1 * 3 + 2] * self.ankle_length + ad_pos[1]
        vec[2] = trans_ad[2 * 3 + 2] * self.ankle_length + ad_pos[2]
        rac = np.linalg.norm(vec)
        arc_cos = np.arccos(
            (rac ** 2 - self.thigh_length ** 2 - self.calf_length ** 2) / (2.0 * self.thigh_length * self.calf_length))
        if np.isnan(arc_cos):
            return False
        joints_position[3] = arc_cos

        # Get ankle roll
        da_pos, da_ori = p.invertTransform(position=ad_pos, orientation=ad_ori)
        # trans_da = p.getMatrixFromQuaternion(da_ori)
        k = np.sqrt(da_pos[1] ** 2 + da_pos[2] ** 2)
        l = np.sqrt(da_pos[1] ** 2 + (da_pos[2] - self.ankle_length) ** 2)
        m = (k ** 2 - l ** 2 - self.ankle_length ** 2) / (2 * l * self.ankle_length)
        if m > 1.:
            m = 1.
        elif m < -1.:
            m = -1.
        arc_cos = np.arccos(m)
        if np.isnan(arc_cos):
            return False
        if da_pos[1] < 0.:
            joints_position[5] = -arc_cos
        else:
            joints_position[5] = arc_cos

        # Get hip yaw
        cd_quat = p.getQuaternionFromEuler([joints_position[5], 0, 0])
        dc_pos, dc_ori = p.invertTransform(position=[0, 0, -self.ankle_length], orientation=cd_quat)
        ac_pos, ac_ori = p.multiplyTransforms(ad_pos, ad_ori, dc_pos, dc_ori)
        trans_ac = p.getMatrixFromQuaternion(ac_ori)
        trans_ac = np.array(trans_ac)
        arc_tan = np.arctan2(-trans_ac[0 * 3 + 1], trans_ac[1 * 3 + 1])
        if np.isinf(arc_tan):
            return False
        joints_position[0] = arc_tan

        # Get hip roll
        arc_tan = np.arctan2(trans_ac[2 * 3 + 1],
                             -trans_ac[0 * 3 + 1] * np.sin(joints_position[0]) + trans_ac[1 * 3 + 1] * np.cos(
                                 joints_position[0]))
        if np.isinf(arc_tan):
            return False
        joints_position[1] = arc_tan

        # Get hip pitch and ankle pitch
        arc_tan = np.arctan2(
            trans_ac[0 * 3 + 2] * np.cos(joints_position[0]) + trans_ac[1 * 3 + 2] * np.sin(joints_position[0]),
            trans_ac[0 * 3 + 0] * np.cos(joints_position[0]) + trans_ac[1 * 3 + 0] * np.sin(joints_position[0]))
        if np.isinf(arc_tan):
            return False
        theta = arc_tan
        k = np.sin(joints_position[3]) * self.calf_length
        l = -self.thigh_length - np.cos(joints_position[3]) * self.calf_length
        m = np.cos(joints_position[0]) * vec[0] + np.sin(joints_position[0]) * vec[1]
        n = np.cos(joints_position[1]) * vec[2] + np.sin(joints_position[0]) * np.sin(joints_position[1]) * vec[
            0] - np.cos(joints_position[0]) * np.sin(joints_position[1]) * vec[1]
        s = (k * n + l * m) / (k ** 2 + l ** 2)
        c = (n - k * s) / l
        arc_tan = np.arctan2(s, c)
        if np.isinf(arc_tan):
            return False
        joints_position[2] = arc_tan
        joints_position[4] = theta - joints_position[3] - joints_position[2]
        return joints_position

    def compute_ik_for_right_leg(self, right_leg_pose):
        right_joint_positions = self.compute_inverse_kinetics(right_leg_pose)
        right_joint_positions = right_joint_positions * np.array([1, 1, 1, 1, -1, -1])
        return right_joint_positions

    def compute_ik_for_left_leg(self, left_leg_pose):
        left_joint_positions = self.compute_inverse_kinetics(left_leg_pose)
        left_joint_positions = left_joint_positions * np.array([1, 1, -1, -1, 1, -1])
        return left_joint_positions

    def update_time_param(self):
        self.period_time_ = self.walking_param_.period_time
        self.dsp_ratio_ = self.walking_param_.dsp_ratio
        self.ssp_ratio_ = 1 - self.dsp_ratio_

        self.x_swap_period_time_ = self.period_time_ / 2
        self.x_move_period_time_ = self.period_time_ * self.ssp_ratio_
        self.y_swap_period_time_ = self.period_time_
        self.y_move_period_time_ = self.period_time_ * self.ssp_ratio_
        self.z_swap_period_time_ = self.period_time_ / 2
        self.z_move_period_time_ = self.period_time_ * self.ssp_ratio_ / 2
        self.a_move_period_time_ = self.period_time_ * self.ssp_ratio_

        self.ssp_time_ = self.period_time_ * self.ssp_ratio_
        self.l_ssp_start_time_ = (1 - self.ssp_ratio_) * self.period_time_ / 4
        self.l_ssp_end_time_ = (1 + self.ssp_ratio_) * self.period_time_ / 4
        self.r_ssp_start_time_ = (3 - self.ssp_ratio_) * self.period_time_ / 4
        self.r_ssp_end_time_ = (3 + self.ssp_ratio_) * self.period_time_ / 4

        self.phase1_time_ = (self.l_ssp_start_time_ + self.l_ssp_end_time_) / 2
        self.phase2_time_ = (self.l_ssp_end_time_ + self.r_ssp_start_time_) / 2
        self.phase3_time_ = (self.r_ssp_start_time_ + self.r_ssp_end_time_) / 2

        self.pelvis_offset_ = self.walking_param_.pelvis_offset
        self.pelvis_swing_ = self.pelvis_offset_ * 0.35

    def update_move_param(self):
        self.x_move_amplitude_ = self.walking_param_.x_move_amplitude
        self.x_swap_amplitude_ = self.walking_param_.x_move_amplitude * self.walking_param_.step_fb_ratio

        if self.previous_x_move_amplitude_ == 0:
            self.x_move_amplitude_ *= 0.5
            self.x_swap_amplitude_ *= 0.5

        self.y_move_amplitude_ = self.walking_param_.y_move_amplitude / 2
        if self.y_move_amplitude_ > 0:
            self.y_move_amplitude_shift_ = self.y_move_amplitude_
        else:
            self.y_move_amplitude_shift_ = -self.y_move_amplitude_
        self.y_swap_amplitude_ = self.walking_param_.y_swap_amplitude + self.y_move_amplitude_shift_ * 0.04

        self.z_move_amplitude_ = self.walking_param_.z_move_amplitude / 2
        self.z_move_amplitude_shift_ = self.z_move_amplitude_ / 2
        self.z_swap_amplitude_ = self.walking_param_.z_swap_amplitude
        self.z_swap_amplitude_shift_ = self.z_swap_amplitude_

        if self.walking_param_.move_aim_on is False:
            self.a_move_amplitude_ = self.walking_param_.angle_move_amplitude / 2
            if self.a_move_amplitude_ > 0:
                self.a_move_amplitude_shift_ = self.a_move_amplitude_
            else:
                self.a_move_amplitude_shift_ = -self.a_move_amplitude_
        else:
            self.a_move_amplitude_ = -self.walking_param_.angle_move_amplitude / 2
            if self.a_move_amplitude_ > 0:
                self.a_move_amplitude_shift_ = -self.a_move_amplitude_
            else:
                self.a_move_amplitude_shift_ = self.a_move_amplitude_

    def update_pose_param(self):
        self.x_offset_ = self.walking_param_.init_x_offset
        self.y_offset_ = self.walking_param_.init_y_offset
        self.z_offset_ = self.walking_param_.init_z_offset
        self.r_offset_ = self.walking_param_.init_roll_offset
        self.p_offset_ = self.walking_param_.init_pitch_offset
        self.a_offset_ = self.walking_param_.init_yaw_offset
        self.hit_pitch_offset_ = self.walking_param_.hip_pitch_offset

    def init_walking_param(self, **kwargs):
        self.walking_param_.init_params(**kwargs)

    def update_walking_param(self, **kwargs):
        self.walking_param_.update_params(**kwargs)


if __name__ == "__main__":
    walking_param = {}
    walkingModule = WalkingModule()
    t = 0
    walkingModule.update_walking_param(init_x_offset=-0.01,
                                       init_y_offset=0.01,
                                       init_z_offset=0.025,
                                       init_roll_offset=0.0,
                                       init_pitch_offset=0.0,
                                       init_yaw_offset=0.0,
                                       period_time=0.933,
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
                                       y_swap_amplitude=0.015,
                                       z_swap_amplitude=0.01,
                                       arm_swing_gain=0.20,
                                       pelvis_offset=0.0872664600611,
                                       hip_pitch_offset=0.0)
    rl_0 = []
    rl_1 = []
    rl_2 = []
    rl_3 = []
    rl_4 = []
    rl_5 = []
    time = []
    period = 0
    while period < 5:
        if t >= 0.933:
            t = 0.0
            period += 1
        t += 0.008
        joint_positions = walkingModule.compute_motor_angles(t, x_move_amplitude=0.03)
        time.append(t + period * 0.933)
        # rl_1.append(ll[1])
        # rl_2.append(ll[2])
        # rl_3.append(ll[3])
        # rl_4.append(ll[4])
        # rl_5.append(ll[5])
    plt.plot(time, rl_1, label='rl1')
    plt.plot(time, rl_2, label='rl2')
    plt.plot(time, rl_3, label='rl3')
    plt.plot(time, rl_4, label='rl4')
    plt.plot(time, rl_5, label='rl5')

    print(walkingModule.x_move_amplitude_)
    plt.legend()
    plt.show()
