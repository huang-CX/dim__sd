import rospy
import time
import pybullet

from std_srvs.srv import Empty
from std_msgs.msg import Float64, String
from gazebo_msgs.srv import SetModelConfiguration
from op3_walking_module_msgs.msg import WalkingParam

from env.gazebo_env.robot_state import RobotState
from env.gazebo_env.rotation import *

# robot state
robot_state = RobotState()
walking_para = WalkingParam()
# publish the action to corresponding topic
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


class ROBOT(object):

    def __init__(self):
        self.rate = rospy.Rate(125)

    def reset_action(self):
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

    def take_action(self, trj_para):
        walking_para.x_move_amplitude = trj_para[0]
        walking_para.z_move_amplitude = trj_para[1]
        walking_para.y_swap_amplitude = trj_para[2]
        walking_para.z_swap_amplitude = trj_para[3]
        walking_para.dsp_ratio = trj_para[4]
        # walking_para.period_time = trj_para[5]
        apply_para.publish(walking_para)
        # sleep_time = self.compute_sleep(trj_para[5], trj_para[4])
        wait_num = int(0.933 / 4 / 0.008)
        t = rospy.Time.now()
        for _ in range(wait_num):
            # time = rospy.Time.now()
            self.rate.sleep()
            # print(time)
        print(rospy.Time.now() - t)

    def compute_sleep(self, period_time, dsp_ratio):
        return period_time / 4

    def compute_sleep_6(self, period_time, dsp_ratio):

        pass

    def reset_walking_para(self):
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
        print("reset")
        self.reset_action()
        self.reset_walking_para()
        apply_para.publish(walking_para)
        send_command.publish("start")
        time.sleep(0.5)
        send_command.publish("stop")
        time.sleep(0.5)

    def compute_rew(self, refer_t):
        # step length
        step_vector = [(robot_state.body_x - robot_state.last_body_x), (robot_state.body_y - robot_state.last_body_y)]
        # step_vector = [(robot_state.body_x - robot_state.last_body_x)]
        step_vector = np.asarray(step_vector)
        step_len_x = np.linalg.norm(step_vector[0])
        step_len = np.linalg.norm(step_vector)
        # print ("step_len:", step_len)
        # the angle of normal vector
        euler = pybullet.getEulerFromQuaternion(robot_state.last_orientation)
        cos_a = np.dot(euler[0:2], step_vector) / (np.linalg.norm(euler[0:2]) * np.linalg.norm(step_vector))
        if np.isnan(cos_a):
            cos_a = 1.0
        # cos_a, euler = cos_angle(robot_state.last_orientation, step_vector)
        if step_len_x < 0.01:
            robot_state.count_of_motionless += 1
        else:
            robot_state.count_of_motionless = 0

        if cos_a > 0 and (robot_state.body_x - robot_state.last_body_x) > 0:
            step_reward = min(step_len * 100 * cos_a, 3)
        else:
            step_reward = -min(abs(step_len * 100 * cos_a), 3)

        # velocity_reward & effort reward
        effort_reward = 0.0
        effort_reward = 3.0 - 0.25 * np.sum(abs(np.asarray(robot_state.effort)))
        effort_reward = np.clip(effort_reward, -0.5, 0.5)
        effort_reward = effort_reward

        height_reward = 0.0
        if robot_state.body_height > 0.18:
            height_reward = 1.0
        else:
            height_reward = 1.0 - 100 * abs(0.18 - robot_state.body_height)
        height_reward = 0.2 * height_reward
        orientation_reward_p = 0.0
        orientation_reward_r = 0.0
        orientation_reward_y = 0.0
        if abs(robot_state.euler[0]) < 5:
            orientation_reward_r = 0.5
        else:
            orientation_reward_r = 0.8 - min(abs(robot_state.euler[0]) / 10, 1.5)

        if abs(robot_state.euler[1]) < 5:
            orientation_reward_p = 0.5
        else:
            orientation_reward_p = 0.8 - min(abs(robot_state.euler[1]) / 10, 1.5)

        if abs(robot_state.euler[2]) < 5:
            orientation_reward_y = 0.5
        else:
            orientation_reward_y = 0.8 - min(abs(robot_state.euler[2]) / 10, 1.5)
        orientation_reward = orientation_reward_r + orientation_reward_p + orientation_reward_y
        orientation_reward = 0.5 * orientation_reward

        # falling down
        fall_reward = 0.0
        if robot_state.body_height <= 0.1 or robot_state.body_height >= 0.205:
            fall_reward = -50
            robot_state.done = True
            robot_state.fall = 1
            print("Height_irregular!", robot_state.body_height)
        elif abs(robot_state.euler[1]) >= 20 or abs(robot_state.euler[0]) >= 20 or abs(robot_state.euler[2]) >= 30:
            fall_reward = -50
            robot_state.done = True
            robot_state.fall = 1
            print("Tilt!", robot_state.euler[0], robot_state.euler[1])
        elif robot_state.count_of_motionless > 20:
            robot_state.done = True
            fall_reward = -50
            print("Motionless")
        reward_vec = [step_reward] + [fall_reward] + [effort_reward] + [height_reward] + [orientation_reward]
        reward = step_reward + fall_reward + effort_reward + height_reward + orientation_reward
        # print("height:", height_reward, "orientation:", orientation_reward, "step:", step_reward, "effort:",
        #       effort_reward)

        # UPDATE last robot state
        robot_state.last_body_x = robot_state.body_x
        robot_state.last_body_y = robot_state.body_y
        robot_state.last_body_height = robot_state.body_height
        robot_state.last_orientation = robot_state.orientation
        robot_state.last_time = time.time()
        return reward, reward_vec, robot_state.done

    def state_normalization(self, state):
        state = np.asarray(state)
        state[13:16] = 0.1 * state[13:16]
        state[20:23] = (1.0 / 30) * state[20:23]
        state = state.reshape((1, state.shape[0]))
        return state


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
        robot_state.set_robot_state()

    def callback_odom(self, data):
        robot_state.body_height = data.pose.pose.position.z
        robot_state.body_x = data.pose.pose.position.x
        robot_state.body_y = data.pose.pose.position.y
        robot_state.set_robot_state()

    def callback_r_contact(self, data):
        if len(data.states) >= 10:
            robot_state.r_contact = 1
        else:
            robot_state.r_contact = 0
        robot_state.set_robot_state()

    def callback_l_contact(self, data):
        if len(data.states) >= 10:
            robot_state.l_contact = 1
        else:
            robot_state.l_contact = 0
        robot_state.set_robot_state()

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
        robot_state.set_robot_state()

