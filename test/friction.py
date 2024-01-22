import pybullet as p
import time
import pybullet_data
import numpy as np
from env.walkingmodule import WalkingModule

# import rospy
# from std_msgs.msg import Float64

# rospy.init_node('pybullet')
# joint_state_pub = rospy.Publisher('pybullet/joint_state_0', Float64, queue_size=10)
# plugin = pkgutil.get_loader('tinyRendererPlugin')
_physics_client_id = p.connect(p.GUI)

# p.loadPlugin()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.18])
p.setGravity(0, 0, -9.8)
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("/home/zhou/exhx5_pybullet/exhx5_w_b_16.urdf", [0, 0, 0], start_orientation)
joints_num = p.getNumJoints(bodyUniqueId=robot)
joint_indices = range(joints_num)
use_joint_indices = []
pd = p.loadPlugin("pdControlPlugin")
controlMode = p.POSITION_CONTROL
for j in joint_indices:
    info = p.getJointInfo(robot, j, physicsClientId=_physics_client_id)
    joint_name = info[1].decode("ascii")
    if info[2] != p.JOINT_REVOLUTE:
        print(f"jointName:{joint_name}")
        continue
    use_joint_indices.append(j)
    lower, upper = (info[8], info[9])
    print(f"jointName:{joint_name}, posLow:{lower}, posHigh:{upper}")
maxControlForces = np.ones_like(use_joint_indices) * 1
print(f"useJointIndices:{use_joint_indices}")
endEffectorIndex = [6, 12]
ikSolver = 0
trailDuration = 15
t = 0
initalPosition = np.array([-0.0429, -0.5874, 1.3598, 0.7725, -0.0429, 0.0429, 0.5874, -1.3598, -0.7725, 0.0429])
prevPose = np.array([-0.0429, -0.5874, 1.3598, 0.7725, -0.0429, 0.0429, 0.5874, -1.3598, -0.7725, 0.0429])
hasPrevPose = 0
targetPosition = np.array([-0.0429, -0.5874, 1.3598, 0.7725, -0.0429, 0.0429, 0.5874, -1.3598, -0.7725, 0.0429])
walkingModule = WalkingModule()
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
                                   y_swap_amplitude=0.018,
                                   z_swap_amplitude=0.006,
                                   arm_swing_gain=0.20,
                                   pelvis_offset=5.0,
                                   hip_pitch_offset=0.0)
episode = 0
joint = []
joint_target = []
print(p.getJointInfo(robot, 1))
print(p.getJointState(robot, 1))
# p.resetJointState(robot, 1, 0.5)
print(p.getDynamicsInfo(robot, 2))
p.changeDynamics(robot, 6, contactStiffness=80000, contactDamping=10, lateralFriction=0.6)
p.changeDynamics(robot, 12, contactStiffness=80000, contactDamping=10, lateralFriction=0.6)
print(p.getDynamicsInfo(robot, 6))
while 1:
    p.stepSimulation()
    time.sleep(1./240.)
    # p.applyExternalForce(robot, -1, [1, 0, 0], [0, 0, -0.18], flags=p.WORLD_FRAME)
    linear, angular = p.getBaseVelocity(robot)
    joint_positions = p.getJointStates(robot, [2])
    # jointPoses = walkingModule.compute_motor_angles(t)
    jointPoses = initalPosition
    # joint_state_pub.publish(jointPoses[0])
    joint_target.append(jointPoses[0])
    joint.append(joint_positions[0][0])
    # print(joint_positions[0][0])
    pos, ori = p.getBasePositionAndOrientation(robot)
    p.resetDebugVisualizerCamera(cameraDistance=3,
                                 cameraYaw=50,
                                 cameraPitch=-35,
                                 cameraTargetPosition=pos)
    for index, i in enumerate(use_joint_indices):
        p.setJointMotorControl2(bodyIndex=robot,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[index],
                                targetVelocity=0,
                                force=4,
                                positionGain=0.3,
                                velocityGain=1.6)
        # prevPose[i] = jointPoses[index]
    hasPrevPose = 1
    t = t + 1./240.
    if t >= 0.933:
        t = 0
        episode += 1
p.disconnect()
plt.plot(joint)
plt.plot(joint_target)
plt.show()
