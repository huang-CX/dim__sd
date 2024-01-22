import pybullet as p
import time
import pybullet_data
import numpy as np
import sys
sys.path.append("..")
from env.walkingmodule import WalkingModule

# import rospy
# from std_msgs.msg import Float64

# rospy.init_node('pybullet')
# joint_state_pub = rospy.Publisher('pybullet/joint_state_0', Float64, queue_size=10)
# plugin = pkgutil.get_loader('tinyRendererPlugin')
_physics_client_id = p.connect(p.GUI)

# p.loadPlugin()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("pendulum5.urdf", [0, 0, 0])
while 1:
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()