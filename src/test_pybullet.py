import pybullet as p
import time
import pybullet_data
import qibullet as q


physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

pepper = q.PepperVirtual()

pepper.loadRobot(translation=[0, 0, 0],
                 quaternion=[0, 0, 0, 1],
                 physicsClientId=physicsClient)  # experimentation

pepper_id = pepper.getRobotModel()

num_joints = p.getNumJoints(pepper_id)
info_joint = p.getJointInfo(pepper_id, 0)
print(num_joints)
print(info_joint)

print(pepper.link_dict)
# for i in range(10000):
#     p.stepSimulation()
#     time.sleep(1. / 240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(pepper_id)
# print(cubePos, cubeOrn)
# p.disconnect()
