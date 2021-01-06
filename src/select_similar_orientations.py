import pybullet as p
import pybullet_data
import math
import time
import keyboard
import numpy as np

pi = math.pi

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, 0)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 2]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)


angle_1_id = p.addUserDebugParameter("angle 1", -pi, pi, 0)
angle_2_id = p.addUserDebugParameter("angle 2", -pi, pi, 0)
angle_3_id = p.addUserDebugParameter("angle 3", -pi, pi, 0)
count_save = 0

for i in range(10000):
    p.stepSimulation()

    angle_1 = p.readUserDebugParameter(angle_1_id)
    angle_2 = p.readUserDebugParameter(angle_2_id)
    angle_3 = p.readUserDebugParameter(angle_3_id)

    quat = p.getQuaternionFromEuler([angle_1, angle_2, angle_3])
    p.resetBasePositionAndOrientation(boxId, cubeStartPos, quat)

    time.sleep(2)
    if keyboard.read_key() == 's':
        print('saving orientation', quat)
        quat = np.array(quat)
        file_name = 'orientation_' + str(count_save)
        np.save(file_name, quat)
        count_save += 1
