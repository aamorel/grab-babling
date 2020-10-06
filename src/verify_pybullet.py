import pybullet as p
import time
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect() 