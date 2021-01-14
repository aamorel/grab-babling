import pyquaternion as pyq
import random

a = []
for i in range(4):
    a.append(random.random() * 2 - 1)

quat = pyq.Quaternion(a)
print(quat)
quat_normalised = quat.normalised
print(quat_normalised)
print(quat_normalised[1])
