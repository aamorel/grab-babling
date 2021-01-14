import numpy as np
import random
import struct


def binary(num):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


a = random.uniform(-1, 1)
print(a)
print(binary(a))
name = 'runs/test.npy'
np.save(name, np.array(a), allow_pickle=True)

b = np.load(name)
print(b)
print(binary(b))
