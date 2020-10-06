import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#### Explore behavior of the canonical system

# define differential equation
def canonical(y, t, alpha, tau):
    s = y
    dydt = -alpha * s / tau
    return dydt

# define constants
alpha = 1
tau = 1
s0 = 1
nb_of_points = 1000

# define time
t = np.linspace(0, 10, nb_of_points)

# integrate
sol = odeint(canonical, s0, t, args=(alpha, tau))

# plot solution
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set(title='Canonical system behavior', xlabel='Time', ylabel='S', \
#     xlim=(0, 10), ylim=(-2, 2))
# ax.plot(t, sol)
# plt.show()



#### Explore Gaussian basis function

# define Gaussian basis
def gaussian_basis(s, h, c):
    return np.exp(-h * np.power((s - c), 2))

# define constants
c = 0
h = 0.2

# define s
s = np.linspace(-5, 5, nb_of_points)

# compute function
values = gaussian_basis(s, h, c)

# plot function
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set(title='Gaussian basis function', xlabel='S', ylabel='Value', \
#     xlim=(-10, 10), ylim=(-10, 10))
# ax.plot(s, values)
# plt.show()


#### Explore External Force behavior

# define constants (must be of same size)
hs = [100, 2, 0.1]
cs = [-0.5, 0, 5]
ws = [1, -2, 1]

# define behavior of s as canonical behavior
s = sol[:,0]
print(s.shape)

values = []
# loop through all Gausian basis
for h, c in zip(hs, cs):
    # compute behavior and save
    values.append(gaussian_basis(s, h, c))

# constuct the external force behavior
external_force_num = np.zeros(nb_of_points)
external_force_den = np.zeros(nb_of_points)
for w, value in zip(ws, values):
    external_force_num = np.add(external_force_num, w * np.multiply(s, value))
    external_force_den = np.add(external_force_num, value)
external_force = np.divide(external_force_num, external_force_den)


# plot behavior
fig, ax = plt.subplots(figsize=(5, 5))
ax.set(title='External force behavior', xlabel='Time', ylabel='Force', \
    xlim=(0, 10), ylim=(-100, 100))
ax.plot(t, external_force)
plt.show()
