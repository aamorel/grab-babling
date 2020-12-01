import gym

baxter_env = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=False)

for i in range(1000):
    o, r, eo, inf = baxter_env.step([0.2, 0, -0.1, 0.2, 0.2, -0.2, -0.2, 1])
    if i % 30 == 0:
        print(o[2])
        

baxter_env.reset()
print('reset made')

for i in range(1000):
    o, r, eo, inf = baxter_env.step([0.2, 0, -0.1, 0.2, 0.2, -0.2, -0.2, 1])
    if i % 30 == 0:
        print(o[2])

baxter_env.reset()
print('reset made')

for i in range(1000):
    o, r, eo, inf = baxter_env.step([0.2, 0, -0.1, 0.2, 0.2, -0.2, -0.2, 1])
    if i % 30 == 0:
        print(o[2])
