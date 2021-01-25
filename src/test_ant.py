import pybullet_envs
import pybullet_envs.gym_locomotion_envs


class DeterministicPybulletAnt(pybullet_envs.gym_locomotion_envs.AntBulletEnv):
    def __init__(self, render=False, random_seed=0):
        self.deterministic_random_seed = random_seed
        pybullet_envs.gym_locomotion_envs.AntBulletEnv.__init__(self, render)

    def reset(self):
        self.seed(self.deterministic_random_seed)
        return pybullet_envs.gym_locomotion_envs.AntBulletEnv.reset(self)


render = False
env = DeterministicPybulletAnt(render=render, random_seed=0)
env.reset()

done = False
count = 0
while not done and count <= 100:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if render:
        env.render()
    count += 1
    
print('end')
