from gym.envs.registration import register

register(id='baxter_grabbing-v0',
         entry_point='gym_baxter_grabbing.envs:Baxter_grabbingEnv',)
