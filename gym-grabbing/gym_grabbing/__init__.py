from gym.envs.registration import register

register(id='baxter_grasping-v0', entry_point='gym_grabbing.envs:BaxterGrasping', max_episode_steps=5000)

register(id='pepper_grasping-v0', entry_point='gym_grabbing.envs:PepperGrasping',max_episode_steps=2000)

register(id='kuka_grasping-v0', entry_point='gym_grabbing.envs:KukaGrasping',max_episode_steps=2500)

register(id='crustcrawler-v0', entry_point='gym_grabbing.envs:CrustCrawler', max_episode_steps=2500)

register(id='kuka_iiwa_allegro-v0', entry_point='gym_grabbing.envs:Kuka_iiwa_allegro', max_episode_steps=2500)

register(id='ur10_shadow-v0', entry_point='gym_grabbing.envs:UR10_shadow', max_episode_steps=2500)

register(id='franka_panda-v0', entry_point='gym_grabbing.envs:Franka_emika_panda', max_episode_steps=2500)
