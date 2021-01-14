import gym
import ballbeam_gym

# pass env arguments as kwargs
kwargs = {'timestep': 0.05,
          'beam_length': 1.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'max_timesteps': 200,
          'action_mode': 'continuous'}

# create env
env = gym.make('BallBeamThrow-v0', **kwargs)


# simulate 1000 steps
for i in range(1000):
    # control theta with a PID controller
    env.render()
    obs, reward, done, info = env.step(0.1)

    if done:
        env.reset()
