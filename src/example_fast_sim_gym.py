import gym
import gym_fastsim  # actually, it is used when making the environment (must me imported)
import time

display = True

env = gym.make('FastsimSimpleNavigation-v0')
env.reset()
action = [10, 11]

if(display):
    env.enable_display()

then = time.time()

for i in range(2000):
    env.render()
    o, r, eo, info = env.step(action)
    o[0] = o[0] / env.maxSensorRange
    print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r,
          info["dist_obj"], str(info["robot_pos"]), str(eo)))
    if(display):
        time.sleep(0.01)
    if eo:
        break

now = time.time()

print("%d timesteps took %f seconds" % (i, now - then))

env.close()
