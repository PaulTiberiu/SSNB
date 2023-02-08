import gym
import time

env = gym.make('Swimmer-v3')
env.reset()

img = env.render()
time.sleep(10)