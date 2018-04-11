# http://blog.ironhead.ninja/2016/09/08/openai-cartpole.html
# https://gym.openai.com/evaluations/eval_2eQdhHWDTHOU1XxThzj76A
# https://gist.github.com/jkarnows/522c2d6000e519482b6deb825d17b34b
import tensorflow as tf
import numpy as np
import gym
from ml_helper import output_folder
from gym import wrappers, logger

logger.set_level(logger.INFO)
env = gym.make('CartPole-v1')

env = wrappers.Monitor(env, directory=output_folder('cartpole-'), force=True)

for _ in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
