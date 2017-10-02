# http://blog.ironhead.ninja/2016/09/08/openai-cartpole.html
# https://gym.openai.com/evaluations/eval_2eQdhHWDTHOU1XxThzj76A
# https://gist.github.com/jkarnows/522c2d6000e519482b6deb825d17b34b
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
