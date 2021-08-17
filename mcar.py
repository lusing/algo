import gym
import numpy as np
env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.action_space)
print(env.action_space.n)

class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, obervation):
        position, velocity = obervation
        lb = min(-0.09*(position+0.25) ** 2 +0.03,
        0.3 * (position + 0.9)**4 + 0.06)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, **args):
        pass

agent = BespokeAgent(env)

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward

env.seed(0)
episode_reward = play_montecarlo(env, agent, render=True)
print ('Award = {}'.format(episode_reward))
env.close()

episode_reward = [play_montecarlo(env, agent) for _ in range(100)]
print('Average episode={}'.format(np.mean(episode_reward)))
