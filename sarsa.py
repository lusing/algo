import gym
import numpy as np
 
class SARSAAgent:
    def __init__(self,env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros(env.observation_space.n, env.action_space.n)

    def decide(self,state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action 

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * self.q[next_state, next_action] * (1.0 - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error
    
agent = SARSAAgent(env)


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation,action,reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_reward.append(episode_reward)


class ExpectedSARSAAgent:
    def __init__(self,env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros(env.observation_space.n, env.action_space.n)

    def decide(self,state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action 

    def learn(self, state, action, reward, next_state, done):
        v = (self.q[next_state].sum() * self.epsilon + self.q[next_state].max() * (1.0- self.epsilon))
        u = reward + self.gamma * v * (1.0 - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation,action,reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward
