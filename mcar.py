import gym
env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.action_space)
print(env.action_space.n)
