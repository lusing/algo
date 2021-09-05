import gym
import numpy as np
env = gym.make('Blackjack-v0')
observation = env.reset()
print(observation)

# 用随机策略玩一个回合

while True:
    print('Player={}'.format(env.player))
    print(env.dealer)
    action = np.random.choice(env.action_space.n)
    print('action={}'.format(action))
    observation, reward, done, _ = env.step(action)
    print(observation)
    print(reward)
    print(done)
    if done:
        break


# 从观测到状态

def ob2state(observation):
    return(observation[0],observation[1],observation[2])


# 同策回合更新策略评估
def evaluation_action_monte_carlo(env, policy, episode=500000):
    q = np.zeros_like(policy)
    
