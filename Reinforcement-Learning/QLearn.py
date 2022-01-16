import gym
import numpy as np

env = gym.make("Taxi-v3")

EPOCHS = 1000
qtable = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate_alpha = 0.1
discount_factior_gamma = 0.2
exploration_coeff = 0.3


# logs of simulations
simulation_training_log = []

observation = env.reset()
reward = 0
for _ in range(EPOCHS): # while having epochs
    env.render()

    if np.random.uniform(0, 1) < exploration_coeff: #explore
        action = env.action_space.sample()
    else:                                           #exploit
        action = np.argmax(qtable[observation])


    new_observation, reward, done, info = env.step(action) #make a step with a new policy

    current_qval = qtable[observation][action] #current
    potentially_best = np.max(qtable[new_observation]) #best outcome after taking a step

    # QLearning model update of Qtable
    qtable[observation][action] = (1 - learning_rate_alpha) * current_qval + learning_rate_alpha * (reward + discount_factior_gamma * potentially_best)

    observation = new_observation

    if done:
        observation = env.reset()
        reward = 0
        simulations_count += 1
        print()

env.close()
