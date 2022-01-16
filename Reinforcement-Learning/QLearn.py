import gym
import numpy as np


class QLearn:

    def __init__(self, qt=None, debug=True) -> None:
        self.env = gym.make("Taxi-v3")
        self.qtable = np.zeros([self.env.observation_space.n, self.env.action_space.n]) if qt is None else qt
        self.debug = debug

        # logs of simulations for training
        self.simulation_training_log = [(0, 0)] # epochs, penalties

    def train(self, alpha=None, gamma=None, expl=None, epochs=None):
        self.EPOCHS = 1000000 if epochs is None else epochs
        self.learning_rate_alpha = 0.1 if alpha is None else alpha
        self.discount_factior_gamma = 0.2 if gamma is None else gamma
        self.exploration_coeff = 0.3 if expl is None else expl

        if self.debug: print(f"Training initiated")
        observation = self.env.reset()
        reward = 0
        penalties = 0
        for epoch in range(self.EPOCHS): # while having epochs
            #env.render()

            if np.random.uniform(0, 1) < self.exploration_coeff: #explore
                action = self.env.action_space.sample()
            else:                                           #exploit
                action = np.argmax(self.qtable[observation])


            new_observation, reward, done, info = self.env.step(action) #make a step with a new policy

            if reward == -10:
                penalties += 1

            current_qval = self.qtable[observation][action] #current
            potentially_best = np.max(self.qtable[new_observation]) #best outcome after taking a step

            # QLearning model update of Qtable
            self.qtable[observation][action] = (1 - self.learning_rate_alpha) * current_qval + self.learning_rate_alpha * (reward + self.discount_factior_gamma * potentially_best)

            observation = new_observation


            if done:
                self.simulation_training_log.append((epoch, penalties))
                observation = self.env.reset()
                reward = 0
                penalties = 0

                if len(self.simulation_training_log) % 1000 == 0 and self.debug:
                    print(f"Simulation: {len(self.simulation_training_log)} Epochs: {self.simulation_training_log[-1][0]} Penalties: {self.simulation_training_log[-1][1]}")

    def __del__(self):
        self.env.close()


if __name__ == "__main__":
    qmodel = QLearn()
    qmodel.train()
    qmodel.train()
