from cgitb import small
from genericpath import samefile
from turtle import pen
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
        self.discount_factior_gamma = 0.6 if gamma is None else gamma
        self.exploration_coeff = 0.1 if expl is None else expl

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

    def evaluate(self, sample=None):
        if sample is None: sample = 1000
        if self.debug: print(f"----Evaluation on-----\nsample: {sample}\nalpha: {self.learning_rate_alpha}\ngamma: {self.discount_factior_gamma}\nexploration: {self.exploration_coeff}")

        min_epoch = 10000000
        max_epoch = 0
        sum_epoch = 0

        min_penal = 10000000
        max_penal = 0
        sum_penal = 0

        for simulation in range(sample):
            observation = self.env.reset()
            epoch, penalties = 0, 0
            done = False

            while not done:
                action = np.argmax(self.qtable[observation])
                observation, reward, done, info = self.env.step(action)

                if(reward == -10): penalties += 1
                epoch += 1

            if min_epoch > epoch: min_epoch = epoch
            if max_epoch < epoch: max_epoch = epoch
            sum_epoch += epoch

            if min_penal > penalties: min_penal = penalties
            if max_penal < penalties: max_penal = penalties
            sum_penal += penalties

        if self.debug:
            print(f"-------Result of sampling {sample} simulations-------\n")
            print(f"Epochs:\n\tmax = {max_epoch}\n\tmin = {min_epoch}\n\tavg = {sum_epoch / sample}")
            print(f"Penalties:\n\tmax = {max_penal}\n\tmin = {min_penal}\n\tavg = {sum_penal / sample}")

        return [(max_epoch, min_epoch, sum_epoch/sample), (max_penal, min_penal, sum_penal/sample)]

    def __del__(self):
        self.env.close()


if __name__ == "__main__":
    qmodel = QLearn()
    qmodel.train()
    qmodel.evaluate(1000)
