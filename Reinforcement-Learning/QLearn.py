from distutils.log import debug
from math import gamma
import gym
import numpy as np
import matplotlib.pyplot as plt
import csv


class QLearn:

    def __init__(self, qt=None, debug=True) -> None:
        self.env = gym.make("Taxi-v3")
        self.qtable = np.zeros([self.env.observation_space.n, self.env.action_space.n]) if qt is None else qt
        self.debug = debug

        # logs of simulations for training
        self.simulation_training_log = [(0, 0)] # epochs, penalties

    def train(self, alpha=None, gamma=None, expl=None, epochs=None, simulations=None):
        self.EPOCHS = 1000000 if epochs is None else epochs
        self.learning_rate_alpha = 0.1 if alpha is None else alpha
        self.discount_factior_gamma = 0.5 if gamma is None else gamma
        self.exploration_coeff = 0.1 if expl is None else expl

        if self.debug: print(f"Training initiated")
        observation = self.env.reset()
        reward = 0
        penalties = 0
        ep = 0
        stop_condition = True
        epoch = 0
        sim_num = 0
        while(stop_condition): # while having epochs
            #env.render()
            epoch += 1
            ep += 1
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

            if epochs is not None:
                if epoch == epochs: stop_condition = False

            if done:
                self.simulation_training_log.append((ep, penalties))
                observation = self.env.reset()
                reward = 0
                penalties = 0
                ep = 0
                sim_num += 1

                if simulations is not None:
                    if sim_num == simulations: stop_condition = False

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

        return [max_epoch, min_epoch, sum_epoch/sample, max_penal, min_penal, sum_penal/sample]

    def __del__(self):
        self.env.close()


def plotter(model: QLearn, compact_scale = 1):
    epochs = np.array(model.simulation_training_log).T[0]
    penal = np.array(model.simulation_training_log).T[1]
    if compact_scale != 1:
        new_epochs = []
        new_penal = []
        start = 0
        while(start + compact_scale < len(model.simulation_training_log)):
            new_epochs.append(np.mean(epochs[start:(start + compact_scale)]))
            new_penal.append(np.mean(penal[start:(start+compact_scale)]))
            start += compact_scale
        epochs = new_epochs
        penal = new_penal
    simulations = len(epochs)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(f"Training simulations packs by {compact_scale}")
    ax1.set_ylabel("Epochs", color='b')
    ax1.plot(range(simulations), epochs, '-b', label="epochs", markersize=3)
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Penalties", color='r')
    ax2.plot(range(simulations), penal, '-r', label="penalties", markersize=3)
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(label=f"Alpha = {model.learning_rate_alpha} Gamma = {model.discount_factior_gamma} Exploration = {model.exploration_coeff}")
    plt.tight_layout()
    plt.gcf().savefig(f"./Reinforcement-Learning/Charts/Alpha={model.learning_rate_alpha}Gamma={model.discount_factior_gamma}Exploration={model.exploration_coeff}.png", format='png')
    plt.clf()

if __name__ == "__main__":
    alphas = [0.1, 0.5, 0.8]
    gammas = [0.1, 0.5, 0.8]
    explores = [0.1, 0.5, 0.8]
    raports = []

    for a in alphas:
        for g in gammas:
            for e in explores:
                qmodel = QLearn(debug=False)
                qmodel.train(alpha=a, gamma=g, expl=e, simulations=5000)
                raports.append([a, g, e] + qmodel.evaluate(1000))
                plotter(qmodel, 100)

    raports = np.array(raports)
    with open("Raport.csv", 'w+') as handle:
        fieldnames = ['Alpha', 'Gamma', 'Exploration', 'Epoch Min', 'Epoch Max', 'Epoch Avg', 'Penal Min', 'Penal Max', 'Penal Avg']
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for r in raports:
            writer.writerow(r)
