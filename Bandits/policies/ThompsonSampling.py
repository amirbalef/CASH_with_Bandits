import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class ThompsonSampling:
    def __init__(self, narms, T=200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.random_init = False

        self.prior_alpha = 1.0
        self.prior_beta = 0.5

    def policy_func(self, seq):
        mean = np.mean(seq)
        return np.random.normal(mean, 1/len(seq))

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms == 0):
            if self.random_init:
                self.selected_arm = np.random.choice(np.where(pulled_arms == 0)[0])
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            policy = np.zeros(self.narms)
            for i in range(self.narms):
                reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[:, 0]
                policy[i] = self.policy_func(reward_seq)
            self.selected_arm = np.argmax(policy)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm is None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )
        self.t += 1