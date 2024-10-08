import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from scipy import special
from scipy import stats

class MaxSearch_Gaussian:
    def __init__(self, narms, T=200, c= 1.0):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.intial_steps = 1
        self.random_init = False
        self.c = c
        self.T = T

    def ierfc(self, x):
        return -x * special.erfc(x) + np.exp(-x * x) / np.sqrt(np.pi)

    def policy_func(self):
        policy = np.zeros(self.narms)
        for i in range(self.narms):
            reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[
                :, 0
            ]
            n = len(reward_seq)
            if n > 1:
                mu = np.mean(reward_seq)
                sigma =  (np.sum(reward_seq**2) - n * mu**2 )/ (n - 1)
                alpha = self.t**(-self.c**2)
                
                mu_hat = mu + stats.t.ppf(
                    1 - alpha / 2,
                    n - 1) * np.sqrt(sigma / n)

                sigma_hat = (n - 1) * sigma / stats.chi2.ppf(alpha / 2, n - 1)

                policy[i] = np.sqrt(sigma_hat / 2) * self.ierfc(
                    (np.max(reward_seq) - mu_hat) / np.sqrt(2 * sigma_hat)
                )
            else:
                policy[i] = np.inf
        #policy = np.nan_to_num(policy)
        #pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        #print("******\n", self.t, policy, pulled_arms, np.argmax(policy), "\n******")
        return policy

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)

        if np.any(pulled_arms < self.intial_steps):
            if self.random_init:
                self.selected_arm = np.random.choice(
                    np.where(pulled_arms < self.intial_steps)[0]
                )
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            policy = self.policy_func()
            # print(self.t, pulled_arms, policy)
            # print(policy)
            self.selected_arm = np.argmax(policy)
            # print(self.t, self.tau,self.selected_arm, policy, pulled_arms)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )

        self.t += 1


class MaxSearch_SubGaussian:
    def __init__(self, narms, T=200, c=1.0 / np.sqrt(13.613)):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.intial_steps = 1
        self.random_init = False
        self.c = c
        self.T = T

    def policy_func(self):
        policy = np.zeros(self.narms)
        for i in range(self.narms):
            reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[
                :, 0
            ]
            n = len(reward_seq)
            beta = self.c * np.sqrt(np.log(self.t)/n)
            gamma = - beta**2 + 2*np.sqrt(2) * beta
            if(gamma>np.log(2)):
                policy[i] = np.inf
            else:
                mu = np.mean(reward_seq)
                sigma = (np.mean(reward_seq**2) - mu**2)/(2 *(np.log(2) - gamma))
                policy[i] = np.sqrt(2 * np.pi * sigma) * special.erfc(
                    (np.max(reward_seq) - mu) / np.sqrt(2 * sigma)
                )
        #print(policy)
        return policy

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)

        if np.any(pulled_arms < self.intial_steps):
            if self.random_init:
                self.selected_arm = np.random.choice(
                    np.where(pulled_arms < self.intial_steps)[0]
                )
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            policy = self.policy_func()
            # print(self.t, pulled_arms, policy)
            # print(policy)
            self.selected_arm = np.argmax(policy)
            # print(self.t, self.tau,self.selected_arm, policy, pulled_arms)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )

        self.t += 1