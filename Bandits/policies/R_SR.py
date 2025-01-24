import numpy as np
from sklearn.preprocessing import MinMaxScaler


class R_SR:
    def __init__(self, narms, window_size=0.25, T=200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.random_init = False
        self.T = T
        self.window_size = window_size

        self.round = 1
        self.on_arms = np.ones(self.narms, dtype=int)
        self.r = (self.N_round(self.round) - self.N_round(self.round - 1) + 1) * sum( self.on_arms )

    def N_round(self, j):
        log_K = 0.5 + np.sum([1 / i for i in range(2, self.narms + 1)])
        return np.ceil((1 / log_K) * (self.T - self.narms)/(self.narms +1 -j))

    def play(self, context=None):
        if(sum(self.on_arms)>1):
            self.selected_arm = np.argwhere(self.on_arms > 0)[ (int(self.r) - 1) % sum(self.on_arms) ][0]
        else:
            self.selected_arm = np.argmax(self.on_arms)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )

        if(self.round < self.narms):
            self.r = self.r -1
            if self.r <= 0:
                means = np.full(self.narms, np.inf)
                for i, state_is_on in enumerate(self.on_arms):
                    if state_is_on:
                        indx = np.asarray(self.pulled_arms) == i
                        reward_seq = np.maximum.accumulate(self.rewards[indx])[:, 0]
                        # reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[
                        #     :, 0
                        # ]
                        n = len(reward_seq)
                        h = int(np.floor(n * self.window_size))
                        means[i] = np.mean(reward_seq[:-(h+1)]) if h > 0 else 0

                selected_arm_to_reject = np.argmin(means)
                self.on_arms[selected_arm_to_reject] = 0
                self.round = self.round + 1
                self.r = (self.N_round(self.round) - self.N_round(self.round - 1) + 1) * sum( self.on_arms )
        self.t += 1

        # print(self.t, self.r, self.on_arms)
