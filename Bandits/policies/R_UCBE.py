import numpy as np
from sklearn.preprocessing import MinMaxScaler


class R_UCBE:
    def __init__(self, narms, alpha=57.12041528623219, window_size=0.25, T=None):
        self.narms = narms
        self.alpha = alpha
        self.window_size = window_size
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.random_init = False
        self.intial_steps = 1
        self.T = T
        self.sigma = 0.05

    def policy_func(self):
        policy = np.zeros(self.narms)
        for i in range(self.narms):
            indx = np.asarray(self.pulled_arms) == i
            reward_seq = np.maximum.accumulate(self.rewards[indx])[:, 0]
            # reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[
            #     :, 0
            # ]
            n = len(reward_seq)
            h = int(np.floor(n * self.window_size)) 
            mu =  0 if h==0  else np.mean(reward_seq[:-(1 + h)])
            mu +=  0 if h==0 else np.sum(
                [
                    (self.T - t + 1) * (reward_seq[t] - reward_seq[t - h]) / h**2
                    for t in range(n - h, n)
                ]
            ) 
            beta =  0 if h==0 else self.sigma * (self.T - n + h - 1)* np.sqrt(self.alpha / h**3)

            policy[i] = mu + beta
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

    def update(self, reward, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.rewards.append(reward)
        self.t += 1

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )
        self.t += 1