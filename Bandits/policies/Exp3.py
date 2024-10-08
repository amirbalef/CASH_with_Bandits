import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class Exp3():
    def __init__(self, narms, gamma=0.1, T=None):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.random_init = False

        self.gamma = gamma
        self.p_t = np.zeros(self.narms)
        self.w_t = np.ones(self.narms)

    def calculate_p(self,w_t):
        return (1 - self.gamma) * (w_t / np.sum(w_t)) + (self.gamma / self.narms)
	
    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms==0):
            self.selected_arm =  (self.t -1)%self.narms
        else:
            self.p_t  = self.calculate_p(self.w_t)
            self.selected_arm =  np.random.choice(self.narms, p=self.p_t)
        return self.selected_arm
	

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))

        self.w_t = np.ones(self.narms)
        for i in range(len(self.pulled_arms)):
            self.p_t  = self.calculate_p(self.w_t)
            xhat_t = np.zeros(self.narms)
            xhat_t[self.pulled_arms[i]] = self.rewards[i] / self.p_t[self.pulled_arms[i]]
            self.w_t *= np.exp(self.gamma * xhat_t / self.narms)
        self.t += 1