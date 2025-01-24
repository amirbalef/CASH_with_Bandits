import numpy as np 
from sklearn.preprocessing import MinMaxScaler

# Sequential Halving : https://proceedings.mlr.press/v28/karnin13.pdf


class Successive_Halving():
    def __init__(self, narms, T = 200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.random_init = False
        self.T =  T

        self.eta = 2.0 #halving rate
        self.rounds = np.log2(self.narms) - 1 
        self.on_arms = np.ones(self.narms, dtype=int)
        self.r = self.T // (np.log2(self.narms) * np.sum(self.on_arms))
        

    def halving(self):
        n_i = int(np.ceil(sum(self.on_arms) / self.eta)) # no. of arms to keep after pruning
        max_rewards = np.zeros(self.narms)
        for x in range(self.narms):
            indx = (np.asarray(self.pulled_arms) == x)        
            max_rewards[x] =  np.mean(self.rewards[indx])
        best_arms = np.argsort(-max_rewards)
        self.on_arms[best_arms[n_i:] ] = 0
        if(sum(self.on_arms)>=1):
            self.r = self.T // (np.log2(self.narms) * np.sum(self.on_arms))

    def play(self, context=None):
        if sum(self.on_arms) > 1:
            self.selected_arm =  np.argwhere(self.on_arms > 0)[(self.t -1)%sum(self.on_arms)][0]
        else:
            self.selected_arm = np.argmax(self.on_arms)
        return self.selected_arm
    

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm 
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))

        if(self.t % self.r == 0 and sum(self.on_arms)>1):
            self.halving()
        self.t += 1

        #print(self.t, self.r, self.on_arms)
        

