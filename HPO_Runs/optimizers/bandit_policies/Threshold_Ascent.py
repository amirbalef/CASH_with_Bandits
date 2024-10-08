import numpy as np 
from sklearn.preprocessing import MinMaxScaler


class Threshold_Ascent:
    def __init__( self,narms,T=200, s =20, delta =0.1):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.random_init = False
        self.policy = np.zeros(narms)

        self.alpha = np.log(2 * self.narms * T / delta)
        self.threshold = -np.inf
        self.s = s
        self.S = np.zeros(self.narms)

    def Chernoff_Interval(self, mu, n, alpha):
        """
        U function for threshold ascent
        """
        if n == 0:
            return np.inf
        return mu + (alpha + np.sqrt(2*n*mu*alpha+alpha**2))/n

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms 
        else:
            
            if self.t > self.s:
                former_thresh = self.threshold
                self.threshold = self.rewards[np.argsort(self.rewards)][-int(self.s)]
                if self.threshold != former_thresh:
                    for i  in range(self.narms):
                        reward_seq =  np.asarray(self.rewards[ np.asarray(self.pulled_arms) == i ])
                        self.S[i] = reward_seq[reward_seq >= self.threshold].shape[0]
            else:
                self.S = pulled_arms
            Idx = np.array(
                [
                    self.Chernoff_Interval(
                        self.S[k] / pulled_arms[k], pulled_arms[k], self.alpha
                    )
                    for k in range(self.narms)
                ]
            )
            self.selected_arm = np.argmax(Idx)
        return self.selected_arm
    
    
    def update(self, reward, arm = None, context=None):
        if arm is None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.rewards.append(reward)
        self.t += 1

    def update_cost(self, cost, arm = None, context=None):
        if arm is None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))
        self.epsilon = 1/(self.t)
        self.t += 1