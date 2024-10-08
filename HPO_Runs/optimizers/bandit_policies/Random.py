import numpy as np 

class Random():
    def __init__(self, narms, T =200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.selected_arm = 0
        self.intial_steps = 1
        self.random_init = False

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
        else:
            self.selected_arm = np.random.choice(np.arange(self.narms))
        return self.selected_arm
    

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm 
        self.pulled_arms.append(arm)
        self.t += 1
