
import numpy as np
from math import log,sqrt
from BanditTools import *

class ThompsonSamplingDiscrete:
    def __init__(self, nbArms, maxDiscrete = None):
        self.nbArms = nbArms
        self.actionSpace = list(range(0, nbArms)) if maxDiscrete is None else list(range(0, len(maxDiscrete)))
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
        self.all_arms_played = False
    
    def selectArm(self):
        if not self.all_arms_played:
            for arm in range(self.nbArms):
                if self.nbDraws[arm] < 1:
                    return arm
            self.all_arms_played = True
        
        mu_rewards_est_array = np.zeros(self.nbArms)
        for arm in range(len(self.actionSpace)):
            # Estimate mean (clipped normal)
            mu_rewards_est = self.cumRewards[arm] / self.nbDraws[arm]
            # Draw sample reward
            mu_rewards_est_array[arm] = np.clip(np.random.normal(mu_rewards_est), 0, np.Inf)
        
        selected_action = np.argmax(mu_rewards_est_array)
        return selected_action

    def playArm(self, arm, reward):
        self.t = self.t + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.cumRewards[arm] = self.cumRewards[arm] + reward



