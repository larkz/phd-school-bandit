import numpy as np
from ThompsonSampling import *
from market import *

if __name__ == "__main__":

    ts = ThompsonSamplingDiscrete(30)
    ts.clear()
    
    p = ts.select_action()
    reward_trial = profit_function(p)
    ts.playArm(p, reward_trial)

    

