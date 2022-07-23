import numpy as np
from ThompsonSampling import *
from market import *
import matplotlib.pyplot as plt

T = 2000
N_ARMS = 30
if __name__ == "__main__":

    ts = ThompsonSamplingDiscrete(N_ARMS)
    ts.clear()

    rewards = []
    prices_played = []
    for t in range(T):
        p = ts.selectArm()
        reward_trial = profit_function(p)
        ts.playArm(p, reward_trial)
        print(reward_trial)
        rewards.append(reward_trial)
        prices_played.append(p)

    print("optimal price: " +str( -(MAX_DEMAND/ELASTICITY)/2 ))

    plt.xlabel("Time t")
    plt.ylabel("Epoch Reward")
    plt.plot(rewards)
    plt.savefig("reward_policy.png", dpi=300)
    plt.show()

    plt.xlabel("Time t")
    plt.ylabel("Prices Played")
    plt.plot(prices_played)
    plt.savefig("prices.png", dpi=300)
    plt.show()

