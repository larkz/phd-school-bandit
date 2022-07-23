# Demand function
import numpy as np

ELASTICITY = -0.4
MAX_DEMAND = 3

def demand_function(price, max_demand = MAX_DEMAND, e = ELASTICITY):
    return np.clip(max_demand + e*price, 0, np.Inf)

def profit_function(price):
    return price * demand_function(price)
