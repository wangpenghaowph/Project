import os
from datetime import datetime
import code0227
import pandas as pd
import numpy as np
from scipy.stats import norm,uniform,rv_discrete, binom, gumbel_r
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import math
import pickle
import os
import importlib
importlib.reload(code0227)

file_path = 'state0309.pkl'
pars = [5, 1, "normal", "normal", 0.5, 0.01]
mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval = pars
start = 0.6
end = 3.1
num = int((end - start) / 0.1) + 1  # Calculate the number of steps
unit_costs = np.linspace(start, end, num)
for Tmin in range(1,201):
    Tmax = 1 * Tmin
    for unit_cost in unit_costs:
        code0227.find_static_policy(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
    code0227.save_global_state(file_path)
    print(f"Task1: Finished Tmin={Tmin}")

for Tmin in range(1,201):
    Tmax = 1 * Tmin
    for unit_cost in unit_costs:
        code0227.best_initial_inventory(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
    code0227.save_global_state(file_path)
    print(f"Task2: Finished Tmin={Tmin}")

for Tmin in range(1,201):
    Tmax = 2 * Tmin
    for unit_cost in unit_costs:
        code0227.find_static_policy(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
        print(f"(best sp inv, best price, profit)={code0227.global_sp[(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)]}")
    code0227.save_global_state(file_path)
    print(f"Task3: Finished Tmin={Tmin}")

for Tmin in range(1,201):
    Tmax = 2 * Tmin
    for unit_cost in unit_costs:
        code0227.best_initial_inventory(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
        print(f"(best dp inv, profit)={code0227.global_dp_inventory[Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval]}")
    code0227.save_global_state(file_path)
    print(f"Task4: Finished Tmin={Tmin}")