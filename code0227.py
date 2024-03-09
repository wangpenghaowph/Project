from matplotlib.pylab import f
import pandas as pd
import numpy as np
from scipy.stats import norm,uniform,rv_discrete, binom, gumbel_r
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pickle
import os
global_memo = {}
global_sp = {}
global_dp_inventory = {}
global_sp_newsvender = {}
def pmf_horizon(t, tmin, tmax, distribution_type):
    if tmin == tmax:
        return 1 if t == tmin else 0
    if distribution_type == "normal":
        mean = (tmax + tmin) / 2
        std = (tmax - mean) / 3  # Assuming 3 sigma rule for normal distribution
        if tmin <= t <= tmax:
            total = norm.cdf(tmax + 0.5, mean, std) - norm.cdf(tmin - 0.5, mean, std)
            return (norm.cdf(t + 0.5, mean, std) - norm.cdf(t - 0.5, mean, std)) / total
        else:
            return 0
    if distribution_type == "uniform":
        return 1 / (tmax - tmin + 1) if tmin <= t <= tmax else 0
    
def conditional_probabilities(t, Tmin, Tmax, distribution_type):
    if t > Tmax:
        return 0, 0
    elif t < Tmin:
        return 0, 1
    P_T_geq_t = np.sum(np.array([pmf_horizon(i, Tmin, Tmax, distribution_type) for i in range(t, Tmax + 1)]))
    P_T_equals_t = pmf_horizon(t, Tmin, Tmax, distribution_type)
    if P_T_geq_t > 0:
        P_T_equals_t_given_geq_t = P_T_equals_t / P_T_geq_t
    else:
        P_T_equals_t_given_geq_t = 0
    return P_T_equals_t_given_geq_t, 1 - P_T_equals_t_given_geq_t
def expected_horizon(Tmin, Tmax, distribution_type):
    if distribution_type == "uniform":
        return (Tmin + Tmax) / 2
    elif distribution_type == "normal":
        return (Tmin + Tmax) / 2
    else:
        raise ValueError("Unsupported distribution type")
def reservation_price_distribution(mean, variance, distribution_type="normal"):
    if distribution_type == "normal":
        return norm(loc=mean, scale=np.sqrt(variance))
    elif distribution_type == 'gumbel':
        scale = np.sqrt(6) * np.sqrt(variance) / np.pi
        loc = mean - np.euler_gamma * scale
        return gumbel_r(loc=loc, scale=scale)
    else:
        raise ValueError("Unsupported distribution type")

def purchase_rate(p, mean, variance, distribution_type):
    dist = reservation_price_distribution(mean, variance, distribution_type)
    return 1 - dist.cdf(p)
# DP joint inventory and pricing
def dynamic_program(curr_inv, t, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval):
    key = (curr_inv, Tmin - t, Tmax - t, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval)
    if key in global_memo:
        return global_memo[key]
    if curr_inv == 0 or t >= Tmax:
        return salvage_value * curr_inv, []
    P_T_equals_t_given_geq_t, P_T_greater_than_t = conditional_probabilities(t, Tmin, Tmax, horizon_distribution_type)
    max_profit = 0
    optimal_price = 0
    std = np.sqrt(variance)
    price_range = np.arange(mean - 2 * std, mean + 2 * std + 1, price_interval)
    for price in price_range:
        selling_profit, _ = dynamic_program(curr_inv - 1, t + 1, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval)
        expected_profit_selling = P_T_greater_than_t * purchase_rate(price, mean, variance, price_distribution_type) * (price + selling_profit)
    
        not_selling_profit, _ = dynamic_program(curr_inv, t + 1, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval)
        expected_profit_not_selling = P_T_greater_than_t * (1 - purchase_rate(price, mean, variance, price_distribution_type)) * not_selling_profit
        
        time_end_profit = P_T_equals_t_given_geq_t * curr_inv * salvage_value
        total_expected_profit = expected_profit_selling + expected_profit_not_selling + time_end_profit
        
        if total_expected_profit > max_profit:
            max_profit = total_expected_profit
            optimal_price = price
    global_memo[key] = (max_profit, optimal_price)
    return max_profit, optimal_price

def best_initial_inventory(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval):
    key = (Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
    if key in global_dp_inventory:
        #print(f"DP: Key Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost} has been stored.")
        return global_dp_inventory[key]
    start_time = datetime.now()
    best_profit = float('-inf')
    best_inventory = 0

    for inv in range(1, Tmax + 1):
        expected_profit, _ = dynamic_program(inv, 0, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval)
        profit_less_cost = expected_profit - (unit_cost * inv)
        if profit_less_cost > best_profit:
            best_profit = profit_less_cost
            best_inventory = inv
    global_dp_inventory[key] = (best_inventory, best_profit)
    end_time = datetime.now()
    #print(f"DP: Key Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost}, has been added with runtime: {end_time - start_time}")
    return best_inventory, best_profit

def newsvender_inv_by_critical_ratio(critical_ratio, Tmin, Tmax, horizon_distribution_type, purchase_prob):
    key = (critical_ratio, Tmin, Tmax, horizon_distribution_type, purchase_prob)
    if key in global_sp_newsvender:
        #print(f'newsvender inventory stored')
        return global_sp_newsvender[key]
    #start_time = datetime.now()
    best_inv = None
    if Tmin == Tmax:
        for inv in range(Tmax + 1):
            if binom.cdf(inv, Tmax, purchase_prob) >= critical_ratio:
                best_inv = inv
                break
    else:
        t_values = np.arange(Tmin, Tmax + 1)
        pmf_values = np.array([pmf_horizon(t, Tmin, Tmax, horizon_distribution_type) for t in t_values])
        cdf = 0
        for inv in range(Tmax + 1):
            binom_pmf = binom.pmf(inv, t_values, purchase_prob)
            cdf += np.sum(pmf_values * binom_pmf)
            # if cdf >= or very close to critical ratio
            if cdf >= critical_ratio - 1e-10:
                best_inv = inv
                break
    # return error if best inventory is not found
    if best_inv is None:
        raise ValueError("Best inventory not found")
    global_sp_newsvender[key] = best_inv
    #end_time = datetime.now()
    #print(f"newsvender inventory: Key Tmin={Tmin},Tmax={Tmax},critical ratio={critical_ratio},purchase probability={purchase_prob} has been added with runtime: {end_time - start_time}")
    return best_inv


def compute_static_profit(price, inventory, mean, variance, Tmin, Tmax, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost):
    if Tmin == Tmax:
        purchase_prob = purchase_rate(price, mean, variance, price_distribution_type)
        # compute the profit if there are x customers have reservation price greater than the static price, and then sum up the profit for all possible x
        profit = 0
        for x in range(Tmin + 1):
            profit += binom.pmf(x, Tmin, purchase_prob) * (min(x, inventory) * price + max(0, inventory - x) * salvage_value - inventory * unit_cost)
        return profit
    else:
        customer_counts = np.arange(Tmin, Tmax + 1)
        pmf_values = [pmf_horizon(t, Tmin, Tmax, horizon_distribution_type) for t in customer_counts]
        purchase_prob = purchase_rate(price, mean, variance, price_distribution_type)
        profit = 0
        for x in range(Tmax + 1):
            binom_pmf = binom.pmf(x, customer_counts, purchase_prob)
            profit += np.sum(pmf_values * binom_pmf) * (min(x, inventory) * price + max(0, inventory - x) * salvage_value - inventory * unit_cost)
        return profit
    
# define a function, named find_best_price_and_profit_given_inventory, to find the best price and profit given the inventory
def find_best_price_and_profit_given_inventory(inventory, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval):
    key = (inventory, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
    if key in global_sp:
        #print(f"SP: Key inventory={inventory}, Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost}, salvage={salvage_value},price distribution=({price_distribution_type},{mean},{variance}) stored.")
        return global_sp[key]
    start_time = datetime.now()
    best_profit = float('-inf')
    std = np.sqrt(variance)
    price_range = np.arange(mean - 2 * std, mean + 2 * std, price_interval)
    for price in price_range:
        profit = compute_static_profit(price, inventory, mean, variance, Tmin, Tmax, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost)
        if profit > best_profit:
            best_profit = profit
            best_price = price
    global_sp[key] = (best_price, best_profit)
    end_time = datetime.now()
    #print(f"SP: Key inventory={inventory}, Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost}, salvage={salvage_value},price distribution=({price_distribution_type},{mean},{variance}), price_interval={price_interval} added with runtime: {end_time - start_time}")
    return best_price, best_profit

def find_static_policy(Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval):
    key = (Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost, price_interval)
    if key in global_sp:
        #print(f"SP: Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost}, salvage={salvage_value},price distribution=({price_distribution_type},{mean},{variance}) stored.")
        return global_sp[key]
    start_time = datetime.now()
    std = np.sqrt(variance)
    price_range = np.arange(mean - 2 * std, mean + 2 * std, price_interval)
    best_profit = float('-inf')
    for price in price_range:
        purchase_prob = purchase_rate(price, mean, variance, price_distribution_type)
        best_inv_for_price = 0
        critical_ratio = (price - unit_cost) / (price - salvage_value)
        best_inv_for_price = newsvender_inv_by_critical_ratio(critical_ratio, Tmin, Tmax, horizon_distribution_type, purchase_prob)
        profit = compute_static_profit(price, best_inv_for_price, mean, variance, Tmin, Tmax, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost)
        if profit > best_profit:
            best_profit = profit
            best_price = price
            best_inv = best_inv_for_price
    global_sp[key] = (best_inv, best_price, best_profit)
    end_time = datetime.now()
    #print(f"SP: Tmin={Tmin},Tmax={Tmax},unit cost={unit_cost}, salvage={salvage_value},price distribution=({price_distribution_type},{mean},{variance}), price_interval={price_interval} added with runtime: {end_time - start_time}")
    return best_inv, best_price, best_profit

def compute_all_methods(memo_list, static_price_list, mean, variance, Tmin, Tmax, price_distribution_type, horizon_distribution_type, unit_cost, salvage_value):
    dp_profits = [] 
    sp_profits = []
    # use data in global memo to compute the profit for each dp profit, memo list=[global_memo, inventory]
    for memo, inv in memo_list:
        profit = dynamic_program(inv, 0, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, 1)[0]
        dp_profits.append(profit)
    # use function compute_static_profit to compute the profit for each static price, static price list=[price, inventory]
    for price, inv in static_price_list:
        profit = compute_static_profit(price, inv, mean, variance, Tmin, Tmax, price_distribution_type, horizon_distribution_type, salvage_value, unit_cost)
        sp_profits.append(profit)
    return dp_profits + sp_profits

def find_best_period_price(mean, variance, price_distribution_type, price_interval, salvage_value):
    best_single_period_price = 0
    best_single_period_reward = float('-inf')
    price_range = np.arange(mean - 2 * np.sqrt(variance), mean + 2 * np.sqrt(variance), price_interval)
    for price in price_range:
        purchase_prob = purchase_rate(price, mean, variance, price_distribution_type)
        single_period_reward = purchase_prob * price + (1 - purchase_prob) * salvage_value
        if single_period_reward > best_single_period_reward:
            best_single_period_reward = single_period_reward
            best_single_period_price = price
    return best_single_period_price

def find_best_per_period_demand_price(inventory, Tmin, Tmax, mean, variance, price_distribution_type, horizon_distribution_type, salvage_value, price_interval):
    exp_horizon = expected_horizon(Tmin, Tmax, horizon_distribution_type)
    per_period_demand = inventory / exp_horizon
    if per_period_demand >= 1:
        return find_best_period_price(mean, variance, price_distribution_type, price_interval, salvage_value)    
    demand_gap = float('inf')
    for price in np.arange(mean - 2 * np.sqrt(variance), mean + 2 * np.sqrt(variance), price_interval):
        purchase_prob = purchase_rate(price, mean, variance, price_distribution_type)
        curr_demand_gap = np.abs(per_period_demand - purchase_prob)
        if curr_demand_gap < demand_gap:
            demand_gap = curr_demand_gap
            best_price = price
    return best_price

def merge_global_state(new_data, existing_data):
    for key in new_data:
        if key in existing_data:
            # Example merge strategy: update existing entries with new values
            existing_data[key].update(new_data[key])
        else:
            # If the key doesn't exist in existing data, add it
            existing_data[key] = new_data[key]
    return existing_data

def save_global_state(file_path):
    # Define your new data structure here, for example:
    new_global_state = {'global_memo': global_memo, 'global_sp': global_sp, 'global_sp_newsvender': global_sp_newsvender, 'global_dp_inventory': global_dp_inventory}
    
    # Load existing data if the file exists
    try:
        with open(file_path, 'rb') as f:
            existing_global_state = pickle.load(f)
    except FileNotFoundError:
        print("No saved state found. Starting with a new file.")
        existing_global_state = {}

    # Merge new data into existing data
    merged_global_state = merge_global_state(new_global_state, existing_global_state)

    # Save the merged data back to the file
    with open(file_path, 'wb') as f:
        pickle.dump(merged_global_state, f)


def load_global_state(file_path):
    try:
        with open(file_path, 'rb') as f:
            global_state = pickle.load(f)
            global global_memo, global_sp, global_sp_newsvender, global_dp_inventory
            global_memo, global_sp, global_sp_newsvender, global_dp_inventory = (global_state['global_memo'], global_state['global_sp'], global_state['global_sp_newsvender'], global_state['global_dp_inventory'])
            print("Global state loaded successfully.")
    except FileNotFoundError:
        print("No saved state found. Starting with empty dictionaries.")

def delete_global_state(file_path):
    global global_memo, global_sp, global_sp_newsvender, global_dp_inventory
    global_memo, global_sp, global_sp_newsvender, global_dp_inventory = ({}, {}, {}, {})
    with open(file_path, 'wb') as f:
        pickle.dump({}, f)
    print("Global state deleted successfully.")