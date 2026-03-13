import os
import sys
import gymnasium as gym
import sumo_rl
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl_traffic_callback import RLTrafficCallback
from routes_generator import generate_randomized_routes


# --- 1. SYSTEM CONFIG ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo"
if os.path.join(os.environ['SUMO_HOME'], 'tools') not in sys.path:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# --- 2. THE SIMULATION RUNNER ---
def run_experiment(algo_type, route_file, experiment_type, use_random_routes=False, TEST_MODE=False):

    # total number of steps the agent take, each step is one decision
    TOTAL_STEPS = 3600 if TEST_MODE else 18000

    log_dir = f"./logs/{algo_type}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # If using random routes, generate the first version before environment creation
    if use_random_routes:
        generate_randomized_routes(route_file)

    env = gym.make('sumo-rl-v0', 
                   net_file='./nets/single-intersection.net.xml', 
                   route_file=route_file, 
                   use_gui=False, 
                   delta_time=5,
                   num_seconds=3600, # time taken for 1 simulation
                   add_system_info=True)
    
    env = Monitor(env, log_dir)
    callback = RLTrafficCallback(algo_name=algo_type, 
                               experiment_type=experiment_type, 
                               use_random_routes=use_random_routes, 
                               route_file=route_file)

    if algo_type == "DQN":
        # Off-policy (Stationary specialist)
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000)
    else:
        # A2C / SARSA (On-policy - Adaptive specialist)
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0007)

    print(f"\nSTARTING TRAINING: {algo_type} on {route_file} (Randomized: {use_random_routes})")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    model.save(f"models/model_{algo_type}_{experiment_type}")
