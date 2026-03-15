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
def run_experiment(algo_type, route_file, experiment_type, use_random_routes=False, is_baseline=False, TEST_MODE=False):
    
    exp_suffix = f"{experiment_type}_baseline" if is_baseline else experiment_type
    TOTAL_STEPS = 3600 if (TEST_MODE or is_baseline) else 18000
    log_dir = f"./logs/{algo_type}/"
    os.makedirs(log_dir, exist_ok=True)
    
    if use_random_routes:
        generate_randomized_routes(route_file)

    env = gym.make('sumo-rl-v0', 
                   net_file='./nets/single-intersection.net.xml', 
                   route_file=route_file, 
                   use_gui=False, 
                   delta_time=5,
                   num_seconds=3600, 
                   add_system_info=True)
    
    env = Monitor(env, log_dir)
    
    # Standard model creation (we need it even for baseline to satisfy the callback)
    if algo_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
    else:
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0007)

    callback = RLTrafficCallback(algo_name=algo_type, 
                               experiment_type=exp_suffix, 
                               use_random_routes=use_random_routes, 
                               route_file=route_file)
    
    # Link the model to the callback manually
    callback.init_callback(model)

    if is_baseline:
        print(f"\nSTARTING FIXED-TIME BASELINE: {route_file}")
        obs, _ = env.reset()
        env.unwrapped.sumo.trafficlight.setProgram('t', '0')
        callback.on_training_start(locals(), globals())
        
        # Note: action, reward, terminated, truncated must exist in this scope
        action = 0 
        for step in range(0, TOTAL_STEPS, 5):
            obs, reward, terminated, truncated, info = env.step(action) 
            
            # This passes the local variables (obs, reward, etc.) to the callback
            callback.on_step() 
            
            if terminated or truncated:
                obs, _ = env.reset() # Reset for next baseline episode if needed
                
        callback.on_training_end()
        env.close()
        print("BASELINE COMPLETED.")
    else:
        print(f"\nSTARTING TRAINING: {algo_type} on {route_file}")
        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
        model.save(f"models/model_{algo_type}_{experiment_type}")