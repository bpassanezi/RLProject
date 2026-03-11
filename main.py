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
from experiment_setup import run_experiment


TEST_MODE = False
# --- 1. SYSTEM CONFIG ---
os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo"
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# --- 3. RUN THE COMPARISON ---
# Run DQN and SARSA on the Adaptive Route
run_experiment("DQN", "nets/single-intersection-gen.rou.xml", "adaptative", TEST_MODE=TEST_MODE)
run_experiment("SARSA", "nets/single-intersection-gen.rou.xml", "adaptative", TEST_MODE=TEST_MODE)

# Run experiment on the baseline route
run_experiment("DQN", "nets/single-intersection-stationary.rou.xml", "stationary", TEST_MODE=TEST_MODE)
run_experiment("SARSA", "nets/single-intersection-stationary.rou.xml", "stationary", TEST_MODE=TEST_MODE)