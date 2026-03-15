import os
import sys
from experiment_setup import run_experiment

# SETTINGS
TEST_MODE = True  # Set to False for long training sessions

# --- 1. SYSTEM CONFIG ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo"
if os.path.join(os.environ['SUMO_HOME'], 'tools') not in sys.path:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# # 2.1 Baseline Comparison (Run 5 episodes with random weights, no training)
# print("\n--- STAGE 1: BASELINE EVALUATION ---")
# run_experiment("DQN", "nets/single-intersection-high-traffic.rou.xml", "high_traffic", use_random_routes=False, is_baseline=True, TEST_MODE=TEST_MODE)
# run_experiment("DQN", "nets/single-intersection-low-traffic.rou.xml", "low_traffic", use_random_routes=False, is_baseline=True, TEST_MODE=TEST_MODE)

# print("\n--- STAGE 2: STATIONARY TRAFFIC DQN ---")
# run_experiment("DQN", "nets/single-intersection-high-traffic.rou.xml", "high_traffic", use_random_routes=False, is_baseline=False, TEST_MODE=TEST_MODE)
# run_experiment("DQN", "nets/single-intersection-low-traffic.rou.xml", "low_traffic", use_random_routes=False, is_baseline=False, TEST_MODE=TEST_MODE)


# 2.2 Adaptive Experiments (Standard training)
print("\n--- STAGE 3: ADAPTATIVE TRAFFIC ---")
run_experiment("DQN", "nets/single-intersection-random.rou.xml", "adaptative_random", use_random_routes=True, TEST_MODE=TEST_MODE)
run_experiment("A2C", "nets/single-intersection-random.rou.xml", "adaptative_random", use_random_routes=True, TEST_MODE=TEST_MODE)

