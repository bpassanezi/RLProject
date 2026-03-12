import os
import sys
from experiment_setup import run_experiment

# SETTINGS
TEST_MODE = False  # Set to False for long training sessions

# --- 1. SYSTEM CONFIG ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo"
if os.path.join(os.environ['SUMO_HOME'], 'tools') not in sys.path:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# --- 2. RUN THE COMPARISON ---

# 2.1 Run Adaptive Experiments (using the random routes generator)
# This will regenerate traffic patterns every episode to prevent overfitting
run_experiment("DQN", "nets/single-intersection-random.rou.xml", "adaptive_random", use_random_routes=True, TEST_MODE=TEST_MODE)
run_experiment("SARSA", "nets/single-intersection-random.rou.xml", "adaptive_random", use_random_routes=True, TEST_MODE=TEST_MODE)

# 2.2 Run Stationary Experiments (using static baseline routes)
# These files should exist in the nets/ directory
run_experiment("DQN", "nets/single-intersection-stationary.rou.xml", "stationary_static", use_random_routes=False, TEST_MODE=TEST_MODE)
run_experiment("SARSA", "nets/single-intersection-stationary.rou.xml", "stationary_static", use_random_routes=False, TEST_MODE=TEST_MODE)