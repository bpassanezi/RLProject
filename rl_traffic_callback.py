import os
import sys
import gymnasium as gym
import sumo_rl
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. SYSTEM CONFIG ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo"
if os.path.join(os.environ['SUMO_HOME'], 'tools') not in sys.path:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

class RLTrafficCallback(BaseCallback):
    """
    Custom Logger to capture Value Function, Optimum Policy, 
    and Environment metrics (Queues, Emissions).
    """
    def __init__(self, algo_name, experiment_type, use_random_routes=False, route_file=None, verbose=0):
        super(RLTrafficCallback, self).__init__(verbose)
        self.algo_name = algo_name
        self.experiment_type = experiment_type
        self.use_random_routes = use_random_routes
        self.route_file = route_file
        self.data_log = []
        self.episode_count = 1

    def _on_step(self) -> bool:
        # Log every 10 steps for detail, or 100 for speed
        if self.n_calls % 10 == 0:
            # SB3 stores the most recent observation in 'new_obs'
            obs = self.locals['new_obs']

            # CRITICAL FIX: If obs is a tuple (obs, info), take only the first element
            if isinstance(obs, tuple):
                obs = obs[0]

            obs_tensor = self.model.policy.obs_to_tensor(obs)

            if isinstance(obs_tensor, tuple):
                obs_tensor = obs_tensor[0]
            
            # 1. ESTIMATE VALUE FUNCTION V(s) & POLICY
            if isinstance(self.model, DQN):
                # V(s) for DQN is the maximum Q-value for the current state
                q_values = self.model.q_net(obs_tensor).detach().cpu().numpy()
                value_est = np.max(q_values)
                policy_choice = np.argmax(q_values)
            else:
                # A2C has a dedicated Value head (Critic)
                value_est = self.model.policy.predict_values(obs_tensor).detach().cpu().numpy()
                # Extract Policy Choice (the action with highest probability)
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs
                policy_choice = probs.argmax().detach().cpu().numpy()

            # 2. CAPTURE ENVIRONMENT METRICS
            info = self.locals['infos']

            # CRITICAL FIX: If info is a list, take only the first element
            if isinstance(info, list):
                info = info[0]
            
            self.data_log.append({
                "episode": self.episode_count,
                "step": self.n_calls,
                "value_estimate": value_est,
                "optimum_action": policy_choice,
                "reward": self.locals['rewards'],
                "avg_waiting_time": info.get('system_mean_waiting_time', 0),
                "total_stopped": info.get('system_total_stopped', 0),
                "co2_emissions": info.get('system_total_abs_co2_emission', 0),
                "actual_action": self.locals['actions']
            })

        if self.locals['dones']:
            self.episode_count += 1
            # If randomized routes are enabled, regenerate the route file for the next episode
            if self.use_random_routes and self.route_file:
                from routes_generator import generate_randomized_routes
                generate_randomized_routes(self.route_file)

        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.data_log)
        filename = f"results_{self.algo_name}_{self.experiment_type}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ SAVED: {filename}")