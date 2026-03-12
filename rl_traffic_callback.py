import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback

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
        # Log every 10 steps
        if self.n_calls % 10 == 0:
            obs = self.locals['new_obs']

            # Extract the single observation from the Vectorized Environment
            if isinstance(obs, tuple):
                obs = obs[0]
            
            # obs_to_tensor returns (tensor, vectorized_env), so we take [0]
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            
            # 1. ESTIMATE VALUE FUNCTION V(s) & POLICY
            if isinstance(self.model, DQN):
                q_values = self.model.q_net(obs_tensor)
                # Use .item() to extract clean scalars for plotting
                value_est = q_values.max().item()
                policy_choice = q_values.argmax().item()
            else:
                # A2C
                value_est = self.model.policy.predict_values(obs_tensor).item()
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs
                policy_choice = probs.argmax().item()

            # 2. CAPTURE ENVIRONMENT METRICS
            info = self.locals['infos'][0] # Unpack from VecEnv array
            
            self.data_log.append({
                "episode": self.episode_count,
                "step": self.n_calls,
                "value_estimate": value_est,
                "optimum_action": policy_choice,
                "reward": self.locals['rewards'][0], # Unpack from VecEnv array
                "avg_waiting_time": info.get('system_mean_waiting_time', 0),
                "total_stopped": info.get('system_total_stopped', 0),
                "co2_emissions": info.get('system_total_abs_co2_emission', 0),
                "actual_action": self.locals['actions'][0] # Unpack from VecEnv array
            })

        # 3. EPISODE END LOGIC & ROUTE GENERATION
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            # Generate the randomized routes for the next upcoming episode
            if self.use_random_routes and self.route_file:
                from routes_generator import generate_randomized_routes
                if self.episode_count % 5 == 0:
                    generate_randomized_routes(self.route_file)

        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.data_log)
        filename = f"results_{self.algo_name}_{self.experiment_type}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ SAVED: {filename}")