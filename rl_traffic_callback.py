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
        # 1. HANDLE VARIABLE ACCESS (Check if we are in SB3 learn() or a manual loop)
        # Use .get() to avoid KeyErrors
        infos = self.locals.get('infos')
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        new_obs = self.locals.get('new_obs')
        actions = self.locals.get('actions')

        # If these are missing, we are in the Baseline manual loop
        # We try to grab them from the local scope of the run_experiment function
        if infos is None:
            # Pulling from the loop variables we set in run_experiment
            info = self.locals.get('info')
            reward = self.locals.get('reward')
            done = self.locals.get('terminated') or self.locals.get('truncated')
            obs = self.locals.get('obs')
            action = self.locals.get('action') # or 0
        else:
            # Standard SB3 Vectorized Env access
            info = infos[0]
            reward = rewards[0]
            done = dones[0]
            obs = new_obs
            action = actions[0]

        # Log every 10 steps
        if self.n_calls % 10 == 0 and obs is not None:
            # Standard logic for Value Function
            if isinstance(obs, tuple):
                obs = obs[0]
            
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            
            if isinstance(self.model, DQN):
                q_values = self.model.q_net(obs_tensor)
                value_est = q_values.max().item()
                policy_choice = q_values.argmax().item()
            else:
                value_est = self.model.policy.predict_values(obs_tensor).item()
                policy_choice = self.model.policy.get_distribution(obs_tensor).distribution.probs.argmax().item()

            self.data_log.append({
                "episode": self.episode_count,
                "step": self.n_calls,
                "value_estimate": value_est,
                "optimum_action": policy_choice,
                "reward": reward if reward is not None else 0,
                "avg_waiting_time": info.get('system_mean_waiting_time', 0) if info else 0,
                "total_stopped": info.get('system_total_stopped', 0) if info else 0,
                "co2_emissions": info.get('system_total_abs_co2_emission', 0) if info else 0,
                "actual_action": action if action is not None else 0
            })

        # 3. EPISODE END LOGIC
        if done:
            self.episode_count += 1
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