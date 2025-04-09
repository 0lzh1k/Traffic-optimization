"""
Reinforcement Learning agents for traffic signal control
"""

import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import sys
from typing import Dict, Any, Optional, Callable
import pickle
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulation.gym_environment import TrafficControlEnv, MultiIntersectionEnv, SingleIntersectionEnv
from config import (
    get_env_config, get_model_config, get_training_config,
    DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG_PPO, DEFAULT_MODEL_CONFIG_DQN, DEFAULT_TRAINING_CONFIG
)

class TrafficMetricsCallback(BaseCallback):
    """
    Custom callback for tracking traffic-specific metrics during training
    """
    
    def __init__(self, log_interval: int = 100, save_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_throughputs = []
        self.episode_queue_lengths = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        if hasattr(self.training_env, 'get_attr'):
            # Get metrics from all environments
            episode_stats_list = self.training_env.get_attr('episode_stats')
            episode_rewards_list = self.training_env.get_attr('episode_reward')
            
            for episode_stats, episode_reward in zip(episode_stats_list, episode_rewards_list):
                self.episode_rewards.append(episode_reward)
                self.episode_delays.append(episode_stats['total_delay'])
                self.episode_throughputs.append(episode_stats['total_throughput'])
                self.episode_queue_lengths.append(episode_stats['avg_queue_length'])
        
        # Log metrics
        if len(self.episode_rewards) > 0:
            self.logger.record("traffic/episode_reward", np.mean(self.episode_rewards[-10:]))
            self.logger.record("traffic/total_delay", np.mean(self.episode_delays[-10:]))
            self.logger.record("traffic/throughput", np.mean(self.episode_throughputs[-10:]))
            self.logger.record("traffic/avg_queue_length", np.mean(self.episode_queue_lengths[-10:]))
        
        # Save metrics periodically
        if self.save_path and len(self.episode_rewards) % self.log_interval == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_delays': self.episode_delays,
            'episode_throughputs': self.episode_throughputs,
            'episode_queue_lengths': self.episode_queue_lengths
        }
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump(metrics, f)

class TrafficRLAgent:
    """
    Main RL agent wrapper for traffic signal control optimization.
    
    This class provides a unified interface for training and evaluating reinforcement
    learning agents (PPO, DQN) on traffic control tasks. It handles environment
    creation, model training, evaluation, and result tracking.
    
    Supported Algorithms:
        - PPO (Proximal Policy Optimization): Good for continuous control
        - DQN (Deep Q-Network): Good for discrete action spaces
    
    Features:
        - Automatic environment configuration based on algorithm
        - Training progress monitoring with custom callbacks
        - Model evaluation and comparison
        - Result saving and loading
    
    Args:
        algorithm: RL algorithm to use ("PPO" or "DQN")
        env_config: Environment configuration parameters
        model_config: Model architecture and training parameters
        training_config: Training hyperparameters and settings
    """
    
    def __init__(self, 
                 algorithm: str = "PPO",
                 env_config: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        
        self.algorithm = algorithm
        self.env_config = env_config or {}
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        
        # Set default configurations
        self._set_default_configs()
        
        # Initialize environment and model
        self.env = None
        self.model = None
        self.training_metrics = None
        
    def _set_default_configs(self):
        """Set default configurations for environment, model, and training"""
        
        # Default environment config
        default_env_config = DEFAULT_ENV_CONFIG.copy()
        self.env_config = {**default_env_config, **self.env_config}
        
        # Default model config based on algorithm
        if self.algorithm == "PPO":
            default_model_config = DEFAULT_MODEL_CONFIG_PPO.copy()
        elif self.algorithm == "DQN":
            default_model_config = DEFAULT_MODEL_CONFIG_DQN.copy()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        self.model_config = {**default_model_config, **self.model_config}
        
        # Default training config
        default_training_config = DEFAULT_TRAINING_CONFIG.copy()
        self.training_config = {**default_training_config, **self.training_config}
    
    def create_environment(self, n_envs: int = 1):
        """
        Create and configure the training environment.
        
        For DQN, uses SingleIntersectionEnv. For PPO, uses TrafficControlEnv
        with the configured number of intersections.
        
        Args:
            n_envs: Number of parallel environments (for vectorized training)
            
        Returns:
            Configured environment (or VecEnv if n_envs > 1)
        """
        
        def make_env():
            # Use SingleIntersectionEnv for DQN, TrafficControlEnv for PPO
            if self.algorithm == "DQN" and self.env_config.get('num_intersections', 4) == 1:
                # Remove num_intersections from config for SingleIntersectionEnv
                single_env_config = {k: v for k, v in self.env_config.items() if k != 'num_intersections'}
                env = SingleIntersectionEnv(**single_env_config)
            else:
                env = TrafficControlEnv(**self.env_config)
            env = Monitor(env)
            return env
        
        if n_envs == 1:
            self.env = make_env()
        else:
            self.env = make_vec_env(make_env, n_envs=n_envs)
        
        return self.env
    
    def create_model(self, env=None):
        """Create RL model"""
        if env is None:
            env = self.env
        
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", env, verbose=1, **self.model_config)
        elif self.algorithm == "DQN":
            self.model = DQN("MlpPolicy", env, verbose=1, **self.model_config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return self.model
    
    def train(self, callback=None):
        """
        Train the RL model with the configured parameters.
        
        Uses the specified training configuration and saves the trained model
        and metrics to the configured paths.
        
        Args:
            callback: Optional custom callback for training monitoring.
                     If None, uses TrafficMetricsCallback.
                     
        Returns:
            Trained model instance
            
        Raises:
            ValueError: If environment or model not created first
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create default callback if none provided
        if callback is None:
            callback = TrafficMetricsCallback(
                save_path=self.training_config['metrics_save_path']
            )
        
        # Train model
        self.model.learn(
            total_timesteps=self.training_config['total_timesteps'],
            callback=callback
        )
        
        # Save final model
        model_path = os.path.join(
            self.training_config['model_save_path'], 
            f"{self.algorithm}_traffic_model"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        return self.model
    
    def evaluate(self, n_episodes: int = 10, render: bool = False):
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not created or loaded.")
        
        episode_rewards = []
        episode_stats = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_stats.append(info['episode_stats'])
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Delay = {info['episode_stats']['total_delay']:.1f}, "
                  f"Throughput = {info['episode_stats']['total_throughput']:.1f}")
        
        # Calculate average metrics
        avg_reward = np.mean(episode_rewards)
        avg_delay = np.mean([stats['total_delay'] for stats in episode_stats])
        avg_throughput = np.mean([stats['total_throughput'] for stats in episode_stats])
        avg_queue_length = np.mean([stats['avg_queue_length'] for stats in episode_stats])
        
        evaluation_results = {
            'avg_reward': avg_reward,
            'avg_delay': avg_delay,
            'avg_throughput': avg_throughput,
            'avg_queue_length': avg_queue_length,
            'episode_rewards': episode_rewards,
            'episode_stats': episode_stats
        }
        
        return evaluation_results
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load trained model"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return self.model

class PolicyComparison:
    """
    Compare different traffic control policies
    """
    
    def __init__(self, env_config: Dict[str, Any] = None):
        self.env_config = env_config or {}
        self.results = {}
    
    def evaluate_fixed_time_policy(self, n_episodes: int = 10):
        """Evaluate fixed-time traffic control"""
        # Create environment with fixed-time control
        env_config = {**self.env_config, 'scenario_type': 'demo'}
        env = TrafficControlEnv(**env_config)
        
        episode_stats = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                # Fixed-time policy: always extend current phase (action = 0)
                action = np.zeros(env.action_space.shape[0], dtype=int)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_stats.append(info['episode_stats'])
        
        # Calculate average metrics
        self.results['fixed_time'] = {
            'avg_delay': np.mean([stats['total_delay'] for stats in episode_stats]),
            'avg_throughput': np.mean([stats['total_throughput'] for stats in episode_stats]),
            'avg_queue_length': np.mean([stats['avg_queue_length'] for stats in episode_stats])
        }
        
        env.close()
        return self.results['fixed_time']
    
    def evaluate_actuated_policy(self, n_episodes: int = 10):
        """Evaluate actuated traffic control (gap-out logic)"""
        env_config = {**self.env_config, 'scenario_type': 'demo'}
        env = TrafficControlEnv(**env_config)
        
        episode_stats = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            phase_timer = 0
            min_green = 10  # Minimum green time
            max_green = 60  # Maximum green time
            
            while not done:
                phase_timer += 1
                
                # Simple actuated logic: switch if minimum green satisfied and queues on other approaches
                action = np.zeros(env.action_space.shape[0], dtype=int)
                
                if phase_timer > min_green:
                    # Check if other approaches have significant queues
                    # This is a simplified logic based on observation
                    if phase_timer > max_green or np.random.random() < 0.1:  # Switch probability
                        action = np.ones(env.action_space.shape[0], dtype=int)  # Switch phase
                        phase_timer = 0
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_stats.append(info['episode_stats'])
        
        self.results['actuated'] = {
            'avg_delay': np.mean([stats['total_delay'] for stats in episode_stats]),
            'avg_throughput': np.mean([stats['total_throughput'] for stats in episode_stats]),
            'avg_queue_length': np.mean([stats['avg_queue_length'] for stats in episode_stats])
        }
        
        env.close()
        return self.results['actuated']
    
    def evaluate_rl_policy(self, model_path: str, algorithm: str = "PPO", n_episodes: int = 10):
        """Evaluate trained RL policy"""
        env_config = {**self.env_config, 'scenario_type': 'demo'}
        env = TrafficControlEnv(**env_config)
        
        # Load trained model
        if algorithm == "PPO":
            model = PPO.load(model_path, env=env)
        elif algorithm == "DQN":
            model = DQN.load(model_path, env=env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        episode_stats = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_stats.append(info['episode_stats'])
        
        self.results['rl'] = {
            'avg_delay': np.mean([stats['total_delay'] for stats in episode_stats]),
            'avg_throughput': np.mean([stats['total_throughput'] for stats in episode_stats]),
            'avg_queue_length': np.mean([stats['avg_queue_length'] for stats in episode_stats])
        }
        
        env.close()
        return self.results['rl']
    
    def compare_policies(self):
        """Generate comparison report"""
        if not self.results:
            print("No results to compare. Run evaluation methods first.")
            return
        
        print("Traffic Control Policy Comparison")
        print("=" * 50)
        
        metrics = ['avg_delay', 'avg_throughput', 'avg_queue_length']
        
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for policy, results in self.results.items():
                print(f"  {policy.replace('_', ' ').title()}: {results[metric]:.2f}")
        
        # Find best policy for each metric
        print("\nBest Policies:")
        for metric in metrics:
            if metric == 'avg_delay' or metric == 'avg_queue_length':
                # Lower is better
                best_policy = min(self.results.keys(), 
                                key=lambda p: self.results[p][metric])
            else:
                # Higher is better
                best_policy = max(self.results.keys(), 
                                key=lambda p: self.results[p][metric])
            
            print(f"  {metric.replace('_', ' ').title()}: {best_policy.replace('_', ' ').title()}")
        
        return self.results
    
    def plot_comparison(self, save_path: str = None):
        """Plot policy comparison"""
        if not self.results:
            print("No results to plot.")
            return
        
        policies = list(self.results.keys())
        metrics = ['avg_delay', 'avg_throughput', 'avg_queue_length']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.results[policy][metric] for policy in policies]
            axes[i].bar(policies, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
