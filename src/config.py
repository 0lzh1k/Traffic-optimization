"""
Configuration file for Smart Traffic Control System

This file contains all default configuration parameters for:
- Environment settings
- RL model hyperparameters
- Training parameters
- Simulation settings
- UI and visualization settings
"""

from typing import Dict, Any


# =============================================================================
# ENVIRONMENT CONFIGURATIONS
# =============================================================================

ENV_CONFIGS = {
    'single_intersection': {
        'max_episode_steps': 3600,  # 1 hour simulation
        'step_length': 30.0,       # 30 seconds per step
        'scenario_type': 'demo',
        'arrival_rates': {
            'normal': 200,  # vehicles per hour
            'peak': 400,
            'low': 100
        }
    },

    'multi_intersection': {
        'num_intersections': 4,
        'max_episode_steps': 3600,  # 1 hour simulation
        'step_length': 30.0,       # 30 seconds per step
        'scenario_type': 'demo',
        'arrival_rates': {
            'peak': 400,    # vehicles/hour
            'normal': 200,
            'low': 100
        }
    },

    'training': {
        'num_intersections': 2,    # Smaller network for faster training
        'max_episode_steps': 1200, # 20 minutes simulation
        'step_length': 30.0,
        'scenario_type': 'demo',
        'arrival_rates': {
            'normal': 200,
            'peak': 300,
            'low': 100
        }
    }
}


# =============================================================================
# RL MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    'ppo': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': [dict(pi=[256, 256], vf=[256, 256])]
        }
    },

    'ppo_light': {  # Lighter version for testing
        'learning_rate': 1e-3,
        'n_steps': 512,
        'batch_size': 32,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
        }
    },

    'dqn': {
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'policy_kwargs': {
            'net_arch': [256, 256]
        }
    },

    'dqn_light': {  # Lighter version for testing
        'learning_rate': 1e-3,
        'buffer_size': 10000,
        'learning_starts': 100,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 500,
        'exploration_fraction': 0.2,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.1,
        'policy_kwargs': {
            'net_arch': [64, 64]
        }
    }
}


# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================

TRAINING_CONFIGS = {
    'default': {
        'total_timesteps': 100000,
        'eval_freq': 5000,
        'n_eval_episodes': 5,
        'save_freq': 10000,
        'model_save_path': 'models/',
        'metrics_save_path': 'training_metrics.pkl',
        'log_interval': 100
    },

    'quick_test': {  # For testing and debugging
        'total_timesteps': 10000,
        'eval_freq': 1000,
        'n_eval_episodes': 3,
        'save_freq': 5000,
        'model_save_path': 'models/',
        'metrics_save_path': 'training_metrics.pkl',
        'log_interval': 50
    },

    'production': {  # For full training runs
        'total_timesteps': 1000000,
        'eval_freq': 50000,
        'n_eval_episodes': 10,
        'save_freq': 50000,
        'model_save_path': 'models/',
        'metrics_save_path': 'training_metrics.pkl',
        'log_interval': 1000
    }
}


# =============================================================================
# SIMULATION CONFIGURATIONS
# =============================================================================

SIMULATION_CONFIGS = {
    'demo': {
        'duration_hours': 1.0,
        'arrival_rate': 200,  # vehicles per hour
        'peak_multiplier': 1.5,
        'control_policy': 'Fixed Time'
    },

    'peak_hour': {
        'duration_hours': 2.0,
        'arrival_rate': 300,
        'peak_multiplier': 2.5,
        'control_policy': 'Fixed Time'
    },

    'incident': {
        'duration_hours': 1.5,
        'arrival_rate': 250,
        'peak_multiplier': 1.8,
        'control_policy': 'Fixed Time'
    },

    'variable_demand': {
        'duration_hours': 4.0,
        'arrival_rate': 180,
        'peak_multiplier': 3.0,
        'control_policy': 'Fixed Time'
    }
}


# =============================================================================
# UI AND VISUALIZATION CONFIGURATIONS
# =============================================================================

UI_CONFIGS = {
    'dashboard': {
        'auto_refresh_interval': 5,  # seconds
        'max_display_points': 100,
        'default_time_range': 24,  # hours
        'chart_height': 400,
        'map_zoom': 14
    },

    'training': {
        'progress_update_freq': 10,  # updates per second
        'metrics_display_freq': 100,  # steps
        'max_training_time': 3600  # seconds
    },

    'comparison': {
        'default_episodes': 10,
        'max_episodes': 50,
        'confidence_interval': 0.95
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_env_config(config_name: str = 'training') -> Dict[str, Any]:
    """Get environment configuration by name."""
    return ENV_CONFIGS.get(config_name, ENV_CONFIGS['training']).copy()


def get_model_config(algorithm: str = 'ppo', variant: str = 'default') -> Dict[str, Any]:
    """Get model configuration for specified algorithm and variant."""
    config_key = f"{algorithm}_{variant}" if variant != 'default' else algorithm
    return MODEL_CONFIGS.get(config_key, MODEL_CONFIGS[algorithm]).copy()


def get_training_config(config_name: str = 'default') -> Dict[str, Any]:
    """Get training configuration by name."""
    return TRAINING_CONFIGS.get(config_name, TRAINING_CONFIGS['default']).copy()


def get_simulation_config(scenario: str = 'demo') -> Dict[str, Any]:
    """Get simulation configuration by scenario name."""
    return SIMULATION_CONFIGS.get(scenario, SIMULATION_CONFIGS['demo']).copy()


def get_ui_config(section: str = 'dashboard') -> Dict[str, Any]:
    """Get UI configuration for specified section."""
    return UI_CONFIGS.get(section, {}).copy()


# =============================================================================
# DEFAULT CONFIGURATIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Default configurations that match the original hardcoded values
DEFAULT_ENV_CONFIG = get_env_config('training')
DEFAULT_MODEL_CONFIG_PPO = get_model_config('ppo')
DEFAULT_MODEL_CONFIG_DQN = get_model_config('dqn')
DEFAULT_TRAINING_CONFIG = get_training_config('default')