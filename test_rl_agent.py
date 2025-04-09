#!/usr/bin/env python3
"""
Test script for RL agent components
Tests action/state shapes, reward calculation, and basic environment steps
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_agent.traffic_agents import TrafficRLAgent
from simulation.gym_environment import SingleIntersectionEnv, TrafficControlEnv


def test_single_intersection_env():
    """Test SingleIntersectionEnv basic functionality"""
    print("Testing SingleIntersectionEnv...")

    # Create environment
    env = SingleIntersectionEnv(max_episode_steps=10, step_length=5.0)

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")

    # Test step
    action = 0  # Extend current phase
    obs_next, reward, terminated, truncated, info = env.step(action)
    print(f"Step reward: {reward}")
    print(f"Episode stats: {info['episode_stats']}")

    # Test action 1 (switch phase)
    obs_next2, reward2, terminated2, truncated2, info2 = env.step(1)
    print(f"Switch phase reward: {reward2}")

    print("SingleIntersectionEnv test passed!\n")


def test_multi_intersection_env():
    """Test TrafficControlEnv basic functionality"""
    print("Testing TrafficControlEnv...")

    # Create environment
    env = TrafficControlEnv(num_intersections=2, max_episode_steps=10, step_length=5.0)

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")

    # Test step with multi-discrete action
    action = np.array([0, 1])  # First intersection extends, second switches
    obs_next, reward, terminated, truncated, info = env.step(action)
    print(f"Step reward: {reward}")
    print(f"Episode stats: {info['episode_stats']}")

    print("TrafficControlEnv test passed!\n")


def test_rl_agent_creation():
    """Test RL agent creation and basic functionality"""
    print("Testing RL Agent creation...")

    # Test PPO agent
    agent = TrafficRLAgent(
        algorithm="PPO",
        env_config={'num_intersections': 2, 'max_episode_steps': 100, 'step_length': 5.0},
        model_config={'learning_rate': 1e-3},
        training_config={'total_timesteps': 1000}
    )

    # Create environment
    env = agent.create_environment()
    print(f"Environment created: {type(env)}")

    # Create model
    model = agent.create_model()
    print(f"Model created: {type(model)}")

    # Test basic prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")

    print("RL Agent creation test passed!\n")


def test_reward_calculation():
    """Test reward calculation logic"""
    print("Testing reward calculation...")

    env = SingleIntersectionEnv(max_episode_steps=20, step_length=5.0)
    obs, info = env.reset(seed=42)

    rewards = []
    for step in range(5):
        # Alternate between actions
        action = step % 2
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"Step {step}: Action={action}, Reward={reward:.3f}")

    print(f"Reward statistics: Mean={np.mean(rewards):.3f}, Std={np.std(rewards):.3f}")
    print("Reward calculation test passed!\n")


def test_shapes_consistency():
    """Test that shapes are consistent across different components"""
    print("Testing shapes consistency...")

    # Single intersection
    single_env = SingleIntersectionEnv()
    obs_single, _ = single_env.reset()
    action_single = single_env.action_space.sample()
    obs_next_single, _, _, _, _ = single_env.step(action_single)

    print(f"Single env - Obs shape: {obs_single.shape}, Action shape: {np.array(action_single).shape}")

    # Multi intersection
    multi_env = TrafficControlEnv(num_intersections=2)
    obs_multi, _ = multi_env.reset()
    action_multi = multi_env.action_space.sample()
    obs_next_multi, _, _, _, _ = multi_env.step(action_multi)

    print(f"Multi env - Obs shape: {obs_multi.shape}, Action shape: {action_multi.shape}")

    # Expected shapes
    expected_single_obs = 14
    expected_multi_obs = 2 * (4 * 3 + 2)  # 2 intersections * (4 approaches * 3 features + 2 light features)

    assert obs_single.shape[0] == expected_single_obs, f"Single obs shape mismatch: {obs_single.shape[0]} != {expected_single_obs}"
    assert obs_multi.shape[0] == expected_multi_obs, f"Multi obs shape mismatch: {obs_multi.shape[0]} != {expected_multi_obs}"

    print("Shapes consistency test passed!\n")


if __name__ == "__main__":
    print("Starting RL Agent Component Tests\n")

    try:
        test_single_intersection_env()
        test_multi_intersection_env()
        test_rl_agent_creation()
        test_reward_calculation()
        test_shapes_consistency()

        print("All tests passed! ✅")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)