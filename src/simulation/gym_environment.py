import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import simpy
import sys
import os


sys.path.append(os.path.dirname(__file__))

from traffic_environment import TrafficSimulation, TrafficNetwork


class SingleIntersectionEnv(gym.Env):    
    """
    Reinforcement Learning environment for a single traffic intersection.
    
    This environment simulates traffic flow at a single intersection with four approaches
    (north, south, east, west). The agent can choose to either extend the current traffic
    light phase or switch to the next phase.
    
    Observation Space:
        14-dimensional vector containing:
        - Queue lengths for each approach (4 values)
        - Waiting times for each approach (4 values) 
        - Current phase indicator (2 values: [NS_green, EW_green])
        - Time in current phase (1 value)
        - Total vehicles served (3 values, one per phase type)
    
    Action Space:
        Discrete(2): [0] extend current phase, [1] switch to next phase
    
    Reward:
        Negative reward based on queue lengths, delays, and positive reward for throughput.
        Formula: base_reward + queue_penalty + delay_penalty + throughput_reward
    
    Args:
        max_episode_steps: Maximum number of steps per episode
        step_length: Duration of each simulation step in seconds
        scenario_type: Type of traffic scenario ("demo", "peak", etc.)
        arrival_rates: Dictionary of arrival rates for different traffic conditions
    """
    
    def __init__(self, 
                 max_episode_steps: int = 3600,  # 1 hour simulation
                 step_length: float = 30.0,     # 30 seconds per step
                 scenario_type: str = "demo",
                 arrival_rates: Optional[Dict] = None):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.scenario_type = scenario_type
        self.current_step = 0
        
        if arrival_rates is None:
            self.arrival_rates = {'normal': 200, 'peak': 400, 'low': 100}
        else:
            self.arrival_rates = arrival_rates
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.simulation = None
        self.episode_reward = 0.0
        self.episode_stats = {
            'total_delay': 0.0,
            'total_throughput': 0.0,
            'avg_queue_length': 0.0
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Creates a new traffic simulation with a single intersection, starts traffic
        light control and vehicle arrival processes, and returns the initial observation.
        
        Args:
            seed: Random seed for reproducible simulations
            options: Additional options (currently unused)
            
        Returns:
            Tuple of (observation, info) where:
            - observation: Initial state vector (14-dimensional)
            - info: Dictionary with episode information including stats
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create new simulation with single intersection
        self.simulation = TrafficSimulation(random_seed=seed or 42)
        # Create single intersection network
        self.simulation.network.add_intersection('main_int', (0.0, 0.0))
        
        # Start traffic light control (use fixed time by default, RL will override actions)
        self.simulation.env.process(
            self.simulation.network.run_fixed_time_control('main_int')
        )
        
        # Start vehicle arrivals for all approaches
        for approach in ['north', 'south', 'east', 'west']:
            arrival_rate = self.arrival_rates.get('normal', 200)  # vehicles per hour
            max_sim_time = self.max_episode_steps * self.step_length
            self.simulation.env.process(
                self.simulation.network.generate_vehicle_arrivals(
                    'main_int', approach, arrival_rate, max_sim_time
                )
            )
        
        # Reset tracking variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_stats = {
            'total_delay': 0.0,
            'total_throughput': 0.0,
            'avg_queue_length': 0.0
        }
        
        # Reset reward tracking variables
        self._previous_delay = 0.0
        self._previous_served = 0
        
        # Run simulation for one step to get initial state
        initial_target = self.simulation.env.now + self.step_length
        self.simulation.env.run(until=initial_target)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Applies the given action to the traffic light control, runs the simulation
        for step_length seconds, calculates the reward, and returns the new state.
        
        Args:
            action: Action to take (0=extend current phase, 1=switch phase)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info):
            - observation: New state vector after action
            - reward: Reward for this step
            - terminated: Whether episode ended naturally
            - truncated: Whether episode was truncated (max steps)
            - info: Additional information including episode stats
        """
        # Apply action to the single intersection
        intersection = list(self.simulation.network.intersections.values())[0]
        
        if action == 0:  # Extend current phase
            pass  # Keep current phase
        elif action == 1:  # Switch phase
            intersection.traffic_light.switch_phase()
        
        # Run simulation for step_length seconds
        target_time = self.simulation.env.now + self.step_length
        self.simulation.env.run(until=target_time)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        
        # Update episode stats
        self._update_episode_stats()
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        """Get current state observation for single intersection"""
        if not self.simulation or not self.simulation.network.intersections:
            return np.zeros(14, dtype=np.float32)
        
        intersection = list(self.simulation.network.intersections.values())[0]
        state_vector = intersection.get_state_vector()
        
        # Ensure we return exactly 14 features
        if len(state_vector) != 14:
            # Pad or truncate to 14 features
            padded_state = np.zeros(14, dtype=np.float32)
            min_len = min(len(state_vector), 14)
            padded_state[:min_len] = state_vector[:min_len]
            return padded_state
        
        return state_vector.astype(np.float32)
        
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic performance metrics.
        
        The reward function encourages efficient traffic flow by penalizing:
        - Current queue lengths (congestion)
        - Incremental delays added this step
        - And rewarding vehicles served this step
        
        Returns:
            Float reward value (typically negative, with positive components for throughput)
        """
        if not self.simulation or not self.simulation.network.intersections:
            return 0.0
        
        intersection = list(self.simulation.network.intersections.values())[0]
        
        # Calculate current step metrics
        current_queue = sum(app.get_queue_length() for app in intersection.approaches.values())
        
        # Get incremental delay (delay added in this step)
        current_total_delay = sum(app.total_delay for app in intersection.approaches.values())
        previous_delay = getattr(self, '_previous_delay', 0.0)
        step_delay = current_total_delay - previous_delay
        self._previous_delay = current_total_delay
        
        # Get incremental throughput (vehicles served in this step)
        current_total_served = sum(app.vehicles_served for app in intersection.approaches.values())
        previous_served = getattr(self, '_previous_served', 0)
        step_served = current_total_served - previous_served
        self._previous_served = current_total_served
        
        # Reward components (scaled appropriately)
        queue_penalty = -current_queue * 1.0      # Penalty for current queue length
        delay_penalty = -step_delay * 0.1         # Penalty for delay added this step
        throughput_reward = step_served * 2.0     # Reward for vehicles served this step
        
        # Base reward to avoid all zeros
        base_reward = -0.1  # Small negative base to encourage efficiency
        
        reward = base_reward + queue_penalty + delay_penalty + throughput_reward
        
        return reward
        
    def _update_episode_stats(self):
        """Update episode statistics"""
        if not self.simulation or not self.simulation.network.intersections:
            return
            
        intersection = list(self.simulation.network.intersections.values())[0]
        
        self.episode_stats['total_delay'] = sum(app.total_delay for app in intersection.approaches.values())
        self.episode_stats['total_throughput'] = sum(app.vehicles_served for app in intersection.approaches.values())
        self.episode_stats['avg_queue_length'] = np.mean([app.get_queue_length() for app in intersection.approaches.values()])
        
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'episode_stats': self.episode_stats.copy()
        }


class TrafficControlEnv(gym.Env):    
    """
    Multi-intersection traffic control environment for reinforcement learning.
    
    This environment simulates traffic flow across multiple intersections in a network.
    The agent controls traffic light phases at each intersection simultaneously.
    
    Observation Space:
        Multi-dimensional vector where each intersection contributes:
        - 4 approaches × 3 features (queue_length, delay, waiting_time) = 12 features
        - 2 light phase features = 14 features per intersection
        Total: num_intersections × 14 features
    
    Action Space:
        MultiDiscrete([2] × num_intersections): 
        For each intersection: [0] extend current phase, [1] switch phase
    
    Reward:
        Aggregated reward across all intersections based on network-wide performance.
        Penalizes total congestion and delays, rewards total throughput.
    
    Args:
        num_intersections: Number of intersections in the network
        max_episode_steps: Maximum steps per episode
        step_length: Duration of each simulation step in seconds
        arrival_rates: Traffic arrival rates for different conditions
        scenario_type: Type of traffic scenario
    """
    
    def __init__(self, 
                 num_intersections: int = 4,
                 max_episode_steps: int = 3600,  # 1 hour simulation
                 step_length: float = 30.0,     # 30 seconds per step
                 arrival_rates: Dict[str, float] = None,
                 scenario_type: str = "demo"):
        
        super().__init__()
        
        self.num_intersections = num_intersections
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.scenario_type = scenario_type
        self.current_step = 0
        
        # Default arrival rates if not provided
        if arrival_rates is None:
            self.arrival_rates = {
                'peak': 400,    # vehicles/hour
                'normal': 200,
                'low': 100
            }
        else:
            self.arrival_rates = arrival_rates
        
        # Define observation space BEFORE reset
        # For each intersection: 4 approaches * 3 features (queue_length, delay, waiting_time) + 2 light features
        obs_size = num_intersections * (4 * 3 + 2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Define action space
        # For each intersection: 0 = extend current phase, 1 = switch phase
        self.action_space = spaces.MultiDiscrete([2] * num_intersections)
        
        # Initialize simulation
        self.simulation = None
        
        # Tracking variables
        self.episode_reward = 0.0
        self.episode_stats = {
            'total_delay': 0.0,
            'total_throughput': 0.0,
            'avg_queue_length': 0.0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create new simulation
        self.simulation = TrafficSimulation(random_seed=seed or 42)
        self.simulation.setup_scenario(self.scenario_type)
        
        # Reset tracking variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_stats = {
            'total_delay': 0.0,
            'total_throughput': 0.0,
            'avg_queue_length': 0.0
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Apply actions to intersections
        intersection_ids = list(self.simulation.network.intersections.keys())
        for i, intersection_id in enumerate(intersection_ids):
            if i < len(action):
                intersection = self.simulation.network.intersections[intersection_id]
                intersection.apply_action(action[i])
        
        # Run simulation for step_length seconds
        target_time = self.simulation.env.now + self.step_length
        self.simulation.env.run(until=target_time)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        
        # Update episode stats
        self._update_episode_stats()
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        observation = []
        
        # Check if simulation and network exist
        if self.simulation is None or not hasattr(self.simulation, 'network'):
            # Return zeros if simulation not ready
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        if not self.simulation.network.intersections:
            # Return zeros if no intersections
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        for intersection in self.simulation.network.intersections.values():
            state_vector = intersection.get_state_vector()
            observation.extend(state_vector)
        
        # Pad observation if needed
        expected_size = self.observation_space.shape[0]
        while len(observation) < expected_size:
            observation.append(0.0)
        
        # Truncate if too long
        observation = observation[:expected_size]
        
        return np.array(observation, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic performance"""
        total_delay = 0.0
        total_queue_length = 0.0
        total_throughput = 0.0
        
        for intersection in self.simulation.network.intersections.values():
            intersection.update_stats()
            total_delay += intersection.stats.total_delay
            total_queue_length += intersection.stats.avg_queue_length
            total_throughput += intersection.stats.throughput
        
        # Reward components
        delay_penalty = -total_delay / 100.0  # Normalize delay penalty
        queue_penalty = -total_queue_length / 10.0  # Normalize queue penalty
        throughput_bonus = total_throughput / 1000.0  # Normalize throughput bonus
        
        # Combined reward
        reward = delay_penalty + queue_penalty + throughput_bonus
        
        return reward
    
    def _update_episode_stats(self):
        """Update episode-level statistics"""
        total_delay = 0.0
        total_throughput = 0.0
        queue_lengths = []
        
        for intersection in self.simulation.network.intersections.values():
            intersection.update_stats()
            total_delay += intersection.stats.total_delay
            total_throughput += intersection.stats.throughput
            queue_lengths.append(intersection.stats.avg_queue_length)
        
        self.episode_stats['total_delay'] = total_delay
        self.episode_stats['total_throughput'] = total_throughput
        self.episode_stats['avg_queue_length'] = np.mean(queue_lengths) if queue_lengths else 0.0
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state"""
        network_state = self.simulation.network.get_network_state()
        
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'episode_stats': self.episode_stats.copy(),
            'network_state': network_state,
            'simulation_time': self.simulation.env.now
        }
        
        return info
    
    def render(self, mode='human'):
        """Render the environment (for debugging)"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Simulation Time: {self.simulation.env.now:.1f}s")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("Intersection States:")
            
            for int_id, intersection in self.simulation.network.intersections.items():
                print(f"  {int_id}:")
                print(f"    Phase: {intersection.traffic_light.current_phase.name}")
                print(f"    Queue Lengths: {[app.get_queue_length() for app in intersection.approaches.values()]}")
                print(f"    Total Delay: {intersection.stats.total_delay:.1f}s")
            print("-" * 50)
    
    def close(self):
        """Clean up environment"""
        if self.simulation:
            del self.simulation
        self.simulation = None

class MultiIntersectionEnv(TrafficControlEnv):
    """
    Extended environment for multi-intersection coordination
    """
    
    def __init__(self, network_config: Dict[str, Any], **kwargs):
        self.network_config = network_config
        super().__init__(**kwargs)
    
    def setup_custom_network(self):
        """Setup custom network from configuration"""
        # Clear existing network
        self.simulation.network = TrafficNetwork(self.simulation.env)
        
        # Add intersections
        for int_config in self.network_config.get('intersections', []):
            self.simulation.network.add_intersection(
                int_config['id'],
                tuple(int_config['coordinates'])
            )
        
        # Add roads
        for road_config in self.network_config.get('roads', []):
            self.simulation.network.add_road(
                road_config['from'],
                road_config['to'],
                road_config.get('travel_time', 60.0),
                road_config.get('capacity', 1000.0)
            )
        
        # Setup traffic generation
        for traffic_config in self.network_config.get('traffic_generation', []):
            arrival_rate = traffic_config.get('arrival_rate', 200)
            duration = traffic_config.get('duration', 3600)
            
            self.simulation.env.process(
                self.simulation.network.generate_vehicle_arrivals(
                    traffic_config['intersection_id'],
                    traffic_config['approach'],
                    arrival_rate,
                    duration
                )
            )
        
        # Start traffic light controllers
        for int_id in self.simulation.network.intersections.keys():
            if self.network_config.get('control_type', 'rl') == 'fixed_time':
                self.simulation.env.process(
                    self.simulation.network.run_fixed_time_control(int_id)
                )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset with custom network setup"""
        observation, info = super().reset(seed, options)
        
        # Setup custom network if configuration provided
        if self.network_config:
            self.setup_custom_network()
            observation = self._get_observation()
            info = self._get_info()
        
        return observation, info

# Utility functions for environment creation

def create_peak_hour_scenario() -> Dict[str, Any]:
    """Create peak hour traffic scenario configuration"""
    return {
        'intersections': [
            {'id': 'main_st_1st', 'coordinates': [0.0, 0.0]},
            {'id': 'main_st_2nd', 'coordinates': [1.0, 0.0]},
            {'id': 'main_st_3rd', 'coordinates': [2.0, 0.0]},
            {'id': 'broadway_1st', 'coordinates': [0.0, 1.0]},
            {'id': 'broadway_2nd', 'coordinates': [1.0, 1.0]},
            {'id': 'broadway_3rd', 'coordinates': [2.0, 1.0]}
        ],
        'roads': [
            {'from': 'main_st_1st', 'to': 'main_st_2nd', 'travel_time': 45},
            {'from': 'main_st_2nd', 'to': 'main_st_3rd', 'travel_time': 45},
            {'from': 'broadway_1st', 'to': 'broadway_2nd', 'travel_time': 45},
            {'from': 'broadway_2nd', 'to': 'broadway_3rd', 'travel_time': 45},
            {'from': 'main_st_1st', 'to': 'broadway_1st', 'travel_time': 30},
            {'from': 'main_st_2nd', 'to': 'broadway_2nd', 'travel_time': 30},
            {'from': 'main_st_3rd', 'to': 'broadway_3rd', 'travel_time': 30}
        ],
        'traffic_generation': [
            {'intersection_id': 'main_st_1st', 'approach': 'north', 'arrival_rate': 600},
            {'intersection_id': 'main_st_1st', 'approach': 'south', 'arrival_rate': 500},
            {'intersection_id': 'main_st_1st', 'approach': 'east', 'arrival_rate': 400},
            {'intersection_id': 'main_st_1st', 'approach': 'west', 'arrival_rate': 400},
        ],
        'control_type': 'rl'
    }

def create_incident_scenario() -> Dict[str, Any]:
    """Create incident scenario with reduced capacity"""
    config = create_peak_hour_scenario()
    
    # Reduce capacity on one road to simulate incident
    for road in config['roads']:
        if road['from'] == 'main_st_1st' and road['to'] == 'main_st_2nd':
            road['capacity'] = 500  # 50% capacity reduction
            
    return config
