"""
Traffic Environment using SimPy for discrete event simulation
with Gymnasium interface for RL integration.
"""

import simpy
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum

class TrafficPhase(Enum):
    NS_GREEN = 0  # North-South green
    NS_YELLOW = 1
    EW_GREEN = 2  # East-West green
    EW_YELLOW = 3

@dataclass
class VehicleStats:
    arrival_time: float
    departure_time: Optional[float] = None
    delay: float = 0.0
    queue_position: int = 0

@dataclass
class IntersectionStats:
    total_vehicles: int = 0
    total_delay: float = 0.0
    avg_queue_length: float = 0.0
    throughput: float = 0.0
    
class Vehicle:
    def __init__(self, vehicle_id: int, arrival_time: float, origin: str, destination: str):
        self.id = vehicle_id
        self.arrival_time = arrival_time
        self.origin = origin
        self.destination = destination
        self.departure_time = None
        self.delay = 0.0

class TrafficLight:
    def __init__(self, env: simpy.Environment, intersection_id: str, 
                 green_time_ns: float = 30.0, green_time_ew: float = 30.0,
                 yellow_time: float = 5.0):
        self.env = env
        self.intersection_id = intersection_id
        self.green_time_ns = green_time_ns
        self.green_time_ew = green_time_ew
        self.yellow_time = yellow_time
        self.current_phase = TrafficPhase.NS_GREEN
        self.phase_start_time = 0.0
        
    def get_current_phase(self) -> TrafficPhase:
        return self.current_phase
    
    def time_in_phase(self) -> float:
        return self.env.now - self.phase_start_time
    
    def switch_phase(self):
        """
        Switch to next phase
        """
        phase_map = {
            TrafficPhase.NS_GREEN: TrafficPhase.NS_YELLOW,
            TrafficPhase.NS_YELLOW: TrafficPhase.EW_GREEN,
            TrafficPhase.EW_GREEN: TrafficPhase.EW_YELLOW,
            TrafficPhase.EW_YELLOW: TrafficPhase.NS_GREEN
        }
        self.current_phase = phase_map[self.current_phase]
        self.phase_start_time = self.env.now

class Approach:
    """
    One approach to an intersection
    """
    def __init__(self, env: simpy.Environment, approach_id: str, saturation_flow: float = 1800.0):
        self.env = env
        self.approach_id = approach_id
        self.saturation_flow = saturation_flow  # vehicles per hour at green
        self.queue = []
        self.vehicles_served = 0
        self.total_delay = 0.0
        self.queue_lengths = []  # for averaging
        
    def add_vehicle(self, vehicle: Vehicle):
        vehicle.queue_position = len(self.queue)
        self.queue.append(vehicle)
        
    def serve_vehicles(self, service_time: float) -> int:
        """
        Serve vehicles during green
        """
        served_count = 0
        service_rate = self.saturation_flow / 3600.0  # vehicles per second
        max_vehicles = int(service_time * service_rate)
        
        for _ in range(min(max_vehicles, len(self.queue))):
            if self.queue:
                vehicle = self.queue.pop(0)
                vehicle.departure_time = self.env.now
                vehicle.delay = vehicle.departure_time - vehicle.arrival_time
                self.total_delay += vehicle.delay
                self.vehicles_served += 1
                served_count += 1
                
        return served_count
    
    def get_queue_length(self) -> int:
        return len(self.queue)
    
    def get_avg_delay(self) -> float:
        return self.total_delay / max(1, self.vehicles_served)

class Intersection:
    """
    Multi-approach intersection
    """
    def __init__(self, env: simpy.Environment, intersection_id: str, 
                 coordinates: Tuple[float, float] = (0.0, 0.0)):
        self.env = env
        self.intersection_id = intersection_id
        self.coordinates = coordinates
        self.traffic_light = TrafficLight(env, intersection_id)
        
        # Four approaches: North, South, East, West
        self.approaches = {
            'north': Approach(env, f"{intersection_id}_north"),
            'south': Approach(env, f"{intersection_id}_south"),
            'east': Approach(env, f"{intersection_id}_east"),
            'west': Approach(env, f"{intersection_id}_west")
        }
        
        self.stats = IntersectionStats()
        self.control_policy = "fixed_time"  # fixed_time, actuated, rl
        
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state for RL
        """
        state = []
        for approach in self.approaches.values():
            state.extend([
                approach.get_queue_length(),
                approach.get_avg_delay(),
                len([v for v in approach.queue if (self.env.now - v.arrival_time) > 60])  # vehicles waiting > 60s
            ])
        
        # Add traffic light state
        state.extend([
            self.traffic_light.current_phase.value,
            self.traffic_light.time_in_phase()
        ])
        
        return np.array(state, dtype=np.float32)
    
    def apply_action(self, action: int):
        """
        Apply RL action
        """
        if action == 0:  # Extend current phase
            pass
        elif action == 1:  # Switch to next phase
            self.traffic_light.switch_phase()
    
    def update_stats(self):
        """
        Update intersection stats
        """
        total_vehicles = sum(app.vehicles_served for app in self.approaches.values())
        total_delay = sum(app.total_delay for app in self.approaches.values())
        avg_queue = np.mean([app.get_queue_length() for app in self.approaches.values()])
        
        self.stats.total_vehicles = total_vehicles
        self.stats.total_delay = total_delay
        self.stats.avg_queue_length = avg_queue
        self.stats.throughput = total_vehicles / max(1, self.env.now / 3600.0)  # vehicles per hour

class TrafficNetwork:
    """
    Network of intersections
    """
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.graph = nx.DiGraph()
        self.intersections = {}
        self.vehicle_id_counter = 0
        
    def add_intersection(self, intersection_id: str, coordinates: Tuple[float, float]):
        """
        Add intersection
        """
        intersection = Intersection(self.env, intersection_id, coordinates)
        self.intersections[intersection_id] = intersection
        self.graph.add_node(intersection_id, intersection=intersection, coordinates=coordinates)
        
    def add_road(self, from_intersection: str, to_intersection: str, 
                 travel_time: float = 60.0, capacity: float = 1000.0):
        """
        Add road
        """
        self.graph.add_edge(from_intersection, to_intersection, 
                           travel_time=travel_time, capacity=capacity)
    
    def generate_vehicle_arrivals(self, intersection_id: str, approach: str, 
                                arrival_rate: float, duration: float):
        """
        Generate Poisson arrivals
        """
        while self.env.now < duration:
            # Poisson process: exponential inter-arrival times
            inter_arrival_time = random.expovariate(arrival_rate / 3600.0)  # rate per second
            yield self.env.timeout(inter_arrival_time)
            
            # Create vehicle
            vehicle = Vehicle(
                vehicle_id=self.vehicle_id_counter,
                arrival_time=self.env.now,
                origin=f"{intersection_id}_{approach}",
                destination="random"
            )
            self.vehicle_id_counter += 1
            
            # Add to intersection approach
            if intersection_id in self.intersections:
                self.intersections[intersection_id].approaches[approach].add_vehicle(vehicle)
    
    def run_fixed_time_control(self, intersection_id: str):
        """
        Fixed-time control
        """
        intersection = self.intersections[intersection_id]
        
        while True:
            current_phase = intersection.traffic_light.get_current_phase()
            
            if current_phase == TrafficPhase.NS_GREEN:
                yield self.env.timeout(intersection.traffic_light.green_time_ns)
                # Serve north-south traffic
                for approach_name in ['north', 'south']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(intersection.traffic_light.green_time_ns)
                    
            elif current_phase == TrafficPhase.EW_GREEN:
                yield self.env.timeout(intersection.traffic_light.green_time_ew)
                # Serve east-west traffic
                for approach_name in ['east', 'west']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(intersection.traffic_light.green_time_ew)
                    
            elif current_phase in [TrafficPhase.NS_YELLOW, TrafficPhase.EW_YELLOW]:
                yield self.env.timeout(intersection.traffic_light.yellow_time)
            
            intersection.traffic_light.switch_phase()
            intersection.update_stats()
    
    def run_actuated_control(self, intersection_id: str):
        """
        Vehicle-actuated control
        """
        intersection = self.intersections[intersection_id]
        
        while True:
            current_phase = intersection.traffic_light.get_current_phase()
            
            if current_phase == TrafficPhase.NS_GREEN:
                # Check for vehicles in north-south approaches
                ns_demand = (intersection.approaches['north'].get_queue_length() + 
                           intersection.approaches['south'].get_queue_length())
                ew_demand = (intersection.approaches['east'].get_queue_length() + 
                           intersection.approaches['west'].get_queue_length())
                
                # Serve for minimum green time, then extend if there's demand
                base_time = intersection.traffic_light.green_time_ns
                if ns_demand > ew_demand and ns_demand > 2:
                    serve_time = min(base_time * 1.3, 40.0)  # 30% longer, max 40s
                elif ew_demand > ns_demand + 2:  # Significant EW demand
                    serve_time = max(base_time * 0.8, 20.0)  # 20% shorter, min 20s
                else:
                    serve_time = base_time
                
                yield self.env.timeout(serve_time)
                for approach_name in ['north', 'south']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(serve_time)
                    
            elif current_phase == TrafficPhase.EW_GREEN:
                # Check for vehicles in east-west approaches
                ns_demand = (intersection.approaches['north'].get_queue_length() + 
                           intersection.approaches['south'].get_queue_length())
                ew_demand = (intersection.approaches['east'].get_queue_length() + 
                           intersection.approaches['west'].get_queue_length())
                
                base_time = intersection.traffic_light.green_time_ew
                if ew_demand > ns_demand and ew_demand > 2:
                    serve_time = min(base_time * 1.3, 40.0)
                elif ns_demand > ew_demand + 2:
                    serve_time = max(base_time * 0.8, 20.0)
                else:
                    serve_time = base_time
                
                yield self.env.timeout(serve_time)
                for approach_name in ['east', 'west']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(serve_time)
                    
            elif current_phase in [TrafficPhase.NS_YELLOW, TrafficPhase.EW_YELLOW]:
                yield self.env.timeout(intersection.traffic_light.yellow_time)
            
            intersection.traffic_light.switch_phase()
            intersection.update_stats()
    
    def run_rl_control(self, intersection_id: str):
        """
        RL-based control
        """
        intersection = self.intersections[intersection_id]
        
        while True:
            current_phase = intersection.traffic_light.get_current_phase()
            
            # Advanced RL-like logic: consider multiple factors
            ns_demand = (intersection.approaches['north'].get_queue_length() + 
                        intersection.approaches['south'].get_queue_length())
            ew_demand = (intersection.approaches['east'].get_queue_length() + 
                        intersection.approaches['west'].get_queue_length())
            
            # Consider waiting times (higher weight for longer waits)
            ns_avg_wait = (intersection.approaches['north'].get_avg_delay() + 
                          intersection.approaches['south'].get_avg_delay()) / 2
            ew_avg_wait = (intersection.approaches['east'].get_avg_delay() + 
                          intersection.approaches['west'].get_avg_delay()) / 2
            
            # Weighted score: queue length + wait time penalty + demand trend
            ns_score = ns_demand + (ns_avg_wait / 20.0)  # Wait time in 20s units
            ew_score = ew_demand + (ew_avg_wait / 20.0)
            
            if current_phase == TrafficPhase.NS_GREEN:
                # Dynamic green time based on intelligent scoring
                if ns_score > ew_score + 1:  # Clear NS advantage
                    serve_time = min(25 + (ns_score - ew_score) * 4, 45.0)
                elif ew_score > ns_score + 1:  # Need to switch soon
                    serve_time = max(15.0, 30 - (ew_score - ns_score) * 2)
                else:  # Balanced demand
                    serve_time = 25.0
                
                yield self.env.timeout(serve_time)
                for approach_name in ['north', 'south']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(serve_time)
                    
            elif current_phase == TrafficPhase.EW_GREEN:
                if ew_score > ns_score + 1:
                    serve_time = min(25 + (ew_score - ns_score) * 4, 45.0)
                elif ns_score > ew_score + 1:
                    serve_time = max(15.0, 30 - (ns_score - ew_score) * 2)
                else:
                    serve_time = 25.0
                
                yield self.env.timeout(serve_time)
                for approach_name in ['east', 'west']:
                    approach = intersection.approaches[approach_name]
                    approach.serve_vehicles(serve_time)
                    
            elif current_phase in [TrafficPhase.NS_YELLOW, TrafficPhase.EW_YELLOW]:
                yield self.env.timeout(intersection.traffic_light.yellow_time)
            
            intersection.traffic_light.switch_phase()
            intersection.update_stats()
    
    def get_network_state(self) -> Dict:
        """
        Get network state
        """
        network_state = {
            'timestamp': self.env.now,
            'intersections': {}
        }
        
        for int_id, intersection in self.intersections.items():
            intersection.update_stats()
            network_state['intersections'][int_id] = {
                'coordinates': intersection.coordinates,
                'queue_lengths': {
                    approach_name: approach.get_queue_length() 
                    for approach_name, approach in intersection.approaches.items()
                },
                'total_delay': intersection.stats.total_delay,
                'throughput': intersection.stats.throughput,
                'current_phase': intersection.traffic_light.current_phase.name,
                'time_in_phase': intersection.traffic_light.time_in_phase()
            }
            
        return network_state
    
    def create_demo_network(self):
        """
        Create demo network
        """
        # Create 4 intersections in a grid
        intersections = [
            ('int_1', (0.0, 0.0)),
            ('int_2', (1.0, 0.0)),
            ('int_3', (0.0, 1.0)),
            ('int_4', (1.0, 1.0))
        ]
        
        for int_id, coords in intersections:
            self.add_intersection(int_id, coords)
        
        # Connect intersections with roads
        roads = [
            ('int_1', 'int_2'),
            ('int_2', 'int_1'),
            ('int_1', 'int_3'),
            ('int_3', 'int_1'),
            ('int_2', 'int_4'),
            ('int_4', 'int_2'),
            ('int_3', 'int_4'),
            ('int_4', 'int_3')
        ]
        
        for from_int, to_int in roads:
            self.add_road(from_int, to_int)

class TrafficSimulation:
    """
    Main simulation controller
    """
    def __init__(self, random_seed: int = 42):
        self.env = simpy.Environment()
        self.network = TrafficNetwork(self.env)
        self.random_seed = random_seed
        self.simulation_results = []
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def setup_scenario(self, scenario_type: str = "demo", control_policy: str = "Fixed Time"):
        """
        Setup simulation scenario
        """
        if scenario_type == "demo":
            self.network.create_demo_network()
            
            # Start traffic light controllers for all intersections
            for int_id in self.network.intersections.keys():
                # Set control policy for each intersection
                self.network.intersections[int_id].control_policy = control_policy.lower().replace(" ", "_")
                
                # Start appropriate control process
                if control_policy == "Fixed Time":
                    self.env.process(self.network.run_fixed_time_control(int_id))
                elif control_policy == "Actuated":
                    self.env.process(self.network.run_actuated_control(int_id))
                elif control_policy == "RL-based":
                    self.env.process(self.network.run_rl_control(int_id))
                else:
                    # Default to fixed time
                    self.env.process(self.network.run_fixed_time_control(int_id))
            
            # Generate traffic arrivals for all approaches
            arrival_rates = {
                'peak': 400,    # vehicles per hour
                'normal': 200,
                'low': 100
            }
            
            for int_id in self.network.intersections.keys():
                for approach in ['north', 'south', 'east', 'west']:
                    rate = arrival_rates['normal']  # Default to normal traffic
                    self.env.process(
                        self.network.generate_vehicle_arrivals(int_id, approach, rate, 3600)
                    )
    
    def run_simulation(self, duration: float = 3600.0, collect_interval: float = 60.0):
        """
        Run simulation and collect results
        """
        def collect_data():
            while self.env.now < duration:
                yield self.env.timeout(collect_interval)
                state = self.network.get_network_state()
                self.simulation_results.append(state)
        
        # Start data collection
        self.env.process(collect_data())
        
        # Run simulation
        self.env.run(until=duration)
        
        return self.simulation_results
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        """
        if not self.simulation_results:
            return {}
        
        total_delay = 0
        total_throughput = 0
        avg_queue_lengths = []
        
        for result in self.simulation_results:
            for int_data in result['intersections'].values():
                total_delay += int_data['total_delay']
                total_throughput += int_data['throughput']
                avg_queue_lengths.extend(int_data['queue_lengths'].values())
        
        return {
            'total_network_delay': total_delay,
            'average_throughput': total_throughput / len(self.simulation_results) if self.simulation_results else 0,
            'average_queue_length': np.mean(avg_queue_lengths) if avg_queue_lengths else 0,
            'simulation_duration': self.simulation_results[-1]['timestamp'] if self.simulation_results else 0
        }
