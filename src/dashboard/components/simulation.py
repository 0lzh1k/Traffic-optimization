import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from simulation.traffic_environment import TrafficSimulation


def show_simulation():
    """Main traffic simulation page"""
    st.header("Traffic Simulation")
    
    st.markdown("Configure and run traffic simulation scenarios")
    
    # Simulation configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        scenario_type = st.selectbox(
            "Scenario Type",
            ["Demo Network", "Peak Hour", "Incident", "Custom"]
        )
        
        duration = st.slider("Simulation Duration (hours)", 0.5, 4.0, 1.0, 0.5)
        
        st.subheader("Traffic Parameters")
        arrival_rate = st.slider("Base Arrival Rate (veh/hour)", 50, 800, 200, 50)
        peak_multiplier = st.slider("Peak Hour Multiplier", 1.0, 3.0, 1.5, 0.1)
        
        control_policy = st.selectbox(
            "Control Policy",
            ["Fixed Time", "Actuated", "RL-based"]
        )
        
        run_simulation = st.button("🚀 Run Simulation", type="primary")
    
    with col2:
        st.subheader("Simulation Results")
        
        if run_simulation:
            with st.spinner("Running traffic simulation..."):
                # Run simulation
                results = run_traffic_simulation(
                    scenario_type, duration, arrival_rate, 
                    peak_multiplier, control_policy
                )
                st.session_state.simulation_results = results
        
        if st.session_state.simulation_results:
            display_simulation_results(st.session_state.simulation_results)


def run_traffic_simulation(scenario_type, duration, arrival_rate, peak_multiplier, control_policy):
    """Run traffic simulation with given parameters"""
    
    # Create simulation
    sim = TrafficSimulation(random_seed=42)
    sim.setup_scenario("demo", control_policy)
    
    # Modify arrival rates based on parameters
    for int_id in sim.network.intersections.keys():
        for approach in ['north', 'south', 'east', 'west']:
            # Apply peak multiplier during certain hours
            rate = arrival_rate
            if scenario_type == "Peak Hour":
                rate = arrival_rate * peak_multiplier
            
            # Generate new arrivals with updated rate
            sim.env.process(
                sim.network.generate_vehicle_arrivals(
                    int_id, approach, rate, duration * 3600
                )
            )
    
    # Run simulation
    duration_seconds = duration * 3600
    results = sim.run_simulation(duration_seconds, collect_interval=300)  # 5-minute intervals
    
    # Calculate performance metrics
    metrics = sim.get_performance_metrics()
    
    return {
        'results': results,
        'metrics': metrics,
        'parameters': {
            'scenario_type': scenario_type,
            'duration': duration,
            'arrival_rate': arrival_rate,
            'peak_multiplier': peak_multiplier,
            'control_policy': control_policy
        }
    }


def display_simulation_results(simulation_data):
    """Display simulation results with visualizations"""
    
    results = simulation_data['results']
    metrics = simulation_data['metrics']
    params = simulation_data['parameters']
    
    # Key metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Network Delay",
            f"{metrics.get('total_network_delay', 0):.1f} sec",
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Throughput",
            f"{metrics.get('average_throughput', 0):.1f} veh/hr",
            delta=None
        )
    
    with col3:
        st.metric(
            "Average Queue Length",
            f"{metrics.get('average_queue_length', 0):.1f} vehicles",
            delta=None
        )
    
    with col4:
        st.metric(
            "Simulation Duration",
            f"{metrics.get('simulation_duration', 0)/3600:.1f} hours",
            delta=None
        )
    
    # Time series plots
    if results:
        st.subheader("Traffic Flow Over Time")
        
        # Prepare data for plotting
        timestamps = [r['timestamp']/3600 for r in results]  # Convert to hours
        
        # Aggregate queue lengths across all intersections
        total_queues = []
        total_delays = []
        
        for result in results:
            queue_sum = 0
            delay_sum = 0
            
            for int_data in result['intersections'].values():
                queue_sum += sum(int_data['queue_lengths'].values())
                delay_sum += int_data['total_delay']
            
            total_queues.append(queue_sum)
            total_delays.append(delay_sum)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Queue Length Over Time', 'Cumulative Delay Over Time'),
            vertical_spacing=0.1
        )
        
        # Queue length plot
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=total_queues,
                mode='lines+markers',
                name='Queue Length',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Delay plot
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=total_delays,
                mode='lines+markers',
                name='Cumulative Delay',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Time (hours)")
        fig.update_yaxes(title_text="Vehicles", row=1, col=1)
        fig.update_yaxes(title_text="Seconds", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Intersection-level analysis
        st.subheader("Per-Intersection Analysis")
        
        if results:
            latest_result = results[-1]
            intersection_data = []
            
            for int_id, int_info in latest_result['intersections'].items():
                intersection_data.append({
                    'Intersection': int_id,
                    'Total Queue': sum(int_info['queue_lengths'].values()),
                    'Total Delay': int_info['total_delay'],
                    'Throughput': int_info['throughput'],
                    'Current Phase': int_info['current_phase']
                })
            
            df = pd.DataFrame(intersection_data)
            
            # Bar charts for intersection comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Intersection', y='Total Queue',
                           title='Queue Length by Intersection')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Intersection', y='Throughput',
                           title='Throughput by Intersection')
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Detailed Results")
            st.dataframe(df, use_container_width=True)
