"""
Traffic Data Utilities

Shared utility functions for generating and processing traffic data
across different dashboard components.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_live_traffic_data():
    """Generate simulated live traffic data with time-based variation"""
    
    current_time = datetime.now()
    
    # Generate data for last 30 minutes
    times = [current_time - timedelta(minutes=30-i*5) for i in range(7)]
    
    data = {
        'times': times,
        'intersections': {}
    }
    
    # Use current time and refresh counter to create truly dynamic variation
    refresh_counter = st.session_state.get('refresh_counter', 0)
    time_seed = int(current_time.timestamp()) + refresh_counter  # Include refresh counter
    np.random.seed(time_seed)
    
    for int_id in ['int_1', 'int_2', 'int_3', 'int_4']:
        # Simulate traffic patterns based on hour
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        
        # Create time-based traffic patterns with more variation
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
            base_flow = 300 + 50 * np.sin(minute * np.pi / 30) + 20 * np.sin(second * np.pi / 30)
        else:
            base_flow = 150 + 30 * np.sin(minute * np.pi / 30) + 10 * np.sin(second * np.pi / 30)
        
        # Add intersection-specific variation
        intersection_multipliers = {'int_1': 1.2, 'int_2': 0.9, 'int_3': 1.1, 'int_4': 0.8}
        base_flow *= intersection_multipliers.get(int_id, 1.0)
        
        # Add refresh-based variation to make data truly dynamic
        refresh_variation = np.sin(refresh_counter * 0.5) * 30
        base_flow += refresh_variation
        
        # Generate time series with realistic variation
        flows = []
        queue_lengths = []
        delays = []
        
        for i, t in enumerate(times):
            # Time-based variation for each data point
            time_var = np.sin((t.minute + i*5) * np.pi / 30) * 20
            flow = max(50, base_flow + time_var + np.random.normal(0, 15))
            flows.append(flow)
            
            # Queue length related to flow (higher flow = longer queues)
            queue = max(0, (flow - 200) / 20 + np.random.normal(0, 3))
            queue_lengths.append(queue)
            
            # Delay related to queue length
            delay = max(10, queue * 5 + np.random.normal(0, 10))
            delays.append(delay)
        
        # Current phase based on time and emergency overrides
        phase_cycle_position = (current_time.minute * 60 + current_time.second) % 120  # 2-minute cycle
        
        # Check for emergency overrides
        if 'emergency_state' in st.session_state and st.session_state.emergency_state:
            if st.session_state.emergency_state.get('flash_active'):
                current_phase = 'EMERGENCY_FLASH'
                phase_time_remaining = 0  # Flash mode
            elif st.session_state.emergency_state.get('all_green_ns_active'):
                current_phase = 'NS_GREEN_EMERGENCY'
                phase_time_remaining = 999  # Indefinite
            else:
                # Normal phase logic
                if phase_cycle_position < 50:
                    current_phase = 'NS_GREEN'
                    phase_time_remaining = 50 - phase_cycle_position
                elif phase_cycle_position < 60:
                    current_phase = 'NS_YELLOW' 
                    phase_time_remaining = 60 - phase_cycle_position
                elif phase_cycle_position < 110:
                    current_phase = 'EW_GREEN'
                    phase_time_remaining = 110 - phase_cycle_position
                else:
                    current_phase = 'EW_YELLOW'
                    phase_time_remaining = 120 - phase_cycle_position
        else:
            # Normal phase logic when no emergency state
            if phase_cycle_position < 50:
                current_phase = 'NS_GREEN'
                phase_time_remaining = 50 - phase_cycle_position
            elif phase_cycle_position < 60:
                current_phase = 'NS_YELLOW' 
                phase_time_remaining = 60 - phase_cycle_position
            elif phase_cycle_position < 110:
                current_phase = 'EW_GREEN'
                phase_time_remaining = 110 - phase_cycle_position
            else:
                current_phase = 'EW_YELLOW'
                phase_time_remaining = 120 - phase_cycle_position
        
        # Check for manual overrides per intersection
        if 'control_overrides' in st.session_state and int_id in st.session_state.control_overrides:
            override = st.session_state.control_overrides[int_id]
            if override.get('active') and override.get('phase') != 'AUTO':
                current_phase = f"{override['phase']}_OVERRIDE"
                phase_time_remaining = 999  # Manual override
        
        data['intersections'][int_id] = {
            'flows': flows,
            'queue_lengths': queue_lengths,
            'delays': delays,
            'current_phase': current_phase,
            'phase_time_remaining': int(phase_time_remaining)
        }
    
    return data


def display_live_traffic_chart(live_data):
    """Display live traffic flow chart"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Traffic Flow (vehicles/5min)', 'Average Queue Length'),
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (int_id, int_data) in enumerate(live_data['intersections'].items()):
        # Traffic flow
        fig.add_trace(
            go.Scatter(
                x=live_data['times'],
                y=int_data['flows'],
                mode='lines+markers',
                name=f'{int_id} Flow',
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=1, col=1
        )
        
        # Queue lengths
        fig.add_trace(
            go.Scatter(
                x=live_data['times'],
                y=int_data['queue_lengths'],
                mode='lines+markers',
                name=f'{int_id} Queue',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500, 
        title_text=f"Real-time Traffic Monitoring (Updated: {datetime.now().strftime('%H:%M:%S')})"
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Vehicles", row=1, col=1)
    fig.update_yaxes(title_text="Queue Length", row=2, col=1)
    
    # Use unique key based on refresh counter to force chart updates
    chart_key = f"live_traffic_chart_{st.session_state.get('refresh_counter', 0)}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def display_intersection_status(live_data):
    """Display current intersection status"""
    
    st.subheader("Current Intersection Status")
    
    for int_id, int_data in live_data['intersections'].items():
        with st.expander(f"Intersection {int_id}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Flow",
                    f"{int_data['flows'][-1]:.0f} veh/5min"
                )
            
            with col2:
                st.metric(
                    "Queue Length",
                    f"{int_data['queue_lengths'][-1]:.0f} vehicles"
                )
            
            with col3:
                st.metric(
                    "Avg Delay",
                    f"{int_data['delays'][-1]:.0f} seconds"
                )
            
            # Phase information
            phase_colors = {
                'NS_GREEN': '🟢',
                'EW_GREEN': '🟢',
                'NS_YELLOW': '🟡',
                'EW_YELLOW': '🟡',
                'EMERGENCY_FLASH': '🔴',
                'NS_GREEN_EMERGENCY': '🚨',
                'NS_GREEN_OVERRIDE': '🔧',
                'EW_GREEN_OVERRIDE': '🔧'
            }
            
            phase_emoji = phase_colors.get(int_data['current_phase'], '⚪')
            
            # Display phase with special formatting for emergency/override modes
            phase_display = int_data['current_phase']
            if 'EMERGENCY' in phase_display:
                st.write(f"**🚨 Emergency Phase:** {phase_emoji} {phase_display}")
            elif 'OVERRIDE' in phase_display:
                st.write(f"**🔧 Manual Override:** {phase_emoji} {phase_display}")
            else:
                st.write(f"**Current Phase:** {phase_emoji} {phase_display}")
            
            # Time remaining display
            if int_data['phase_time_remaining'] == 999:
                st.write("**Time Remaining:** Manual control")
            elif int_data['phase_time_remaining'] == 0:
                st.write("**Time Remaining:** Flash mode")
            else:
                st.write(f"**Time Remaining:** {int_data['phase_time_remaining']} seconds")
            
            # Progress bar for phase timing (only for normal phases)
            if int_data['phase_time_remaining'] not in [0, 999]:
                progress = 1 - (int_data['phase_time_remaining'] / 60)  # Assume 60s max phase
                st.progress(max(0, min(1, progress)))
            elif int_data['phase_time_remaining'] == 0:
                # Flash mode - show blinking effect
                st.progress(0.5)
                st.caption("🔴 Emergency Flash Mode")
            else:
                # Manual control
                st.progress(1.0)
                st.caption("🔧 Manual Control Active")


def display_network_performance_alerts(live_data):
    """Display network-wide performance alerts and summary"""
    
    st.subheader("Network Performance")
    
    # Calculate network-wide metrics
    total_flow = sum(int_data['flows'][-1] for int_data in live_data['intersections'].values())
    avg_queue = np.mean([int_data['queue_lengths'][-1] for int_data in live_data['intersections'].values()])
    avg_delay = np.mean([int_data['delays'][-1] for int_data in live_data['intersections'].values()])
    
    # Network metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Network Flow",
            f"{total_flow:.0f} veh/5min",
            delta=f"{np.random.uniform(-5, 10):.1f}" if np.random.random() > 0.5 else None
        )
    
    with col2:
        queue_status = "🟢" if avg_queue < 5 else "🟡" if avg_queue < 10 else "🔴"
        st.metric(
            "Avg Queue Length",
            f"{avg_queue:.1f} veh",
            delta=f"{np.random.uniform(-1, 2):.1f}" if np.random.random() > 0.5 else None
        )
        st.caption(f"{queue_status} Queue Status")
    
    with col3:
        delay_status = "🟢" if avg_delay < 30 else "🟡" if avg_delay < 60 else "🔴"
        st.metric(
            "Avg Delay",
            f"{avg_delay:.1f} sec",
            delta=f"{np.random.uniform(-3, 5):.1f}" if np.random.random() > 0.5 else None
        )
        st.caption(f"{delay_status} Delay Status")
    
    with col4:
        efficiency = max(60, 100 - avg_delay * 0.8 - avg_queue * 2)
        eff_status = "🟢" if efficiency > 85 else "🟡" if efficiency > 70 else "🔴"
        st.metric(
            "Network Efficiency",
            f"{efficiency:.1f}%",
            delta=f"{np.random.uniform(-2, 3):.1f}%" if np.random.random() > 0.5 else None
        )
        st.caption(f"{eff_status} Efficiency")
    
    # Performance alerts
    alerts = []
    
    if avg_queue > 10:
        alerts.append("High queue lengths detected")
    if avg_delay > 60:
        alerts.append("Excessive delays detected")
    if total_flow > 1000:
        alerts.append("High traffic volume")
    
    # Critical intersection detection
    critical_intersections = []
    for int_id, int_data in live_data['intersections'].items():
        if int_data['queue_lengths'][-1] > 15 or int_data['delays'][-1] > 90:
            critical_intersections.append(int_id)
    
    if critical_intersections:
        alerts.append(f"Critical congestion at: {', '.join(critical_intersections)}")
    
    if alerts:
        st.subheader("⚠️ Active Alerts")
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ No active performance alerts")
