import streamlit as st
import time
import sys
import os
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.traffic_data import (
    generate_live_traffic_data,
    display_live_traffic_chart,
    display_intersection_status,
    display_network_performance_alerts
)


def show_realtime_control():
    """Main real-time control dashboard"""
    st.header("Real-time Traffic Control")
    
    st.markdown("Monitor and control traffic signals in real-time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Traffic Status")
        
        col_auto, col_manual = st.columns([1, 1])
        
        with col_auto:
            auto_refresh = st.checkbox("Auto-refresh (5 seconds)")
        
        with col_manual:
            manual_refresh = st.button("Refresh Now")
        
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        if 'auto_refresh_active' not in st.session_state:
            st.session_state.auto_refresh_active = False
        
        st.session_state.auto_refresh_active = auto_refresh
        
        if manual_refresh:
            st.session_state.refresh_counter += 1
            st.session_state.last_refresh_time = time.time()
        
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        
        current_time_seconds = time.time()
        
        should_refresh = manual_refresh
        if auto_refresh and (current_time_seconds - st.session_state.last_refresh_time >= 5):
            should_refresh = True
            st.session_state.refresh_counter += 1
            st.session_state.last_refresh_time = current_time_seconds
        
        current_time = datetime.now()
        live_data = generate_live_traffic_data()
        
        with metrics_placeholder.container():
            st.write(f"**Last Updated:** {current_time.strftime('%H:%M:%S')}")
            if auto_refresh:
                st.write(f"**Auto-refresh:** Enabled")
                st.write(f"**Refresh Count:** {st.session_state.refresh_counter}")
                time_since_refresh = current_time_seconds - st.session_state.last_refresh_time
                next_refresh_in = max(0, 5 - time_since_refresh)
                st.write(f"**Next refresh in:** {next_refresh_in:.1f}s")
            else:
                st.write(f"**Auto-refresh:** Disabled")
        
        with chart_placeholder.container():
            display_live_traffic_chart(live_data)
        
        with status_placeholder.container():
            display_intersection_status(live_data)
            display_network_performance_alerts(live_data)
        
        if auto_refresh:
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.subheader("Control Panel")

        st.write("**Manual Override**")

        selected_intersection = st.selectbox(
            "Select Intersection",
            ["int_1", "int_2", "int_3", "int_4"]
        )

        manual_phase = st.selectbox(
            "Force Phase",
            ["NS_GREEN", "EW_GREEN", "AUTO"]
        )

        if st.button("Apply Override"):
            if 'control_overrides' not in st.session_state:
                st.session_state.control_overrides = {}
            st.session_state.control_overrides[selected_intersection] = {
                'phase': manual_phase,
                'timestamp': datetime.now(),
                'active': True
            }
            st.success(f"Phase override applied to {selected_intersection}: {manual_phase}")

        if 'control_overrides' in st.session_state:
            active_overrides = [k for k, v in st.session_state.control_overrides.items() if v.get('active', False)]
            if active_overrides:
                st.info(f"Active overrides: {', '.join(active_overrides)}")

        st.markdown("---")

        if 'emergency_state' in st.session_state and st.session_state.emergency_state:
            active_modes = []
            for key, value in st.session_state.emergency_state.items():
                if key.endswith('_active') and value:
                    mode_name = key.replace('_active', '')
                    start_key = f"{mode_name}_start"
                    if start_key in st.session_state.emergency_state:
                        duration = datetime.now() - st.session_state.emergency_state[start_key]
                        active_modes.append(f"{mode_name.replace('_', ' ').title()}: Active for {duration.seconds}s")
            if active_modes:
                st.warning("**Emergency Mode Active**")
                for mode_info in active_modes:
                    st.caption(f"- {mode_info}")

        st.write("**Emergency Controls**")

        col1, col2_2 = st.columns(2)

        with col1:
            if st.button("Emergency Flash", type="secondary", use_container_width=True):
                if 'emergency_state' not in st.session_state:
                    st.session_state.emergency_state = {}
                st.session_state.emergency_state['flash_active'] = True
                st.session_state.emergency_state['flash_start'] = datetime.now()
                st.warning("Emergency flash activated for all intersections")

        with col2_2:
            if st.button("All Green NS", type="secondary", use_container_width=True):
                if 'emergency_state' not in st.session_state:
                    st.session_state.emergency_state = {}
                st.session_state.emergency_state['all_green_ns_active'] = True
                st.session_state.emergency_state['all_green_ns_start'] = datetime.now()
                st.info("All North-South phases set to green")

        if st.button("Reset to Auto", type="primary", use_container_width=True):
            if 'control_overrides' in st.session_state:
                st.session_state.control_overrides = {}
            if 'emergency_state' in st.session_state:
                st.session_state.emergency_state = {}
            st.success("All intersections reset to automatic control")

        st.markdown("---")

        st.write("**System Status**")

        system_health = np.random.choice(["Optimal", "Good", "Warning"], p=[0.7, 0.2, 0.1])

        if system_health == "Optimal":
            st.success("System Optimal")
        elif system_health == "Good":
            st.info("System Good")
        else:
            st.warning("System Warning")

        st.metric("Active Intersections", "4/4")

        import random
        efficiency = round(85 + random.uniform(0, 10), 1)
        st.metric("Network Efficiency", f"{efficiency}%")
