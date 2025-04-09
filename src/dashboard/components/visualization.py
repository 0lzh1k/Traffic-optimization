import streamlit as st
import folium
from streamlit_folium import folium_static
import numpy as np


def show_network_visualization():
    """Display network visualization section"""
    st.header("Network Visualization")
    
    st.markdown("Interactive map visualization of traffic network and flows")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Map Settings")
        visualization_type = st.selectbox(
            "Visualization Type",
            ["Traffic Flow", "Queue Lengths", "Delay Heatmap", "Signal Status"]
        )
        show_intersections = st.checkbox("Show Intersections", value=True)
        show_roads = st.checkbox("Show Roads", value=True)
        show_traffic = st.checkbox("Show Traffic Flow", value=True)
        if st.session_state.simulation_results:
            time_step = st.slider(
                "Time Step",
                0, len(st.session_state.simulation_results['results']) - 1,
                0
            )
        else:
            time_step = 0
        refresh_map = st.button("Refresh Map")
    with col2:
        create_traffic_map(visualization_type, show_intersections, 
                         show_roads, show_traffic, time_step)


def create_traffic_map(viz_type, show_intersections, show_roads, show_traffic, time_step):
    """Create interactive traffic network map"""
    
    center_lat, center_lon = 51.1694, 71.4491
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    intersections = {
        'int_1': {'lat': 51.1694, 'lon': 71.4491, 'name': 'Dostyk St & Turan Ave'},
        'int_2': {'lat': 51.1704, 'lon': 71.4501, 'name': 'Dostyk St & Orynbor St'},
        'int_3': {'lat': 51.1684, 'lon': 71.4481, 'name': 'Kabanbay Batyr Ave & Turan Ave'},
        'int_4': {'lat': 51.1674, 'lon': 71.4471, 'name': 'Kabanbay Batyr Ave & Orynbor St'}
    }
    if show_intersections:
        for int_id, info in intersections.items():
            if st.session_state.simulation_results and time_step < len(st.session_state.simulation_results['results']):
                result = st.session_state.simulation_results['results'][time_step]
                int_data = result['intersections'].get(int_id, {})
                queue_total = sum(int_data.get('queue_lengths', {}).values())
                phase = int_data.get('current_phase', 'Unknown')
                delay = int_data.get('total_delay', 0)
                if queue_total > 20:
                    color = 'red'
                elif queue_total > 10:
                    color = 'orange'
                else:
                    color = 'green'
                popup_text = f"""
                <b>{info['name']}</b><br>
                Queue Length: {queue_total} vehicles<br>
                Current Phase: {phase}<br>
                Total Delay: {delay:.1f} seconds
                """
            else:
                color = 'blue'
                popup_text = f"<b>{info['name']}</b><br>No data available"
            folium.CircleMarker(
                location=[info['lat'], info['lon']],
                radius=10,
                popup=popup_text,
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    if show_roads:
        roads = [
            (['int_1', 'int_2'], 'Main Street'),
            (['int_3', 'int_4'], 'Broadway'),
            (['int_1', 'int_3'], '1st Avenue'),
            (['int_2', 'int_4'], '2nd Avenue')
        ]
        for road, name in roads:
            if len(road) >= 2:
                start_int = intersections[road[0]]
                end_int = intersections[road[1]]
                folium.PolyLine(
                    locations=[
                        [start_int['lat'], start_int['lon']],
                        [end_int['lat'], end_int['lon']]
                    ],
                    color='gray',
                    weight=3,
                    opacity=0.7,
                    popup=name
                ).add_to(m)
    if show_traffic and viz_type == "Traffic Flow":
        for int_id, info in intersections.items():
            approaches = [
                (info['lat'] + 0.001, info['lon'], 'North'),
                (info['lat'] - 0.001, info['lon'], 'South'),
                (info['lat'], info['lon'] + 0.001, 'East'),
                (info['lat'], info['lon'] - 0.001, 'West')
            ]
            for lat, lon, direction in approaches:
                flow_intensity = np.random.uniform(0.3, 1.0)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=flow_intensity * 8,
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=flow_intensity,
                    popup=f"{direction} approach flow"
                ).add_to(m)
    folium_static(m, width=800, height=500)
    st.markdown("""
    **Legend:**
    - Red: High congestion (>20 vehicles in queue)
    - Orange: Medium congestion (10-20 vehicles)
    - Green: Low congestion (<10 vehicles)
    - Circle size indicates traffic flow intensity
    """)
