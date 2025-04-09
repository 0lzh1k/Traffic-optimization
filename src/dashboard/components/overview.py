import streamlit as st
import plotly.graph_objects as go


def show_overview():
    """Display the project overview page"""
    st.header("Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Smart Traffic Flow Optimization for Urban Planning
        
        This project aims to reduce urban congestion by predicting traffic patterns and 
        optimizing control policies using machine learning and reinforcement learning.
        
        **Key Features:**
        - **Multi-intersection traffic simulation** with realistic arrival patterns
        - **Machine Learning prediction** of short-term traffic states
        - **Reinforcement Learning optimization** of signal timing policies
        - **Interactive dashboard** for scenario analysis and policy comparison
        - **Real-time visualization** of traffic flows and performance metrics
        
        **Technologies Used:**
        - **Simulation**: SimPy, NetworkX
        - **ML/RL**: Scikit-learn, Stable-baselines3
        - **Visualization**: Streamlit, Plotly, Folium
        """)
    
    with col2:
        st.markdown("""
        ### Quick Stats
        """)
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Intersections", "4", delta="Demo Network")
            st.metric("Avg Delay Reduction", "25%", delta="vs Fixed-time")
        with col2_2:
            st.metric("Training Episodes", "1000", delta="RL Model")
            st.metric("Prediction Accuracy", "85%", delta="15-min forecast")
    st.header("System Workflow")
    workflow_steps = [
        "1. **Load/Define Road Network** - Create intersection and road topology",
        "2. **Generate Traffic Arrivals** - Poisson processes or historical data",
        "3. **Run SimPy Simulation** - Discrete event simulation with state collection",
        "4. **Train/Evaluate Policies** - Fixed-time, actuated, or RL-based control",
        "5. **Visualize Results** - KPIs, heatmaps, and performance comparisons"
    ]
    for step in workflow_steps:
        st.markdown(step)
    st.header("System Architecture")
    fig = go.Figure()
    components = [
        {"name": "Traffic\nSimulation", "x": 1, "y": 3, "color": "#1f77b4"},
        {"name": "RL Agent", "x": 3, "y": 3, "color": "#ff7f0e"},
        {"name": "Prediction\nModel", "x": 2, "y": 1, "color": "#2ca02c"},
        {"name": "Dashboard", "x": 4, "y": 2, "color": "#d62728"}
    ]
    for comp in components:
        fig.add_shape(
            type="rect",
            x0=comp["x"]-0.4, y0=comp["y"]-0.3,
            x1=comp["x"]+0.4, y1=comp["y"]+0.3,
            fillcolor=comp["color"],
            opacity=0.3,
            line=dict(color=comp["color"], width=2)
        )
        fig.add_annotation(
            x=comp["x"], y=comp["y"],
            text=comp["name"],
            showarrow=False,
            font=dict(size=12, color="white")
        )
    arrows = [
        (1, 3, 3, 3),
        (2, 1, 1, 3),
        (3, 3, 4, 2),
        (2, 1, 4, 2)
    ]
    for arrow in arrows:
        fig.add_annotation(
            x=arrow[2], y=arrow[3],
            ax=arrow[0], ay=arrow[1],
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor="gray"
        )
    fig.update_layout(
        title="System Component Interaction",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
