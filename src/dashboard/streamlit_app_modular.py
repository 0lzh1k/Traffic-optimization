"""
Streamlit dashboard for traffic optimization - Main App
"""
import os
import sys
import streamlit as st

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import components
from src.dashboard.components.overview import show_overview
from src.dashboard.components.realtime_control import show_realtime_control
from src.dashboard.components.prediction import show_prediction_models
from src.dashboard.components.simulation import show_simulation
from src.dashboard.components.rl_training import show_rl_training
from src.dashboard.components.visualization import show_network_visualization
from src.dashboard.components.comparison import show_policy_comparison

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Flow Optimization",
    page_icon=None,  # Removed emoji icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None


def main():
    st.title("Smart Traffic Flow Optimization")
    st.markdown("**Urban Planning Dashboard for Traffic Signal Control**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Traffic Simulation", "RL Training", "Prediction Models", 
         "Network Visualization", "Policy Comparison", "Real-time Control"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Traffic Simulation":
        show_simulation()
    elif page == "RL Training":
        show_rl_training()
    elif page == "Prediction Models":
        show_prediction_models()
    elif page == "Network Visualization":
        show_network_visualization()
    elif page == "Policy Comparison":
        show_policy_comparison()
    elif page == "Real-time Control":
        show_realtime_control()


# Now all components are available via imports above

if __name__ == "__main__":
    main()
