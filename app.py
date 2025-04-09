"""
Smart Traffic Flow Optimization - Main Application Entry Point
"""

import streamlit as st
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)
from src.dashboard.streamlit_app_modular import main

if __name__ == "__main__":
    main()
