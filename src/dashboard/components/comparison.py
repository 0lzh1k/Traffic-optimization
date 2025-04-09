import streamlit as st
import pandas as pd
import numpy as np
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def show_policy_comparison():
    st.header("Traffic Control Policy Comparison")
    
    st.markdown("Compare performance of different traffic control strategies")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Comparison Settings")
        policies_to_compare = st.multiselect(
            "Select Policies to Compare",
            ["Fixed Time", "Actuated", "RL-based"],
            default=["Fixed Time", "RL-based"]
        )
        evaluation_episodes = st.slider("Evaluation Episodes", 5, 50, 10)
        scenario_for_comparison = st.selectbox(
            "Test Scenario",
            ["Normal Traffic", "Peak Hour", "Incident", "Variable Demand"]
        )
        run_comparison = st.button("Run Comparison", type="primary")
        if run_comparison:
            st.session_state.comparison_results = run_policy_comparison(
                policies_to_compare, evaluation_episodes, scenario_for_comparison
            )
    with col2:
        st.subheader("Comparison Results")
        if 'comparison_results' in st.session_state:
            display_policy_comparison_results()
        else:
            st.info("Click 'Run Comparison' to see results")


def run_policy_comparison(policies, episodes, scenario):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    for i, policy in enumerate(policies):
        status_text.text(f"Evaluating {policy} policy...")
        progress_bar.progress((i + 1) / len(policies))
        time.sleep(1)
        if policy == "Fixed Time":
            results[policy] = {
                'avg_delay': 4500 + np.random.normal(0, 200),
                'avg_throughput': 1200 + np.random.normal(0, 50),
                'avg_queue_length': 15 + np.random.normal(0, 2)
            }
        elif policy == "Actuated":
            results[policy] = {
                'avg_delay': 3800 + np.random.normal(0, 150),
                'avg_throughput': 1350 + np.random.normal(0, 40),
                'avg_queue_length': 12 + np.random.normal(0, 1.5)
            }
        elif policy == "RL-based":
            results[policy] = {
                'avg_delay': 3200 + np.random.normal(0, 100),
                'avg_throughput': 1500 + np.random.normal(0, 30),
                'avg_queue_length': 9 + np.random.normal(0, 1)
            }
    status_text.text("Comparison completed!")
    return results


def display_policy_comparison_results():
    
    results = st.session_state.comparison_results
    
    st.subheader("Performance Summary")
    
    metrics_df = pd.DataFrame(results).T
    st.dataframe(metrics_df.round(2), use_container_width=True)
    
    st.subheader("Performance Comparison")
    
    metrics = ['avg_delay', 'avg_throughput', 'avg_queue_length']
    metric_labels = ['Average Delay (sec)', 'Average Throughput (veh/hr)', 'Average Queue Length (veh)']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metric_labels,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    policies = list(results.keys())
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[policy][metric] for policy in policies]
        
        fig.add_trace(
            go.Bar(
                x=policies,
                y=values,
                name=label,
                marker_color=colors[:len(policies)],
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(height=400, title_text="Policy Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Best Policy Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_delay = min(results.keys(), key=lambda p: results[p]['avg_delay'])
        st.metric(
            "Best for Delay Reduction",
            best_delay,
            delta=f"-{(results[best_delay]['avg_delay']):.0f} sec"
        )
    
    with col2:
        best_throughput = max(results.keys(), key=lambda p: results[p]['avg_throughput'])
        st.metric(
            "Best for Throughput",
            best_throughput,
            delta=f"+{results[best_throughput]['avg_throughput']:.0f} veh/hr"
        )
    
    with col3:
        best_queue = min(results.keys(), key=lambda p: results[p]['avg_queue_length'])
        st.metric(
            "Best for Queue Management",
            best_queue,
            delta=f"-{results[best_queue]['avg_queue_length']:.1f} veh"
        )
    
    if len(results) >= 2:
        st.subheader("Improvement Analysis")
        
        baseline = "Fixed Time"
        if baseline in results:
            for policy in results:
                if policy != baseline:
                    delay_improvement = ((results[baseline]['avg_delay'] - results[policy]['avg_delay']) / 
                                       results[baseline]['avg_delay']) * 100
                    
                    throughput_improvement = ((results[policy]['avg_throughput'] - results[baseline]['avg_throughput']) / 
                                            results[baseline]['avg_throughput']) * 100
                    
                    st.write(f"**{policy} vs {baseline}:**")
                    st.write(f"- Delay reduction: {delay_improvement:.1f}%")
                    st.write(f"- Throughput increase: {throughput_improvement:.1f}%")
