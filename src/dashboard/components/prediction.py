import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from prediction.traffic_prediction import TrafficFlowPredictor, generate_synthetic_traffic_data


def show_prediction_models():
    """Main prediction models page"""
    st.header("Traffic Flow Prediction")
    
    st.markdown("Train and evaluate short-term traffic flow prediction models")
    
    tab1, tab2 = st.tabs(["Train Model", "Predictions"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Model Configuration")
            
            model_type = st.selectbox(
                "Model Type",
                ["Random Forest", "Linear Regression", "Ridge Regression"]
            )
            
            prediction_horizon = st.slider(
                "Prediction Horizon (minutes)", 
                5, 60, 15, 5
            )
            
            # Data generation options
            st.subheader("Training Data")
            use_synthetic = st.checkbox("Generate Synthetic Data", value=True)
            
            if use_synthetic:
                n_days = st.slider("Days of Data", 7, 90, 30)
                data_noise = st.slider("Data Noise Level", 0.0, 1.0, 0.1)
            
            train_model_btn = st.button("🚀 Train Prediction Model", type="primary")
        
        with col2:
            st.subheader("Training Results")
            
            if train_model_btn:
                train_prediction_model(model_type, prediction_horizon, 
                                     use_synthetic, n_days if use_synthetic else 30)
            
            if st.session_state.prediction_model:
                display_prediction_training_results()
    
    with tab2:
        st.subheader("Make Predictions")
        
        if st.session_state.prediction_model:
            make_traffic_predictions()
        else:
            st.info("Please train a prediction model first.")


def train_prediction_model(model_type, prediction_horizon, use_synthetic, n_days):
    """Train traffic prediction model"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Generating training data...")
        progress_bar.progress(0.2)
        
        # Generate synthetic data
        if use_synthetic:
            data = generate_synthetic_traffic_data(n_days=n_days)
        else:
            # Would load real data here
            data = generate_synthetic_traffic_data(n_days=30)
        
        status_text.text("Creating features...")
        progress_bar.progress(0.4)
        
        # Create predictor
        model_type_map = {
            "Random Forest": "random_forest",
            "Linear Regression": "linear",
            "Ridge Regression": "ridge"
        }
        
        predictor = TrafficFlowPredictor(model_type=model_type_map[model_type])
        
        # Create features
        features_df = predictor.create_features(data)
        
        status_text.text("Preparing training data...")
        progress_bar.progress(0.6)
        
        # Prepare training data
        X, y, feature_names = predictor.prepare_training_data(
            features_df, 
            prediction_horizon=prediction_horizon//15  # Convert to 15-min intervals
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        status_text.text("Training model...")
        progress_bar.progress(0.8)
        
        # Train model
        predictor.fit(X_train, y_train)
        
        # Evaluate model
        metrics = predictor.evaluate(X_test, y_test)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Model training completed!")
        
        # Save results
        st.session_state.prediction_model = {
            'predictor': predictor,
            'metrics': metrics,
            'model_type': model_type,
            'prediction_horizon': prediction_horizon,
            'feature_names': feature_names,
            'test_data': (X_test, y_test)
        }
        
        st.success("Prediction model trained successfully!")
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")


def display_prediction_training_results():
    """Display prediction model training results"""
    
    model_info = st.session_state.prediction_model
    metrics = model_info['metrics']
    
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{metrics['mae']:.2f}")
    
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    
    with col3:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
    
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.1f}%")
    
    # Feature importance (if available)
    if hasattr(model_info['predictor'], 'model') and hasattr(model_info['predictor'].model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': model_info['feature_names'],
            'Importance': model_info['predictor'].model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs actual plot
    X_test, y_test = model_info['test_data']
    predictions = model_info['predictor'].predict(X_test)
    y_pred = predictions.predicted_flows
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=y_test[:100],  # Limit to first 100 points for clarity
        y=y_pred[:100],
        mode='markers',
        name='Predictions',
        opacity=0.6
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test[:100]), min(y_pred[:100]))
    max_val = max(max(y_test[:100]), max(y_pred[:100]))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Predictions vs Actual Values',
        xaxis_title='Actual Traffic Flow',
        yaxis_title='Predicted Traffic Flow'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def make_traffic_predictions():
    """Make traffic flow predictions"""
    
    st.subheader("Traffic Flow Forecasting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        intersection_id = st.selectbox(
            "Intersection",
            ["int_1", "int_2", "int_3", "int_4"]
        )
        
        approach = st.selectbox(
            "Approach",
            ["north", "south", "east", "west"]
        )
        
        forecast_horizon = st.slider(
            "Forecast Horizon (hours)",
            1, 12, 4
        )
        
        make_prediction = st.button("🔮 Generate Forecast")
    
    with col2:
        if make_prediction:
            # Generate sample forecast
            timestamps = pd.date_range(
                start=datetime.now(),
                periods=forecast_horizon * 4,  # 15-min intervals
                freq='15min'
            )
            
            # Simulate realistic traffic pattern
            base_flows = []
            for ts in timestamps:
                hour = ts.hour
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
                    base_flow = 300
                elif 10 <= hour <= 16:  # Daytime
                    base_flow = 200
                else:  # Off-peak
                    base_flow = 100
                
                # Add some randomness
                flow = base_flow + np.random.normal(0, 20)
                base_flows.append(max(0, flow))
            
            # Create forecast plot
            fig = go.Figure()
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=base_flows,
                mode='lines+markers',
                name='Predicted Flow',
                line=dict(color='blue', width=2)
            ))
            
            # Add a marker for current time instead of vline to avoid timestamp issues
            current_time_marker = timestamps[0]
            fig.add_trace(go.Scatter(
                x=[current_time_marker],
                y=[base_flows[0]],
                mode='markers',
                name='Current Time',
                marker=dict(color='red', size=10, symbol='diamond'),
                showlegend=True
            ))
            
            # Confidence intervals
            upper_bound = [f + 30 for f in base_flows]
            lower_bound = [max(0, f - 30) for f in base_flows]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor='rgba(0,100,200,0.2)'
            ))
            
            fig.update_layout(
                title=f'Traffic Flow Forecast - {intersection_id} ({approach})',
                xaxis_title='Time',
                yaxis_title='Vehicles per 15 minutes',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Forecast Summary")
            
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Peak Flow", f"{max(base_flows):.0f} veh/15min")
            
            with col2_2:
                st.metric("Average Flow", f"{np.mean(base_flows):.0f} veh/15min")
            
            with col2_3:
                st.metric("Total Vehicles", f"{sum(base_flows):.0f} vehicles")
