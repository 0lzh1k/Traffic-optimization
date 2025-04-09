"""
Traffic flow prediction models for short-term forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
import datetime
from dataclasses import dataclass

@dataclass
class PredictionResult:
    """Container for prediction results"""
    predicted_flows: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None

class TrafficFlowPredictor:
    """
    Base class for traffic flow prediction models
    """
    
    def __init__(self, model_type: str = "random_forest", model_params: Dict = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
    def _create_model(self):
        """Create prediction model based on type"""
        if self.model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == "linear":
            default_params = {}  # LinearRegression doesn't take random_state
            params = {**default_params, **self.model_params}
            self.model = LinearRegression(**params)
            
        elif self.model_type == "ridge":
            default_params = {'alpha': 1.0, 'random_state': 42}
            params = {**default_params, **self.model_params}
            self.model = Ridge(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def create_features(self, traffic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for traffic flow prediction
        
        Args:
            traffic_data: DataFrame with columns ['timestamp', 'intersection_id', 
                         'approach', 'vehicle_count', 'avg_speed', 'occupancy']
        
        Returns:
            DataFrame with engineered features
        """
        features_df = traffic_data.copy()
        
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(features_df['timestamp']):
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Time-based features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # Peak hour indicators
        features_df['is_morning_peak'] = ((features_df['hour'] >= 7) & 
                                        (features_df['hour'] <= 9)).astype(int)
        features_df['is_evening_peak'] = ((features_df['hour'] >= 17) & 
                                        (features_df['hour'] <= 19)).astype(int)
        
        # Sort by timestamp for lag features
        features_df = features_df.sort_values(['intersection_id', 'approach', 'timestamp'])
        
        # Lag features (previous time periods)
        lag_columns = ['vehicle_count', 'avg_speed', 'occupancy']
        for col in lag_columns:
            if col in features_df.columns:
                # 1, 2, 3 periods back
                for lag in [1, 2, 3]:
                    features_df[f'{col}_lag_{lag}'] = features_df.groupby(
                        ['intersection_id', 'approach'])[col].shift(lag)
                
                # Rolling averages - fix the index issue
                for window in [3, 6, 12]:
                    rolling_values = features_df.groupby(
                        ['intersection_id', 'approach'])[col].rolling(
                        window=window, min_periods=1).mean()
                    
                    # Properly align the rolling values with the original DataFrame
                    features_df[f'{col}_rolling_avg_{window}'] = rolling_values.values
        
        # Traffic pattern features
        if 'vehicle_count' in features_df.columns:
            # Rate of change
            features_df['vehicle_count_change'] = features_df.groupby(
                ['intersection_id', 'approach'])['vehicle_count'].diff()
            
            # Normalized by historical average for same hour
            hourly_avg = features_df.groupby(['intersection_id', 'approach', 'hour'])['vehicle_count'].transform('mean')
            features_df['vehicle_count_normalized'] = features_df['vehicle_count'] / (hourly_avg + 1e-6)
        
        # Weather features (placeholder - would integrate with weather API)
        features_df['weather_temp'] = 20.0  # Default temperature
        features_df['weather_precipitation'] = 0.0  # Default no rain
        features_df['weather_visibility'] = 10.0  # Default good visibility
        
        # Event features (placeholder - would integrate with event calendar)
        features_df['is_holiday'] = 0  # Default no holiday
        features_df['special_event'] = 0  # Default no special event
        
        return features_df
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            target_column: str = 'vehicle_count',
                            prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data with features and targets
        
        Args:
            features_df: DataFrame with features
            target_column: Column to predict
            prediction_horizon: Number of time steps ahead to predict
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Make a clean copy and reset index to avoid indexing issues
        features_df = features_df.copy().reset_index(drop=True)
        
        # Remove rows with missing values
        features_df = features_df.dropna()
        
        if len(features_df) == 0:
            raise ValueError("No valid data after removing missing values")
        
        # Create target variable (future values)
        features_df = features_df.sort_values(['intersection_id', 'approach', 'timestamp']).reset_index(drop=True)
        
        # Create target using shift with proper grouping
        target_series = features_df.groupby(['intersection_id', 'approach'])[target_column].shift(-prediction_horizon)
        features_df['target'] = target_series
        
        # Remove rows without target values
        features_df = features_df.dropna(subset=['target'])
        
        if len(features_df) == 0:
            raise ValueError("No valid data after creating target variable")
        
        # Select feature columns (exclude metadata and target)
        exclude_columns = ['timestamp', 'intersection_id', 'approach', target_column, 'target']
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        if len(feature_columns) == 0:
            raise ValueError("No feature columns available")
        
        X = features_df[feature_columns].values
        y = features_df['target'].values
        
        self.feature_names = feature_columns
        
        return X, y, feature_columns
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the prediction model"""
        if self.model is None:
            self._create_model()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Calculate confidence intervals for ensemble models
        confidence_intervals = None
        if self.model_type == "random_forest" and hasattr(self.model, 'estimators_'):
            # Use prediction variance from individual trees
            tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            pred_std = np.std(tree_predictions, axis=0)
            confidence_intervals = np.column_stack([
                predictions - 1.96 * pred_std,  # Lower bound (95% CI)
                predictions + 1.96 * pred_std   # Upper bound (95% CI)
            ])
        
        return PredictionResult(
            predicted_flows=predictions,
            confidence_intervals=confidence_intervals,
            feature_importance=feature_importance
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        y_pred = predictions.predicted_flows
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save trained model and scaler"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load trained model and scaler"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']

class RealTimePredictor:
    """
    Real-time traffic flow predictor for live system integration
    """
    
    def __init__(self, model_path: str = None):
        self.predictor = TrafficFlowPredictor()
        self.recent_data = {}  # Store recent observations by intersection/approach
        self.prediction_cache = {}  # Cache recent predictions
        self.cache_duration = 300  # Cache predictions for 5 minutes
        
        if model_path and os.path.exists(model_path):
            self.predictor.load_model(model_path)
    
    def update_sensor_data(self, intersection_id: str, approach: str, 
                          timestamp: datetime.datetime, vehicle_count: int,
                          avg_speed: float = None, occupancy: float = None):
        """
        Update with new sensor reading
        
        Args:
            intersection_id: Intersection identifier
            approach: Approach direction (north, south, east, west)
            timestamp: Timestamp of measurement
            vehicle_count: Number of vehicles observed
            avg_speed: Average speed of vehicles (optional)
            occupancy: Lane occupancy percentage (optional)
        """
        key = f"{intersection_id}_{approach}"
        
        if key not in self.recent_data:
            self.recent_data[key] = []
        
        # Add new observation
        observation = {
            'timestamp': timestamp,
            'intersection_id': intersection_id,
            'approach': approach,
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed or 30.0,  # Default speed
            'occupancy': occupancy or 0.1     # Default occupancy
        }
        
        self.recent_data[key].append(observation)
        
        # Keep only recent data (last 2 hours)
        cutoff_time = timestamp - datetime.timedelta(hours=2)
        self.recent_data[key] = [
            obs for obs in self.recent_data[key] 
            if obs['timestamp'] > cutoff_time
        ]
    
    def predict_flow(self, intersection_id: str, approach: str, 
                    prediction_horizon_minutes: int = 15) -> PredictionResult:
        """
        Predict traffic flow for specific intersection/approach
        
        Args:
            intersection_id: Intersection to predict for
            approach: Approach direction
            prediction_horizon_minutes: How far ahead to predict (in minutes)
        
        Returns:
            PredictionResult with predicted flows
        """
        if not self.predictor.is_fitted:
            raise ValueError("Predictor model not trained. Train model first.")
        
        key = f"{intersection_id}_{approach}"
        cache_key = f"{key}_{prediction_horizon_minutes}"
        
        # Check cache
        current_time = datetime.datetime.now()
        if (cache_key in self.prediction_cache and 
            (current_time - self.prediction_cache[cache_key]['timestamp']).seconds < self.cache_duration):
            return self.prediction_cache[cache_key]['result']
        
        # Get recent data for this intersection/approach
        if key not in self.recent_data or len(self.recent_data[key]) < 3:
            # Not enough data for prediction
            return PredictionResult(
                predicted_flows=np.array([200.0]),  # Default prediction
                confidence_intervals=np.array([[180.0, 220.0]]),
                metrics={'data_quality': 'insufficient'}
            )
        
        # Convert to DataFrame
        recent_df = pd.DataFrame(self.recent_data[key])
        
        # Create features
        features_df = self.predictor.create_features(recent_df)
        
        # Get the most recent complete observation
        if len(features_df) == 0:
            return PredictionResult(
                predicted_flows=np.array([200.0]),
                confidence_intervals=np.array([[180.0, 220.0]]),
                metrics={'data_quality': 'insufficient_features'}
            )
        
        # Take the last complete row
        latest_features = features_df.iloc[[-1]]
        
        # Remove metadata columns
        exclude_columns = ['timestamp', 'intersection_id', 'approach', 'vehicle_count']
        feature_columns = [col for col in latest_features.columns if col not in exclude_columns]
        
        # Handle missing feature columns
        for feature in self.predictor.feature_names:
            if feature not in feature_columns:
                latest_features[feature] = 0.0  # Default value for missing features
        
        # Select only model features in correct order
        X = latest_features[self.predictor.feature_names].values
        
        # Make prediction
        result = self.predictor.predict(X)
        
        # Cache result
        self.prediction_cache[cache_key] = {
            'timestamp': current_time,
            'result': result
        }
        
        return result
    
    def predict_network_flows(self, network_intersections: List[Tuple[str, List[str]]], 
                            prediction_horizon_minutes: int = 15) -> Dict[str, Dict[str, PredictionResult]]:
        """
        Predict flows for entire network
        
        Args:
            network_intersections: List of (intersection_id, [approaches]) tuples
            prediction_horizon_minutes: Prediction horizon
        
        Returns:
            Nested dict with predictions by intersection and approach
        """
        network_predictions = {}
        
        for intersection_id, approaches in network_intersections:
            network_predictions[intersection_id] = {}
            
            for approach in approaches:
                try:
                    prediction = self.predict_flow(
                        intersection_id, approach, prediction_horizon_minutes
                    )
                    network_predictions[intersection_id][approach] = prediction
                except Exception as e:
                    # Handle prediction errors gracefully
                    network_predictions[intersection_id][approach] = PredictionResult(
                        predicted_flows=np.array([200.0]),
                        confidence_intervals=np.array([[180.0, 220.0]]),
                        metrics={'error': str(e)}
                    )
        
        return network_predictions

def generate_synthetic_traffic_data(n_days: int = 30, 
                                  intersections: List[str] = None,
                                  approaches: List[str] = None) -> pd.DataFrame:
    """
    Generate synthetic traffic data for testing prediction models
    
    Args:
        n_days: Number of days to generate
        intersections: List of intersection IDs
        approaches: List of approach directions
    
    Returns:
        DataFrame with synthetic traffic data
    """
    if intersections is None:
        intersections = ['int_1', 'int_2', 'int_3', 'int_4']
    
    if approaches is None:
        approaches = ['north', 'south', 'east', 'west']
    
    # Generate time series
    start_date = datetime.datetime.now() - datetime.timedelta(days=n_days)
    timestamps = pd.date_range(start=start_date, periods=n_days*24*4, freq='15min')  # 15-minute intervals
    
    data = []
    
    for intersection in intersections:
        for approach in approaches:
            for timestamp in timestamps:
                # Create realistic traffic patterns
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                
                # Base traffic level
                if day_of_week < 5:  # Weekday
                    if 7 <= hour <= 9:  # Morning peak
                        base_flow = 300
                    elif 17 <= hour <= 19:  # Evening peak
                        base_flow = 350
                    elif 10 <= hour <= 16:  # Daytime
                        base_flow = 200
                    else:  # Off-peak
                        base_flow = 100
                else:  # Weekend
                    if 10 <= hour <= 18:
                        base_flow = 150
                    else:
                        base_flow = 80
                
                # Add randomness
                vehicle_count = max(0, np.random.poisson(base_flow/4))  # 15-minute count
                avg_speed = np.random.normal(30, 5)  # Speed in km/h
                avg_speed = max(5, min(60, avg_speed))  # Clamp between 5-60 km/h
                occupancy = min(1.0, vehicle_count / 100.0)  # Simple occupancy model
                
                data.append({
                    'timestamp': timestamp,
                    'intersection_id': intersection,
                    'approach': approach,
                    'vehicle_count': vehicle_count,
                    'avg_speed': avg_speed,
                    'occupancy': occupancy
                })
    
    return pd.DataFrame(data)
