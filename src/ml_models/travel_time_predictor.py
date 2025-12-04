import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TravelTimePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class TravelTimeModel:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, train_data: Dict, weather_data: Dict = None) -> np.ndarray:
        """Prepare features for travel time prediction"""
        features = []
        
        # Train characteristics
        features.extend([
            train_data.get('max_speed_kmph', 0) / 100,  # Normalized
            train_data.get('weight_tonnes', 0) / 1000,  # Normalized
            train_data.get('length_m', 0) / 500,  # Normalized
            train_data.get('priority', 3) / 5,  # Normalized priority
        ])
        
        # Segment characteristics
        segment_data = train_data.get('segment', {})
        features.extend([
            segment_data.get('length_km', 0) / 100,  # Normalized
            segment_data.get('gradient', 0) / 10,  # Normalized
            segment_data.get('curvature', 0) / 100,  # Normalized
        ])
        
        # Time features
        current_time = datetime.now()
        features.extend([
            current_time.hour / 24,
            current_time.weekday() / 7,
            1 if current_time.month in [12, 1, 2] else 0,  # Winter
            1 if current_time.month in [6, 7, 8] else 0,  # Summer
        ])
        
        # Weather features if available
        if weather_data:
            features.extend([
                weather_data.get('temperature', 20) / 50,  # Normalized
                weather_data.get('precipitation_mm', 0) / 100,  # Normalized
                weather_data.get('visibility_km', 10) / 10,  # Normalized
            ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_travel_time(self, train_data: Dict, segment_id: str, 
                           historical_patterns: Dict = None) -> float:
        """Predict travel time for a train on a segment"""
        try:
            # Prepare features
            features = self.prepare_features(train_data)
            
            if self.model:
                # Use neural network prediction
                features_tensor = torch.FloatTensor(features)
                with torch.no_grad():
                    prediction = self.model(features_tensor)
                base_time = prediction.item()
            else:
                # Fallback to rule-based prediction
                base_time = self.rule_based_prediction(train_data, segment_id)
            
            # Adjust based on historical patterns
            if historical_patterns and segment_id in historical_patterns:
                hist_avg = historical_patterns[segment_id].get('avg_travel_time', base_time)
                hist_std = historical_patterns[segment_id].get('std_travel_time', 0)
                
                # Blend prediction with historical average
                base_time = 0.7 * base_time + 0.3 * hist_avg
                
                # Add uncertainty
                uncertainty = hist_std * np.random.normal(0, 0.5)
                base_time += uncertainty
            
            return max(5, base_time)  # Minimum 5 minutes
        
        except Exception as e:
            logger.error(f"Travel time prediction error: {e}")
            # Return conservative estimate
            return 30.0  # 30 minutes default
    
    def rule_based_prediction(self, train_data: Dict, segment_id: str) -> float:
        """Rule-based travel time prediction (fallback)"""
        segment_length = train_data.get('segment', {}).get('length_km', 10)
        train_speed = train_data.get('max_speed_kmph', 60)
        
        # Adjust speed based on priority
        speed_multiplier = {
            1: 1.1,  # Highest priority - faster
            2: 1.05,
            3: 1.0,  # Medium priority - normal
            4: 0.95,
            5: 0.9   # Lowest priority - slower
        }
        
        priority = train_data.get('priority', 3)
        adjusted_speed = train_speed * speed_multiplier.get(priority, 1.0)
        
        # Calculate base travel time
        base_time_minutes = (segment_length / adjusted_speed) * 60
        
        # Add buffer for acceleration/deceleration
        buffer = 5  # minutes
        
        return base_time_minutes + buffer
    
    def train_model(self, training_data: List[Dict], epochs: int = 50):
        """Train the neural network model"""
        logger.info(f"Training model on {len(training_data)} samples")
        
        # Prepare training data
        X = []
        y = []
        
        for sample in training_data:
            features = self.prepare_features(sample)
            X.append(features)
            y.append(sample['actual_travel_time'])
        
        X = np.array(X).squeeze(1)  # Remove extra dimension
        y = np.array(y).reshape(-1, 1)
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        
        # Initialize model
        input_dim = X_scaled.shape[1]
        self.model = TravelTimePredictor(input_dim)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_tensor.unsqueeze(1))  # Add sequence dimension
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        logger.info("Model training completed")
    
    def save_model(self, path: str):
        """Save model to disk"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        try:
            checkpoint = torch.load(path)
            
            input_dim = len(checkpoint['feature_columns'])
            self.model = TravelTimePredictor(input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
