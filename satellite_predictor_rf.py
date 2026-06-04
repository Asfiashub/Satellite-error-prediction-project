"""
Satellite Error Prediction Model (Random Forest Version)
Predicts satellite ephemeris and clock errors using Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')


class SatelliteErrorPredictor:
    """
    A class to predict satellite ephemeris and clock errors using Random Forest models.
    
    This model predicts the 8th day errors based on 7 days of historical data.
    """
    
    def __init__(self, sequence_length=7):
        """
        Initialize the predictor.
        
        Args:
            sequence_length (int): Number of time steps to look back (default: 7 days)
        """
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
        
    def load_and_preprocess_data(self, file_path, satellite_type='MEO'):
        """
        Load and preprocess satellite data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            satellite_type (str): Type of satellite orbit ('GEO' or 'MEO')
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Load data
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        column_mapping = {
            'x_error  (m)': 'x_error (m)',
            'y_error  (m)': 'y_error (m)',
            'z_error  (m)': 'z_error (m)',
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert time to datetime
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        df = df.sort_values('utc_time').reset_index(drop=True)
        
        # Handle missing values
        df[self.feature_columns] = df[self.feature_columns].interpolate(method='linear')
        df.fillna(method='bfill', inplace=True)
        
        # Outlier detection and treatment using IQR method
        for col in self.feature_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers instead of removing
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
        
        # Add time-based features
        df['hour'] = df['utc_time'].dt.hour
        df['day_of_week'] = df['utc_time'].dt.dayofweek
        df['day_of_year'] = df['utc_time'].dt.dayofyear
        
        # Add rolling statistics as features
        for col in self.feature_columns:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=24, min_periods=1).std()
        
        return df
    
    def create_sequences(self, data, target_idx):
        """
        Create sequences for time series prediction.
        
        Args:
            data (np.array): Input data
            target_idx (int): Index of target column
            
        Returns:
            tuple: X (features) and y (targets) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, target_idx])
            
        return np.array(X), np.array(y)
    
    def train_model(self, df, satellite_type, n_estimators=100, max_depth=None):
        """
        Train Random Forest models for each error component.
        
        Args:
            df (pd.DataFrame): Input dataframe
            satellite_type (str): Type of satellite ('GEO' or 'MEO')
            n_estimators (int): Number of trees in Random Forest
            max_depth (int): Maximum depth of trees (None for unlimited)
            
        Returns:
            dict: Training history for each model
        """
        histories = {}
        feature_cols = [col for col in df.columns if col != 'utc_time']
        
        for error_col in self.feature_columns:
            # Prepare data
            scaler = StandardScaler()
            data = df[feature_cols].values
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            target_idx = feature_cols.index(error_col)
            X, y = self.create_sequences(scaled_data, target_idx)
            
            # Reshape X for Random Forest (samples, features)
            X_rf = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_rf, y, test_size=0.2, shuffle=False
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Store model and scaler
            model_key = f'{satellite_type}_{error_col}'
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Calculate training and validation scores
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            histories[error_col] = {
                'loss': [1 - train_score],  # Convert R² to loss
                'val_loss': [1 - val_score]
            }
        
        return histories
    
    def predict_8th_day(self, df, satellite_type):
        """
        Predict errors for the 8th day.
        
        Args:
            df (pd.DataFrame): Input dataframe
            satellite_type (str): Type of satellite ('GEO' or 'MEO')
            
        Returns:
            dict: Predicted errors for each component
        """
        feature_cols = [col for col in df.columns if col != 'utc_time']
        predictions = {}
        
        # Get the last sequence
        last_sequence = df[feature_cols].values[-self.sequence_length:]
        
        for error_col in self.feature_columns:
            model_key = f'{satellite_type}_{error_col}'
            
            if model_key not in self.models:
                continue
                
            # Scale the sequence
            scaler = self.scalers[model_key]
            scaled_sequence = scaler.transform(last_sequence)
            
            # Reshape for Random Forest
            X = scaled_sequence.reshape(1, -1)
            
            # Predict
            model = self.models[model_key]
            y_pred_scaled = model.predict(X)
            
            # Inverse transform the prediction
            dummy = np.zeros((1, len(feature_cols)))
            target_idx = feature_cols.index(error_col)
            dummy[0, target_idx] = y_pred_scaled[0]
            y_pred = scaler.inverse_transform(dummy)[0, target_idx]
            
            predictions[error_col] = y_pred
        
        return predictions
    
    def save_models(self, directory='models'):
        """Save trained models and scalers to disk."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_key, model in self.models.items():
            model_path = f"{directory}/{model_key}_model.pkl"
            scaler_path = f"{directory}/{model_key}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[model_key], f)
    
    def load_models(self, directory='models'):
        """Load trained models and scalers from disk."""
        import os
        import glob
        
        model_files = glob.glob(f"{directory}/*_model.pkl")
        
        for model_path in model_files:
            model_key = os.path.basename(model_path).replace('_model.pkl', '')
            scaler_path = model_path.replace('_model.pkl', '_scaler.pkl')
            
            with open(model_path, 'rb') as f:
                self.models[model_key] = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scalers[model_key] = pickle.load(f)