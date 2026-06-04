"""
# Satellite Error Prediction Model
# Predicts satellite ephemeris and clock errors using LSTM neural networks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')


class SatelliteErrorPredictor:

    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'x_error (m)',
            'y_error (m)',
            'z_error (m)',
            'satclockerror (m)'
        ]

    # ============================
    # FEATURE ENGINEERING (FIXED)
    # ============================
    def engineer_features(self, df):

        df = df.copy()

        rename_map = {
            "clock_error_m": "satclockerror (m)",
            "x_error_m": "x_error (m)",
            "y_error_m": "y_error (m)",
            "z_error_m": "z_error (m)"
        }

        df.rename(columns=rename_map, inplace=True)

        df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

        df["total_position_error"] = np.sqrt(
            df["x_error (m)"]**2 +
            df["y_error (m)"]**2 +
            df["z_error (m)"]**2
        )

        df["hour"] = df["utc_time"].dt.hour
        df["day"] = df["utc_time"].dt.day
        df["day_of_week"] = df["utc_time"].dt.dayofweek

        for col in self.feature_columns:
            df[f"{col}_rolling_mean"] = (
                df[col].rolling(window=3, min_periods=1).mean()
            )

            df[f"{col}_rolling_std"] = (
                df[col].rolling(window=3, min_periods=1).std().fillna(0)
            )

        if "satname" in df.columns:
            df.drop(columns=["satname"], inplace=True)

        if "ephemeris_error_m" in df.columns:
            df.drop(columns=["ephemeris_error_m"], inplace=True)

        return df

    # ============================
    # LOAD & PREPROCESS
    # ============================
    def load_and_preprocess_data(self, file_path, satellite_type='MEO'):

        print(f"\n{'='*60}")
        print(f"Loading {satellite_type} data from: {file_path}")
        print(f"{'='*60}")

        df = pd.read_csv(file_path)

        df.columns = df.columns.str.strip()

        column_mapping = {
            'x_error  (m)': 'x_error (m)',
            'y_error  (m)': 'y_error (m)',
            'z_error  (m)': 'z_error (m)',
        }

        df.rename(columns=column_mapping, inplace=True)

        df['utc_time'] = pd.to_datetime(df['utc_time'])
        df = df.sort_values('utc_time').reset_index(drop=True)

        print(f"✓ Loaded {len(df)} records")

        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            df[self.feature_columns] = df[self.feature_columns].interpolate(method='linear')
            df.ffill(inplace=True) # Prefer ffill over bfill to prevent data leakage

        print("\n🔍 Detecting and treating outliers...")

        for col in self.feature_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR

            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper

        print("\n🔧 Engineering features...")

        df = self.engineer_features(df)

        return df

    # ============================
    # SEQUENCE CREATION
    # ============================
    def create_sequences(self, data, target_col):

        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])

        return np.array(X), np.array(y)

    # ============================
    # MODEL
    # ============================
    def build_lstm_model(self, input_shape):

        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    # ============================
    # TRAIN
    # ============================
    def train_model(self, df, satellite_type='MEO', epochs=50, batch_size=32):

        histories = {}

        feature_cols = [c for c in df.columns if c != 'utc_time']
        data = df[feature_cols].values

        # Fit the scaler once per dataset, not once per target
        scaler = StandardScaler()
        
        # Fit the scaler only on the training set (first 80%) to prevent data leakage
        train_size = int(len(data) * 0.8)
        if train_size > 0:
            scaler.fit(data[:train_size])
        else:
            scaler.fit(data)
            
        scaled = scaler.transform(data)

        for target in self.feature_columns:

            self.scalers[f'{satellite_type}_{target}'] = scaler

            target_idx = feature_cols.index(target)

            X, y = self.create_sequences(scaled, target_idx)

            if len(X) < 10:
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            model = self.build_lstm_model((X.shape[1], X.shape[2]))

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            self.models[f'{satellite_type}_{target}'] = model
            histories[target] = history.history

        return histories

    # ============================
    # PREDICT
    # ============================
    def predict_8th_day(self, df, satellite_type='MEO'):

        predictions = {}

        df = self.engineer_features(df)

        feature_cols = [c for c in df.columns if c != 'utc_time']

        for target in self.feature_columns:

            key = f'{satellite_type}_{target}'

            if key not in self.models:
                continue

            data = df[feature_cols].values
            print("\nFEATURE COLUMNS:")
            for i, col in enumerate(feature_cols):
             print(i, col)
            print("\nFEATURE COUNT:", len(feature_cols))

            scaler = self.scalers[key]
            print("\nSCALER EXPECTS:")
            print(scaler.n_features_in_)
            scaled = scaler.transform(data)

            last_seq = scaled[-self.sequence_length:]
            last_seq = last_seq.reshape(1, self.sequence_length, -1)

            model = self.models[key]

            pred_scaled = model.predict(last_seq, verbose=0)

            dummy = np.zeros((1, len(feature_cols)))
            idx = feature_cols.index(target)

            dummy[0, idx] = pred_scaled[0, 0]

            pred = scaler.inverse_transform(dummy)[0, idx]

            predictions[target] = pred

        return predictions

    # ============================
    # SAVE / LOAD
    # ============================
    def save_models(self, save_dir='models'):

        import os
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            model.save(f'{save_dir}/{name}_model.h5')

        with open(f'{save_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)

    def load_models(self, save_dir='models'):

        import os

        for file in os.listdir(save_dir):
            if file.endswith('_model.h5'):
                name = file.replace('_model.h5', '')
                self.models[name] = keras.models.load_model(
                    f'{save_dir}/{file}',
                    compile=False
                )

        with open(f'{save_dir}/scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)

        print(f"✓ Models loaded from {save_dir}/")


# ============================
# MAIN
# ============================
def main():

    predictor = SatelliteErrorPredictor(sequence_length=7)

    geo_df = predictor.load_and_preprocess_data('DATA_GEO_Train.csv', 'GEO')
    meo_df = predictor.load_and_preprocess_data('DATA_MEO_Train.csv', 'MEO')

    predictor.train_model(geo_df, 'GEO')
    predictor.train_model(meo_df, 'MEO')

    predictor.save_models('models')


if __name__ == "__main__":
    main()