"""
Configuration file for Employee Sales Forecasting Project
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, 
                  MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Generation Parameters
N_EMPLOYEES = 150
N_MONTHS = 36  # 3 years of historical data
RANDOM_SEED = 42

# Model Parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CV_FOLDS = 5

# Target Metrics
TARGET_ACCURACY = 0.90
TARGET_MAPE = 10.0
MAX_RESPONSE_TIME = 2.0  # seconds

# Feature Engineering
FEATURE_GROUPS = {
    'historical': ['sales_lag_1', 'sales_lag_3', 'sales_lag_6', 
                   'sales_rolling_mean_3', 'sales_rolling_std_3'],
    'employee': ['experience_years', 'training_hours', 'performance_score'],
    'temporal': ['month', 'quarter', 'is_holiday_season'],
    'territory': ['territory_size', 'market_potential', 'competition_level']
}

# Model Hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_SEED
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_samples_split': 10,
        'random_state': RANDOM_SEED
    },
    'xgboost': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED
    }
}

# LSTM Parameters
LSTM_PARAMS = {
    'sequence_length': 6,
    'units': 64,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}

# Deployment Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DASHBOARD_PORT = 8501

# Retraining Schedule
RETRAIN_FREQUENCY_DAYS = 14
PERFORMANCE_THRESHOLD_MAPE = 12.0  # Trigger retraining if MAPE exceeds this
