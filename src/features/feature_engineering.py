"""
Feature engineering module for Employee Sales Forecasting
Creates lag features, rolling statistics, and temporal features
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """Feature engineering for time series sales forecasting"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            df: DataFrame with sales data
        """
        self.df = df.copy()
        self.df = self.df.sort_values(['employee_id', 'date']).reset_index(drop=True)
        
    def create_lag_features(self, lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
        """Create lagged sales features"""
        
        for lag in lags:
            self.df[f'sales_lag_{lag}'] = self.df.groupby('employee_id')['sales'].shift(lag)
            
        return self.df
    
    def create_rolling_features(self, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create rolling statistics"""
        
        for window in windows:
            self.df[f'sales_rolling_mean_{window}'] = (
                self.df.groupby('employee_id')['sales']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            self.df[f'sales_rolling_std_{window}'] = (
                self.df.groupby('employee_id')['sales']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            
            self.df[f'sales_rolling_max_{window}'] = (
                self.df.groupby('employee_id')['sales']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )
            
        return self.df
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Create temporal features"""
        
        # Month cyclical encoding
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Quarter encoding
        self.df['quarter_sin'] = np.sin(2 * np.pi * self.df['quarter'] / 4)
        self.df['quarter_cos'] = np.cos(2 * np.pi * self.df['quarter'] / 4)
        
        # Time index (months from start)
        self.df['time_index'] = (
            self.df.groupby('employee_id').cumcount()
        )
        
        return self.df
    
    def create_employee_aggregates(self) -> pd.DataFrame:
        """Create employee-level aggregate features"""
        
        # Average historical sales
        self.df['employee_avg_sales'] = (
            self.df.groupby('employee_id')['sales']
            .transform(lambda x: x.expanding().mean())
        )
        
        # Sales trend (difference from average)
        self.df['sales_vs_avg'] = self.df['sales'] - self.df['employee_avg_sales']
        
        # Coefficient of variation
        self.df['employee_cv'] = (
            self.df.groupby('employee_id')['sales']
            .transform(lambda x: x.expanding().std() / (x.expanding().mean() + 1))
        )
        
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features"""
        
        # Experience x Performance
        self.df['exp_performance'] = (
            self.df['experience_years'] * self.df['performance_score']
        )
        
        # Market potential x Territory size
        self.df['market_territory'] = (
            self.df['market_potential'] * self.df['territory_size']
        )
        
        # Training boost
        self.df['training_impact'] = (
            self.df['training_hours'] * self.df['performance_score'] / 100
        )
        
        return self.df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Create all features"""
        
        print("🔧 Engineering features...")
        
        self.create_lag_features()
        print("  ✓ Lag features created")
        
        self.create_rolling_features()
        print("  ✓ Rolling features created")
        
        self.create_temporal_features()
        print("  ✓ Temporal features created")
        
        self.create_employee_aggregates()
        print("  ✓ Employee aggregates created")
        
        self.create_interaction_features()
        print("  ✓ Interaction features created")
        
        # Fill NaN values for first few rows
        self.df = self.df.fillna(method='bfill')
        
        print(f"✅ Feature engineering complete! Final shape: {self.df.shape}")
        
        return self.df
    
    def get_feature_names(self, exclude: List[str] = None) -> List[str]:
        """Get list of feature column names"""
        
        if exclude is None:
            exclude = ['employee_id', 'date', 'sales', 'year']
            
        features = [col for col in self.df.columns if col not in exclude]
        return features


def prepare_ml_dataset(df: pd.DataFrame, target: str = 'sales') -> tuple:
    """
    Prepare final dataset for ML modeling
    
    Args:
        df: DataFrame with all features
        target: Target variable name
        
    Returns:
        X, y, feature_names
    """
    
    exclude_cols = ['employee_id', 'date', 'sales', 'year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target].values
    
    return X, y, feature_cols


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/raw/employee_sales_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    engineer = FeatureEngineer(df)
    df_featured = engineer.engineer_all_features()
    
    df_featured.to_csv("../data/processed/sales_features.csv", index=False)
    print(f"✅ Features saved with {len(engineer.get_feature_names())} features")
