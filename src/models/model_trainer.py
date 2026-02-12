"""
Model training and evaluation utilities
Supports Random Forest, Gradient Boosting, XGBoost, LSTM, and Ensemble models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Tuple, List
import joblib
import time


class ModelTrainer:
    """Train and evaluate machine learning models for sales forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Prevent division by zero
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'Accuracy': 100 - (mean_absolute_percentage_error(y_true, y_pred) * 100)
        }
        
        return metrics
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           params: Dict = None) -> Dict:
        """Train Random Forest model"""
        
        print("🌲 Training Random Forest...")
        start_time = time.time()
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        training_time = time.time() - start_time
        
        self.models['random_forest'] = model
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            }
        }
        
        self.results['random_forest'] = results
        
        print(f"✅ Random Forest trained in {training_time:.2f}s")
        print(f"   Val Accuracy: {val_metrics['Accuracy']:.2f}% | MAPE: {val_metrics['MAPE']:.2f}%")
        
        return results
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               params: Dict = None) -> Dict:
        """Train Gradient Boosting model"""
        
        print("📈 Training Gradient Boosting...")
        start_time = time.time()
        
        if params is None:
            params = {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 8,
                'min_samples_split': 10,
                'random_state': 42
            }
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        training_time = time.time() - start_time
        
        self.models['gradient_boosting'] = model
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            }
        }
        
        self.results['gradient_boosting'] = results
        
        print(f"✅ Gradient Boosting trained in {training_time:.2f}s")
        print(f"   Val Accuracy: {val_metrics['Accuracy']:.2f}% | MAPE: {val_metrics['MAPE']:.2f}%")
        
        return results
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     params: Dict = None) -> Dict:
        """Train XGBoost model"""
        
        print("🚀 Training XGBoost...")
        start_time = time.time()
        
        if params is None:
            params = {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        training_time = time.time() - start_time
        
        self.models['xgboost'] = model
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            }
        }
        
        self.results['xgboost'] = results
        
        print(f"✅ XGBoost trained in {training_time:.2f}s")
        print(f"   Val Accuracy: {val_metrics['Accuracy']:.2f}% | MAPE: {val_metrics['MAPE']:.2f}%")
        
        return results
    
    def train_ensemble(self, X_val: np.ndarray, y_val: np.ndarray,
                      weights: List[float] = None) -> Dict:
        """Create ensemble from trained models"""
        
        print("🎯 Creating Ensemble Model...")
        
        if weights is None:
            # Equal weights
            weights = [1/len(self.results)] * len(self.results)
        
        # Get predictions from all models
        val_predictions = []
        for model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            if model_name in self.results:
                val_predictions.append(self.results[model_name]['predictions']['val'])
        
        # Weighted average
        ensemble_val_pred = np.average(np.array(val_predictions), axis=0, weights=weights[:len(val_predictions)])
        
        # Metrics
        val_metrics = self.calculate_metrics(y_val, ensemble_val_pred)
        
        results = {
            'model': 'ensemble',
            'weights': weights,
            'val_metrics': val_metrics,
            'predictions': {
                'val': ensemble_val_pred
            }
        }
        
        self.results['ensemble'] = results
        
        print(f"✅ Ensemble created")
        print(f"   Val Accuracy: {val_metrics['Accuracy']:.2f}% | MAPE: {val_metrics['MAPE']:.2f}%")
        
        return results
    
    def get_feature_importance(self, feature_names: List[str], 
                              model_name: str = 'random_forest',
                              top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return feature_imp
        else:
            raise ValueError(f"Model {model_name} does not have feature_importances_")
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """Load model from disk"""
        
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"✅ Model loaded from {filepath}")
        return model
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        
        comparison = []
        
        for model_name, results in self.results.items():
            if 'val_metrics' in results:
                row = {
                    'Model': model_name,
                    'Accuracy (%)': results['val_metrics']['Accuracy'],
                    'MAPE (%)': results['val_metrics']['MAPE'],
                    'RMSE': results['val_metrics']['RMSE'],
                    'MAE': results['val_metrics']['MAE'],
                    'R²': results['val_metrics']['R2']
                }
                
                if 'training_time' in results:
                    row['Training Time (s)'] = results['training_time']
                
                comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison).sort_values('Accuracy (%)', ascending=False)
        
        return comparison_df


def prepare_train_val_test_split(df: pd.DataFrame, 
                                 feature_cols: List[str],
                                 target_col: str = 'sales',
                                 test_size: float = 0.15,
                                 val_size: float = 0.15) -> Tuple:
    """
    Split data into train, validation, and test sets
    Time-based split to prevent data leakage
    """
    
    # Sort by date
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    n = len(df_sorted)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    train_df = df_sorted.iloc[:val_idx]
    val_df = df_sorted.iloc[val_idx:test_idx]
    test_df = df_sorted.iloc[test_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df


if __name__ == "__main__":
    print("Model trainer module loaded successfully!")
