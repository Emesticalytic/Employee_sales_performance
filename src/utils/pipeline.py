"""
Complete ML Pipeline for Employee Sales Forecasting
Integrates data generation, feature engineering, and model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('..')
from src.data.data_generator import SalesDataGenerator
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer, prepare_train_val_test_split


class SalesForecastingPipeline:
    """End-to-end pipeline for sales forecasting"""
    
    def __init__(self, n_employees=150, n_months=36, random_seed=42):
        self.n_employees = n_employees
        self.n_months = n_months
        self.random_seed = random_seed
        self.df = None
        self.df_featured = None
        self.trainer = None
        
    def generate_data(self):
        """Step 1: Generate synthetic sales data"""
        print("=" * 70)
        print("STEP 1: DATA GENERATION")
        print("=" * 70)
        
        generator = SalesDataGenerator(
            n_employees=self.n_employees,
            n_months=self.n_months,
            random_seed=self.random_seed
        )
        
        self.df = generator.generate_and_save('../data/raw/employee_sales_data.csv')
        return self.df
    
    def engineer_features(self):
        """Step 2: Feature engineering"""
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 70)
        
        engineer = FeatureEngineer(self.df)
        self.df_featured = engineer.engineer_all_features()
        
        # Save featured dataset
        self.df_featured.to_csv('../data/processed/sales_features.csv', index=False)
        print(f"\n✅ Features saved! Shape: {self.df_featured.shape}")
        
        # Get feature names
        self.feature_names = engineer.get_feature_names()
        print(f"📋 Total features: {len(self.feature_names)}")
        
        return self.df_featured
    
    def prepare_data(self):
        """Step 3: Prepare train/val/test splits"""
        print("\n" + "=" * 70)
        print("STEP 3: DATA PREPARATION")
        print("=" * 70)
        
        # Get features
        exclude_cols = ['employee_id', 'date', 'sales', 'year']
        self.feature_names = [col for col in self.df_featured.columns if col not in exclude_cols]
        
        # Split data
        result = prepare_train_val_test_split(
            self.df_featured,
            self.feature_names,
            target_col='sales',
            test_size=0.15,
            val_size=0.15
        )
        
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test,
         self.train_df, self.val_df, self.test_df) = result
        
        print(f"\n📊 Feature matrix shape: {self.X_train.shape}")
        print(f"🎯 Target variable: sales")
        
        return result
    
    def train_models(self):
        """Step 4: Train all models"""
        print("\n" + "=" * 70)
        print("STEP 4: MODEL TRAINING")
        print("=" * 70)
        
        self.trainer = ModelTrainer()
        
        # Train Random Forest
        print("\n" + "-" * 70)
        rf_results = self.trainer.train_random_forest(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        
        # Train Gradient Boosting
        print("\n" + "-" * 70)
        gb_results = self.trainer.train_gradient_boosting(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        
        # Train XGBoost
        print("\n" + "-" * 70)
        xgb_results = self.trainer.train_xgboost(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        
        # Create Ensemble
        print("\n" + "-" * 70)
        ensemble_results = self.trainer.train_ensemble(
            self.X_val, self.y_val,
            weights=[0.3, 0.4, 0.3]  # GB gets higher weight
        )
        
        return self.trainer
    
    def evaluate_models(self):
        """Step 5: Evaluate and compare models"""
        print("\n" + "=" * 70)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 70)
        
        # Get comparison
        comparison = self.trainer.compare_models()
        print("\n📊 MODEL COMPARISON:")
        print(comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv('../reports/model_comparison.csv', index=False)
        print("\n✅ Comparison saved to reports/model_comparison.csv")
        
        return comparison
    
    def save_models(self):
        """Step 6: Save trained models"""
        print("\n" + "=" * 70)
        print("STEP 6: SAVING MODELS")
        print("=" * 70)
        
        model_names = ['random_forest', 'gradient_boosting', 'xgboost']
        
        for model_name in model_names:
            filepath = f'../models/{model_name}_model.pkl'
            self.trainer.save_model(model_name, filepath)
        
        print("\n✅ All models saved successfully!")
    
    def visualize_results(self):
        """Step 7: Create visualizations"""
        print("\n" + "=" * 70)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Get predictions from best model
        best_model_name = self.trainer.compare_models().iloc[0]['Model']
        best_results = self.trainer.results[best_model_name]
        
        val_predictions = best_results['predictions']['val']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Performance Analysis - {best_model_name.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_val, val_predictions, alpha=0.5)
        axes[0, 0].plot([self.y_val.min(), self.y_val.max()],
                       [self.y_val.min(), self.y_val.max()],
                       'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Sales ($)')
        axes[0, 0].set_ylabel('Predicted Sales ($)')
        axes[0, 0].set_title('Actual vs Predicted Sales')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score
        r2 = best_results['val_metrics']['R2']
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}',
                       transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat'),
                       fontweight='bold')
        
        # 2. Residuals
        residuals = self.y_val - val_predictions
        axes[0, 1].scatter(val_predictions, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Sales ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Model comparison
        comparison = self.trainer.compare_models()
        axes[1, 1].barh(comparison['Model'], comparison['Accuracy (%)'],
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_xlabel('Accuracy (%)')
        axes[1, 1].set_title('Model Accuracy Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].axvline(x=90, color='r', linestyle='--', linewidth=2, label='Target: 90%')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('../reports/figures/08_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to reports/figures/")
        
        # Feature importance
        self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        
        feature_imp = self.trainer.get_feature_importance(
            self.feature_names,
            model_name='random_forest',
            top_n=20
        )
        
        plt.figure(figsize=(12, 8))
        plt.barh(feature_imp['feature'], feature_imp['importance'],
                color='steelblue', edgecolor='black')
        plt.xlabel('Importance', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('../reports/figures/09_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plot saved!")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print("🚀 EMPLOYEE SALES FORECASTING - COMPLETE ML PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all steps
        self.generate_data()
        self.engineer_features()
        self.prepare_data()
        self.train_models()
        comparison = self.evaluate_models()
        self.save_models()
        self.visualize_results()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        
        best_model = comparison.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Accuracy (%)']:.2f}%")
        print(f"   MAPE: {best_model['MAPE (%)']:.2f}%")
        print(f"   R² Score: {best_model['R² Score']:.3f}")
        
        # Check targets
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"   Accuracy Target (90%): {'✅ ACHIEVED' if best_model['Accuracy (%)'] >= 90 else '❌ NOT MET'}")
        print(f"   MAPE Target (≤10%): {'✅ ACHIEVED' if best_model['MAPE (%)'] <= 10 else '❌ NOT MET'}")
        
        print(f"\n📁 Generated Files:")
        print(f"   • Data: data/raw/employee_sales_data.csv")
        print(f"   • Features: data/processed/sales_features.csv")
        print(f"   • Models: models/*.pkl")
        print(f"   • Reports: reports/model_comparison.csv")
        print(f"   • Figures: reports/figures/*.png")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Launch dashboard: streamlit run deployment/app.py")
        print(f"   2. Review notebooks for detailed analysis")
        print(f"   3. Deploy models to production")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = SalesForecastingPipeline(
        n_employees=150,
        n_months=36,
        random_seed=42
    )
    
    pipeline.run_complete_pipeline()
