"""
Quick Start Script - Employee Sales Forecasting
Run this script to execute the complete ML pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_pipeline():
    """Execute the complete ML pipeline"""
    
    print_header("🚀 EMPLOYEE SALES FORECASTING - QUICK START")
    
    print("📋 This script will:")
    print("   1. Generate synthetic employee sales data (150 employees, 36 months)")
    print("   2. Perform feature engineering")
    print("   3. Train multiple ML models (RF, GB, XGBoost, Ensemble)")
    print("   4. Generate performance reports and visualizations")
    print("   5. Save trained models for deployment")
    
    response = input("\n🤔 Would you like to continue? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        return
    
    print_header("⏳ RUNNING ML PIPELINE")
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run the pipeline
    try:
        print("🔄 Executing pipeline...")
        exec(open('src/utils/pipeline.py').read())
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        return
    
    print_header("🎉 SETUP COMPLETE!")
    
    print("📊 Your forecasting system is ready!")
    print("\n🚀 Launch Dashboard:")
    print("   Run: streamlit run deployment/app.py")
    print("\n📓 View Analysis:")
    print("   Open: notebooks/02_eda_analysis.ipynb")
    print("\n📁 Check Outputs:")
    print("   • Data: data/raw/employee_sales_data.csv")
    print("   • Models: models/*.pkl")
    print("   • Reports: reports/model_comparison.csv")
    print("   • Visualizations: reports/figures/*.png")
    
    # Ask if user wants to launch dashboard
    response = input("\n🎯 Launch the Streamlit dashboard now? (y/n): ")
    
    if response.lower() == 'y':
        print("\n🚀 Launching dashboard...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "deployment/app.py"])
        except KeyboardInterrupt:
            print("\n✅ Dashboard closed")
        except Exception as e:
            print(f"\n❌ Error launching dashboard: {e}")
            print("💡 You can manually run: streamlit run deployment/app.py")


if __name__ == "__main__":
    run_pipeline()
