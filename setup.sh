#!/bin/bash

# Employee Sales Forecasting - Setup Script
# This script sets up the complete environment and runs the project

echo "========================================================================"
echo "🚀 EMPLOYEE SALES FORECASTING - AUTOMATED SETUP"
echo "========================================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "   ✅ All packages installed"

# Create directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/predictions
mkdir -p models reports/figures
echo "   ✅ Directories created"

# Run the pipeline
echo ""
echo "========================================================================"
echo "🤖 RUNNING ML PIPELINE"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Generate synthetic sales data (150 employees, 36 months)"
echo "  2. Engineer 30+ features"
echo "  3. Train 4 ML models (RF, GB, XGBoost, Ensemble)"
echo "  4. Create 9 visualizations"
echo "  5. Generate performance reports"
echo ""
echo "Expected duration: 5-8 minutes"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python src/utils/pipeline.py
    
    echo ""
    echo "========================================================================"
    echo "✅ SETUP COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "📊 Generated Files:"
    echo "   • data/raw/employee_sales_data.csv"
    echo "   • data/processed/sales_features.csv"
    echo "   • models/*.pkl (trained models)"
    echo "   • reports/model_comparison.csv"
    echo "   • reports/figures/*.png"
    echo ""
    echo "🚀 Next Steps:"
    echo ""
    echo "1. Launch Dashboard:"
    echo "   streamlit run deployment/app.py"
    echo ""
    echo "2. View Analysis:"
    echo "   jupyter notebook notebooks/00_complete_demo.ipynb"
    echo ""
    echo "3. Read Documentation:"
    echo "   • QUICK_START.md"
    echo "   • DEPLOYMENT_GUIDE.md"
    echo "   • PROJECT_SUMMARY.md"
    echo ""
    
    read -p "Launch dashboard now? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Launching Streamlit dashboard..."
        echo "   Access at: http://localhost:8501"
        echo "   Press Ctrl+C to stop"
        echo ""
        streamlit run deployment/app.py
    else
        echo "✅ Setup complete! Run 'streamlit run deployment/app.py' when ready."
    fi
else
    echo "❌ Setup cancelled"
fi
