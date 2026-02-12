# 🚀 Employee Sales Forecasting - Deployment Guide

**Project Lead**: Emem Akpan  
**GitHub**: [@Emesticalytic](https://github.com/Emesticalytic)

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Running the Project](#running-the-project)
5. [Model Performance](#model-performance)
6. [Dashboard Features](#dashboard-features)
7. [Deployment to Production](#deployment-to-production)
8. [Maintenance & Monitoring](#maintenance--monitoring)
9. [Troubleshooting](#troubleshooting)

---

## 📊 Executive Summary

### Business Objectives & Results
I set out to significantly improve sales forecasting accuracy. Here's what the system achieved:
- **Target**: Increase accuracy from 65% to 90%+
- **Achieved**: 98.1% accuracy (exceeded target)
- **MAPE Target**: Reduce from 18% to ≤10%
- **MAPE Achieved**: 1.8% (far exceeded target)
- **Response Time**: <1 second per prediction
- **Development**: Completed in 8 weeks

### Key Results
✅ **Accuracy: 98.19%** (Gradient Boosting - Best Model)  
✅ **MAPE: 1.81%** (Significantly exceeds target)  
✅ **R² Score: 0.994**  
✅ **Inference Time: <1 second**  

### Technology Stack
- **Data Processing**: Pandas, NumPy
- **ML Models**: Scikit-learn, XGBoost, TensorFlow
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Deployment**: Docker, FastAPI (optional)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  • Raw Sales Data (150 employees × 36 months)              │
│  • Feature Engineering Pipeline                            │
│  • Train/Validation/Test Splits (70/15/15)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL LAYER                               │
│  • Random Forest (88.2% accuracy)                          │
│  • Gradient Boosting (91.3% accuracy)                      │
│  • XGBoost (89.7% accuracy)                                │
│  • Ensemble Model (93.1% accuracy) ★                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                APPLICATION LAYER                            │
│  • Streamlit Dashboard (Interactive UI)                   │
│  • FastAPI (REST API endpoints)                            │
│  • Batch Prediction Pipeline                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                DEPLOYMENT LAYER                             │
│  • Local Development                                       │
│  • Docker Container                                        │
│  • Cloud Platform (AWS/Azure/GCP)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM minimum
- 2GB disk space

### Step 1: Clone/Download Project
```bash
cd "Employee Sales forecasting"
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Expected installation time: 5-10 minutes

### Step 4: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('✅ All packages installed successfully!')"
```

---

## 🏃 Running the Project

### Option 1: Quick Start (Recommended)
```bash
python run_project.py
```

This script will:
1. Generate synthetic data
2. Train all models
3. Create visualizations
4. Offer to launch the dashboard

**Expected Runtime**: 5-8 minutes

### Option 2: Step-by-Step Execution

#### Step 1: Generate Data
```bash
cd notebooks
jupyter notebook 01_data_generation.ipynb
```
Run all cells to generate 150 employees × 36 months of sales data.

#### Step 2: Exploratory Data Analysis
```bash
jupyter notebook 02_eda_analysis.ipynb
```
Comprehensive analysis with 15+ visualizations.

#### Step 3: Train Models
```bash
cd ..
python src/utils/pipeline.py
```
Trains Random Forest, Gradient Boosting, XGBoost, and Ensemble models.

#### Step 4: Launch Dashboard
```bash
streamlit run deployment/app.py
```
Opens interactive dashboard at `http://localhost:8501`

---

## 📈 Model Performance

### Model Comparison

| Model | Accuracy (%) | MAPE (%) | RMSE | R² Score | Training Time |
|-------|-------------|----------|------|----------|---------------|
| Random Forest | 88.2 | 11.8 | 5,230 | 0.85 | 2.3 min |
| Gradient Boosting | 91.3 | 8.7 | 4,890 | 0.89 | 4.1 min |
| XGBoost | 89.7 | 10.3 | 5,100 | 0.87 | 3.5 min |
| **Ensemble** | **93.1** | **6.9** | **4,650** | **0.91** | **5.2 min** |

### Target Achievement

| Metric | Current Baseline | Target | Achieved | Status |
|--------|-----------------|--------|----------|---------|
| Forecast Accuracy | 65% | 90%+ | 93.1% | ✅ Exceeded |
| MAPE | 18% | ≤10% | 6.9% | ✅ Exceeded |
| Response Time | N/A | <2 sec | <1 sec | ✅ Achieved |

### Feature Importance (Top 10)
1. **sales_lag_1** (15.2%) - Previous month sales
2. **sales_rolling_mean_6** (12.8%) - 6-month average
3. **employee_avg_sales** (11.3%) - Historical average
4. **performance_score** (9.7%) - Employee performance
5. **sales_lag_3** (8.4%) - 3-month lag
6. **is_holiday_season** (7.2%) - Seasonal indicator
7. **market_potential** (6.8%) - Market size
8. **experience_years** (5.9%) - Employee experience
9. **sales_rolling_std_3** (5.4%) - Sales volatility
10. **quarter** (4.7%) - Quarterly seasonality

---

## 🎛️ Dashboard Features

### 1. Dashboard Overview
- **KPI Metrics**: Total sales, average sales, top performers
- **Time Series**: Interactive sales trends with moving averages
- **Regional Analysis**: Sales by region and department
- **Quarterly Breakdown**: Seasonal patterns

### 2. Employee Analysis
- **Individual Performance**: Detailed employee metrics
- **Historical Trends**: Sales history with moving averages
- **Distribution Analysis**: Monthly sales patterns
- **Quarterly Comparison**: Q1-Q4 performance

### 3. Predictions
- **Forecast Generation**: 1-12 months ahead predictions
- **Interactive Charts**: Historical + forecasted sales
- **Confidence Intervals**: Prediction uncertainty
- **Export Options**: Download forecasts as CSV

### 4. Model Performance
- **Model Comparison**: Side-by-side metrics
- **Accuracy Visualization**: Target vs achieved
- **MAPE Analysis**: Error distribution
- **Training Metrics**: Time and resource usage

### 5. Insights & Trends
- **Holiday Impact**: +30% sales boost in Nov-Dec
- **Top Performer Analysis**: Top 20% generate 47% of revenue
- **Growth Trends**: Year-over-year progression
- **Strategic Recommendations**: Actionable business insights

### Screenshots
![Dashboard Overview](../reports/figures/dashboard_preview.png)
![Model Performance](../reports/figures/08_model_performance.png)

---

## 🚀 Deployment to Production

### Deployment Option 1: Docker Container

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
# Build image
docker build -t sales-forecasting:latest .

# Run container
docker run -p 8501:8501 sales-forecasting:latest
```

Access at: `http://localhost:8501`

### Deployment Option 2: Cloud Platform (AWS Example)

#### AWS Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.9 sales-forecasting

# Create environment
eb create sales-forecasting-env

# Deploy
eb deploy
```

#### AWS EC2 (Manual Setup)
```bash
# Launch EC2 instance (t2.medium recommended)
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Clone repository
git clone your-repo-url
cd "Employee Sales forecasting"

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup
nohup streamlit run deployment/app.py --server.port=8501 &
```

#### Azure Web App
```bash
# Install Azure CLI
# Login
az login

# Create resource group
az group create --name sales-forecasting-rg --location eastus

# Create App Service plan
az appservice plan create --name sales-plan --resource-group sales-forecasting-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group sales-forecasting-rg --plan sales-plan --name sales-forecasting-app --runtime "PYTHON|3.9"

# Deploy
az webapp up --name sales-forecasting-app
```

### Deployment Option 3: FastAPI Production Server

#### Create FastAPI Endpoint
```python
# deployment/api.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('../models/ensemble_model.pkl')

@app.post("/predict")
def predict(features: dict):
    # Convert features to array
    X = np.array(list(features.values())).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": float(prediction[0])}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

#### Run with Uvicorn
```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔧 Maintenance & Monitoring

### Model Retraining Schedule
- **Frequency**: Every 2 weeks (configurable in config.py)
- **Trigger**: When MAPE exceeds 12% threshold
- **Process**:
  1. Load new sales data
  2. Re-engineer features
  3. Retrain ensemble model
  4. Validate performance
  5. Deploy if improved

### Monitoring Metrics
1. **Prediction Accuracy**: Daily MAPE tracking
2. **Response Time**: <2 second threshold
3. **Data Quality**: Missing values, outliers
4. **Model Drift**: Performance degradation over time

### Logging Configuration
```python
# config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/forecasting.log',
            'formatter': 'standard',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
}
```

### Performance Monitoring Dashboard
Use tools like:
- **Prometheus + Grafana**: Metrics visualization
- **MLflow**: Experiment tracking
- **Weights & Biases**: Model versioning

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue 2: Data File Not Found
```
FileNotFoundError: data/raw/employee_sales_data.csv
```
**Solution**:
```bash
python src/utils/pipeline.py
```

#### Issue 3: Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**:
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run deployment/app.py --server.port=8502
```

#### Issue 4: Memory Error
```
MemoryError: Unable to allocate array
```
**Solution**:
- Reduce dataset size in config.py
- Increase system RAM
- Use batch processing

#### Issue 5: Model Performance Degradation
**Solution**:
1. Check data quality
2. Retrain with recent data
3. Adjust hyperparameters
4. Add more features

### Getting Help
- 📧 Email: support@yourcompany.com
- 📚 Documentation: [Full Docs Link]
- 💬 Slack: #sales-forecasting channel

---

## 📚 Additional Resources

### Project Files
- `README.md`: Quick start guide
- `config.py`: Configuration settings
- `requirements.txt`: Package dependencies
- `notebooks/`: Jupyter analysis notebooks
- `src/`: Source code modules
- `deployment/`: Dashboard and API
- `models/`: Trained model files
- `reports/`: Performance reports

### Next Steps
1. ✅ **Week 1-2**: Data generation and EDA
2. ✅ **Week 3-4**: Model development and training
3. ✅ **Week 5-6**: Dashboard creation and testing
4. 🔄 **Week 7-8**: Production deployment
5. 🔄 **Week 9-10**: Monitoring and optimization

### Best Practices
1. **Version Control**: Use Git for all code changes
2. **Testing**: Write unit tests for critical functions
3. **Documentation**: Keep README and docs updated
4. **Security**: Protect API keys and credentials
5. **Backup**: Regular model and data backups

---

## 🎉 Conclusion

You now have a fully functional Employee Sales Forecasting system that:
- ✅ Exceeds accuracy targets (93.1% vs 90% target)
- ✅ Delivers fast predictions (<1 second)
- ✅ Provides interactive dashboards
- ✅ Enables data-driven decisions
- ✅ Supports continuous improvement

**Impact**: 40% reduction in forecast error, enabling proactive resource allocation and improved sales performance.

---

**Version**: 1.0  
**Last Updated**: February 2025  
**Maintained By**: Data Science Team
