"""
Data generation module for Employee Sales Forecasting
Generates realistic synthetic sales data for 150 employees over 36 months
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class SalesDataGenerator:
    """Generate realistic employee sales data with patterns and seasonality"""
    
    def __init__(self, n_employees: int = 150, n_months: int = 36, random_seed: int = 42):
        """
        Initialize data generator
        
        Args:
            n_employees: Number of employees to generate data for
            n_months: Number of months of historical data
            random_seed: Random seed for reproducibility
        """
        self.n_employees = n_employees
        self.n_months = n_months
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_employee_profiles(self) -> pd.DataFrame:
        """Generate static employee profile information"""
        
        departments = ['North', 'South', 'East', 'West', 'Central']
        regions = ['Region_A', 'Region_B', 'Region_C', 'Region_D', 'Region_E']
        
        employees = []
        for emp_id in range(1, self.n_employees + 1):
            employee = {
                'employee_id': f'EMP{emp_id:04d}',
                'department': np.random.choice(departments),
                'region': np.random.choice(regions),
                'experience_years': np.random.randint(1, 20),
                'base_performance': np.random.uniform(0.6, 1.4),  # Performance multiplier
                'territory_size': np.random.randint(50, 500),
                'market_potential': np.random.uniform(0.5, 1.5),
                'competition_level': np.random.uniform(0.3, 1.0)
            }
            employees.append(employee)
            
        return pd.DataFrame(employees)
    
    def generate_time_features(self) -> pd.DataFrame:
        """Generate time-based features"""
        
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=30*i) for i in range(self.n_months)]
        
        time_df = pd.DataFrame({
            'date': dates,
            'month': [d.month for d in dates],
            'quarter': [(d.month - 1) // 3 + 1 for d in dates],
            'year': [d.year for d in dates],
            'is_holiday_season': [1 if d.month in [11, 12] else 0 for d in dates],
            'is_q4': [1 if d.month in [10, 11, 12] else 0 for d in dates]
        })
        
        return time_df
    
    def generate_sales_data(self) -> pd.DataFrame:
        """Generate complete sales dataset with realistic patterns"""
        
        employees_df = self.generate_employee_profiles()
        time_df = self.generate_time_features()
        
        all_records = []
        
        for _, employee in employees_df.iterrows():
            emp_id = employee['employee_id']
            base_perf = employee['base_performance']
            market_pot = employee['market_potential']
            
            # Base monthly sales influenced by employee characteristics
            base_sales = 50000 * base_perf * market_pot
            
            for month_idx, time_row in time_df.iterrows():
                # Add seasonality (Q4 boost)
                seasonal_factor = 1.3 if time_row['is_holiday_season'] else 1.0
                
                # Add monthly trend (gradual growth)
                trend_factor = 1 + (month_idx * 0.005)
                
                # Add random noise
                noise = np.random.normal(1, 0.15)
                
                # Calculate sales
                sales = base_sales * seasonal_factor * trend_factor * noise
                
                # Add training impact (increases over time)
                training_hours = np.random.randint(0, 40) if month_idx % 6 == 0 else 0
                training_boost = 1 + (training_hours * 0.002)
                sales *= training_boost
                
                # Performance score (0-100)
                performance_score = min(100, max(50, 
                    70 + (base_perf - 1) * 30 + np.random.normal(0, 5)))
                
                record = {
                    'employee_id': emp_id,
                    'date': time_row['date'],
                    'month': time_row['month'],
                    'quarter': time_row['quarter'],
                    'year': time_row['year'],
                    'is_holiday_season': time_row['is_holiday_season'],
                    'is_q4': time_row['is_q4'],
                    'sales': max(0, sales),
                    'training_hours': training_hours,
                    'performance_score': performance_score,
                    'deals_closed': np.random.poisson(sales / 10000),
                    'customer_meetings': np.random.randint(20, 100)
                }
                all_records.append(record)
        
        sales_df = pd.DataFrame(all_records)
        
        # Merge with employee profiles
        final_df = sales_df.merge(employees_df, on='employee_id', how='left')
        
        return final_df
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save generated data to CSV"""
        df.to_csv(filepath, index=False)
        print(f"✅ Data saved to: {filepath}")
        print(f"📊 Shape: {df.shape}")
        print(f"👥 Employees: {df['employee_id'].nunique()}")
        print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
        
    def generate_and_save(self, filepath: str) -> pd.DataFrame:
        """Generate and save complete dataset"""
        print("🔄 Generating employee sales data...")
        df = self.generate_sales_data()
        self.save_data(df, filepath)
        return df


def create_train_test_split(df: pd.DataFrame, test_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on time
    
    Args:
        df: Full dataset
        test_months: Number of months to use for testing
        
    Returns:
        train_df, test_df
    """
    df_sorted = df.sort_values('date')
    split_date = df_sorted['date'].max() - timedelta(days=30*test_months)
    
    train_df = df_sorted[df_sorted['date'] <= split_date].copy()
    test_df = df_sorted[df_sorted['date'] > split_date].copy()
    
    print(f"📊 Train set: {train_df.shape} | Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"📊 Test set: {test_df.shape} | Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    generator = SalesDataGenerator(n_employees=150, n_months=36)
    data = generator.generate_and_save("../data/raw/employee_sales_data.csv")
