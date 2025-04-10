import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import engineer_features

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        # Create sample retail data
        date_range = pd.date_range(start='2022-01-01', end='2022-01-31', freq='D')
        store_ids = [1, 2]
        product_ids = [101, 102]
        
        rows = []
        
        for date in date_range:
            for store_id in store_ids:
                for product_id in product_ids:
                    # Create random sales data
                    demand = np.random.randint(10, 50)
                    price = 10 + np.random.randint(0, 10)
                    promotion = np.random.choice([0, 1], p=[0.9, 0.1])
                    inventory = demand + np.random.randint(5, 20)
                    
                    rows.append({
                        'Date': date,
                        'StoreID': store_id,
                        'ProductID': product_id,
                        'UnitsSold': demand,
                        'Price': price,
                        'Promotion': promotion,
                        'InventoryLevel': inventory
                    })
        
        self.sample_data = pd.DataFrame(rows)
        
        # Add holiday column
        holidays = ['2022-01-01', '2022-01-17']  # New Year's Day and MLK Day
        self.sample_data['Holiday'] = self.sample_data['Date'].isin(pd.to_datetime(holidays)).astype(int)
    
    def test_engineer_features(self):
        """Test feature engineering function"""
        # Engineer features for store 1, product 101
        result = engineer_features(self.sample_data, 1, 101)
        
        # Check that result is not empty
        self.assertFalse(result.empty)
        
        # Check that expected features are created
        expected_features = [
            'DayOfWeek', 'Month', 'Year', 'DayOfMonth', 'WeekOfYear',
            'Sales_Lag_1', 'Sales_Lag_3', 'Sales_Lag_7', 'Sales_Lag_14',
            'Sales_Rolling_Mean_3', 'Sales_Rolling_Max_3', 'Sales_Rolling_Min_3',
            'Sales_Rolling_Mean_7', 'Sales_Rolling_Max_7', 'Sales_Rolling_Min_7',
            'Sales_Rolling_Mean_14', 'Sales_Rolling_Max_14', 'Sales_Rolling_Min_14',
            'Sales_Rolling_Mean_30', 'Sales_Rolling_Max_30', 'Sales_Rolling_Min_30',
            'OOS'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Expected feature {feature} is missing")
        
        # Check that NaN values are dropped
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Test with non-existent store/product
        empty_result = engineer_features(self.sample_data, 999, 999)
        self.assertTrue(empty_result.empty)

if __name__ == '__main__':
    unittest.main()