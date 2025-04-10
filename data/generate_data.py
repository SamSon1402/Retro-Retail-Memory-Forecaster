import pandas as pd
import numpy as np
import datetime

def generate_synthetic_data(start_date='2022-01-01', end_date='2023-01-31', 
                           num_stores=5, num_products=5, output_file=None):
    """
    Generate synthetic retail sales data for demonstration purposes.
    
    Parameters:
    -----------
    start_date : str
        Start date for the data in YYYY-MM-DD format
    end_date : str
        End date for the data in YYYY-MM-DD format
    num_stores : int
        Number of stores to generate data for
    num_products : int
        Number of products per store to generate data for
    output_file : str, optional
        If provided, save the generated data to this CSV file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic retail sales data
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    store_ids = list(range(1, num_stores + 1))
    product_ids = list(range(101, 101 + num_products))
    
    rows = []
    
    for date in date_range:
        for store_id in store_ids:
            for product_id in product_ids:
                # Base demand (different for each product)
                base_demand = 20 + (product_id % 100) * 5
                
                # Store effect (larger stores sell more)
                store_effect = store_id * 2
                
                # Day of week effect (weekend boost)
                dow_effect = 5 if date.dayofweek >= 5 else 0
                
                # Monthly seasonality (higher in summer)
                month_effect = 10 * np.sin(np.pi * date.month / 6)
                
                # Trend (slight growth over time)
                days_since_start = (date - pd.Timestamp(start_date)).days
                trend_effect = days_since_start * 0.05
                
                # Random promotion (10% chance)
                promotion = 1 if np.random.random() < 0.1 else 0
                
                # Promotion effect
                promo_effect = 15 if promotion else 0
                
                # Price (base + random variation, lower during promotions)
                base_price = 10 + (product_id % 100)
                price = base_price * (0.8 if promotion else 1.0) * np.random.uniform(0.95, 1.05)
                
                # Calculate final demand
                demand = base_demand + store_effect + dow_effect + month_effect + trend_effect + promo_effect
                
                # Add some noise
                demand = max(0, int(demand * np.random.uniform(0.8, 1.2)))
                
                # Simulate out-of-stock events (5% chance)
                if np.random.random() < 0.05:
                    demand = 0
                    inventory = 0
                else:
                    inventory = max(0, demand + np.random.randint(5, 20))
                
                rows.append({
                    'Date': date,
                    'StoreID': store_id,
                    'ProductID': product_id,
                    'UnitsSold': demand,
                    'Price': round(price, 2),
                    'Promotion': promotion,
                    'InventoryLevel': inventory
                })
    
    df = pd.DataFrame(rows)
    
    # Add a few holiday dates
    holidays = ['2022-01-01', '2022-12-25', '2022-07-04', '2022-11-24', '2022-11-25',
                '2023-01-01', '2023-12-25', '2023-07-04', '2023-11-23', '2023-11-24']
    df['Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate a sample dataset when run as a script
    df = generate_synthetic_data(output_file='sample_retail_data.csv')
    print(f"Generated {len(df)} rows of synthetic retail data")
    
    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Number of Stores: {df['StoreID'].nunique()}")
    print(f"Number of Products: {df['ProductID'].nunique()}")
    print(f"Average Daily Sales: {df['UnitsSold'].mean():.2f} units")
    print(f"Promotion Days: {df['Promotion'].sum()} ({df['Promotion'].mean()*100:.1f}%)")