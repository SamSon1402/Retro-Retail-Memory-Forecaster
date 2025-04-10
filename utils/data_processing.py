import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load retail data from a CSV file and perform basic preprocessing.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Ensure Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Fill any missing values (if any)
    if df.isnull().any().any():
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Forward fill for other columns
        df = df.fillna(method='ffill')
    
    return df

def engineer_features(df, store_id, product_id):
    """
    Engineer features for a specific store and product combination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing retail data
    store_id : int
        Store ID to filter data for
    product_id : int
        Product ID to filter data for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    # Filter data for specific store and product
    data = df[(df['StoreID'] == store_id) & (df['ProductID'] == product_id)].copy()
    
    # Check if we have data for this combination
    if len(data) == 0:
        print(f"No data available for Store ID {store_id} and Product ID {product_id}")
        return pd.DataFrame()  # Return empty dataframe
        
    data = data.sort_values('Date')
    
    # Extract date features
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['DayOfMonth'] = data['Date'].dt.day
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    
    # Create lag features
    for lag in [1, 3, 7, 14]:
        data[f'Sales_Lag_{lag}'] = data['UnitsSold'].shift(lag)
    
    # Create rolling window features
    for window in [3, 7, 14, 30]:
        data[f'Sales_Rolling_Mean_{window}'] = data['UnitsSold'].rolling(window=window).mean()
        data[f'Sales_Rolling_Max_{window}'] = data['UnitsSold'].rolling(window=window).max()
        data[f'Sales_Rolling_Min_{window}'] = data['UnitsSold'].rolling(window=window).min()
    
    # Simulate OOS (Out of Stock) flag based on inventory and rolling average
    data['OOS'] = 0
    # Mark as OOS if sales are zero but recent average was high
    rolling_avg = data['UnitsSold'].rolling(window=7).mean().shift(1)
    data.loc[(data['UnitsSold'] == 0) & (rolling_avg > 5), 'OOS'] = 1
    
    # Drop NaN values (from lag and rolling features)
    data = data.dropna()
    
    return data

def get_store_product_performance(df):
    """
    Calculate performance metrics for each store-product combination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing retail data
        
    Returns:
    --------
    pandas.DataFrame
        Summary metrics for each store-product combination
    """
    # Group by store and product
    grouped = df.groupby(['StoreID', 'ProductID']).agg({
        'UnitsSold': ['sum', 'mean', 'std', 'min', 'max'],
        'Price': 'mean',
        'Promotion': 'mean',
        'InventoryLevel': 'mean'
    }).reset_index()
    
    # Flatten the column hierarchy
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Calculate additional metrics
    grouped['Revenue'] = grouped['UnitsSold_sum'] * grouped['Price_mean']
    grouped['Volatility'] = grouped['UnitsSold_std'] / grouped['UnitsSold_mean']
    grouped['Promotion_Percentage'] = grouped['Promotion_mean'] * 100
    
    return grouped.sort_values('Revenue', ascending=False)