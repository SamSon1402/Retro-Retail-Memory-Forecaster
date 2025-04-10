import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def prophet_forecast(data, forecast_days=30, include_holidays=True, include_promotions=True, include_oos=True):
    """
    Train a Prophet model and generate forecasts.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with features already engineered
    forecast_days : int
        Number of days to forecast ahead
    include_holidays : bool
        Whether to include holiday effects
    include_promotions : bool
        Whether to include promotion effects
    include_oos : bool
        Whether to include out-of-stock effects
        
    Returns:
    --------
    tuple
        (model, forecast_df)
    """
    # Prepare data for Prophet
    prophet_data = data[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    
    # Ensure we have enough data to train the model
    if len(prophet_data) < 2:
        raise ValueError("Not enough data points for forecasting. Need at least 2 data points.")
        
    # Initialize model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Track what regressors we've added
    regressors_added = []
    
    # Add holidays if included
    if include_holidays and 'Holiday' in data.columns:
        holidays_df = data[data['Holiday'] == 1][['Date']].rename(columns={'Date': 'ds'})
        holidays_df['holiday'] = 'custom_holiday'
        model.add_country_holidays(country_name='US')
    
    # Add regressor for promotions
    if include_promotions and 'Promotion' in data.columns:
        prophet_data['promotion'] = data['Promotion']
        model.add_regressor('promotion')
        regressors_added.append('promotion')
    
    # Add regressor for OOS
    if include_oos and 'OOS' in data.columns:
        prophet_data['oos'] = data['OOS']
        model.add_regressor('oos')
        regressors_added.append('oos')
    
    # Store the regressors we added for later reference
    model.extra_regressors_added = regressors_added
    
    # Save the original prophet_data for later use in predictions
    model.training_data = prophet_data.copy()
    
    # Fit model
    model.fit(prophet_data)
    
    # Create future dataframe
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add regressors to future dataframe
    if include_promotions and 'Promotion' in data.columns and 'promotion' in model.extra_regressors_added:
        # Randomly assign promotions to future dates (10% chance)
        future_df['promotion'] = np.random.choice([0, 1], size=len(future_df), p=[0.9, 0.1])
    
    if include_oos and 'OOS' in data.columns and 'oos' in model.extra_regressors_added:
        # Initialize OOS as 0 for future dates (assuming no stockouts in forecast)
        future_df['oos'] = 0
    
    # Make future predictions
    forecast = model.predict(future_df)
    
    return model, forecast

def ml_forecast(data, model_type='lightgbm', forecast_days=30, include_holidays=True, include_promotions=True, include_oos=True):
    """
    Train a machine learning model (LightGBM or Random Forest) and generate forecasts.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with features already engineered
    model_type : str
        Type of model to use ('lightgbm' or 'random_forest')
    forecast_days : int
        Number of days to forecast ahead
    include_holidays : bool
        Whether to include holiday features
    include_promotions : bool
        Whether to include promotion features
    include_oos : bool
        Whether to include out-of-stock features
        
    Returns:
    --------
    tuple
        (model, future_df, test_preds, y_test, mape, rmse, feature_importance)
    """
    # Prepare features
    feature_cols = ['DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear']
    
    # Add lag and rolling features
    lag_cols = [col for col in data.columns if 'Lag' in col or 'Rolling' in col]
    feature_cols.extend(lag_cols)
    
    # Add other features based on toggles
    if include_holidays and 'Holiday' in data.columns:
        feature_cols.append('Holiday')
    
    if include_promotions and 'Promotion' in data.columns:
        feature_cols.append('Promotion')
    
    if include_oos and 'OOS' in data.columns:
        feature_cols.append('OOS')
    
    # Prepare X and y
    X = data[feature_cols]
    y = data['UnitsSold']
    
    # Split data
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    if model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            verbose=-1
        )
    else:  # Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set for validation
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    # Prepare data for future prediction
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Year'] = future_df['Date'].dt.year
    future_df['DayOfMonth'] = future_df['Date'].dt.day
    future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week
    
    # Create lag features for future prediction using the last known values
    last_values = data.tail(30)['UnitsSold'].values
    
    for i, lag in enumerate([1, 3, 7, 14]):
        if f'Sales_Lag_{lag}' in feature_cols:
            # For the first future date, use the last known values
            future_df.loc[0, f'Sales_Lag_{lag}'] = last_values[-lag] if len(last_values) >= lag else 0
    
    # For rolling features, use the last calculated values for the first future date
    for window in [3, 7, 14, 30]:
        if f'Sales_Rolling_Mean_{window}' in feature_cols:
            future_df.loc[0, f'Sales_Rolling_Mean_{window}'] = data[f'Sales_Rolling_Mean_{window}'].iloc[-1]
            future_df.loc[0, f'Sales_Rolling_Max_{window}'] = data[f'Sales_Rolling_Max_{window}'].iloc[-1]
            future_df.loc[0, f'Sales_Rolling_Min_{window}'] = data[f'Sales_Rolling_Min_{window}'].iloc[-1]
    
    # Add holiday flags
    if include_holidays and 'Holiday' in feature_cols:
        holidays = ['2022-01-01', '2022-12-25', '2022-07-04', '2022-11-24', '2022-11-25',
                    '2023-01-01', '2023-12-25', '2023-07-04', '2023-11-23', '2023-11-24']
        future_df['Holiday'] = future_df['Date'].isin(pd.to_datetime(holidays)).astype(int)
    
    # Add promotion flags (randomly for demonstration)
    if include_promotions and 'Promotion' in feature_cols:
        future_df['Promotion'] = np.random.choice([0, 1], size=len(future_df), p=[0.9, 0.1])
    
    # Add OOS flags (all 0 for future prediction)
    if include_oos and 'OOS' in feature_cols:
        future_df['OOS'] = 0
    
    # Iteratively predict future values
    future_preds = []
    for i in range(forecast_days):
        if i > 0:
            # Update lag features based on previous predictions
            for lag in [1, 3, 7, 14]:
                if f'Sales_Lag_{lag}' in feature_cols:
                    if i >= lag:
                        future_df.loc[i, f'Sales_Lag_{lag}'] = future_preds[i-lag]
                    else:
                        # For early future dates where we don't have enough predictions yet
                        last_idx = len(last_values) - lag + i
                        future_df.loc[i, f'Sales_Lag_{lag}'] = last_values[last_idx] if last_idx >= 0 else 0
            
            # Update rolling features based on previous predictions and known values
            for window in [3, 7, 14, 30]:
                if f'Sales_Rolling_Mean_{window}' in feature_cols:
                    # Combine historical values with predictions for the rolling window
                    if i < window:
                        values_to_use = list(last_values[-(window-i):]) + future_preds[:i]
                    else:
                        values_to_use = future_preds[i-window:i]
                    
                    future_df.loc[i, f'Sales_Rolling_Mean_{window}'] = np.mean(values_to_use) if values_to_use else 0
                    future_df.loc[i, f'Sales_Rolling_Max_{window}'] = np.max(values_to_use) if values_to_use else 0
                    future_df.loc[i, f'Sales_Rolling_Min_{window}'] = np.min(values_to_use) if values_to_use else 0
        
        # Make prediction for this future date
        future_X = future_df.iloc[i:i+1][feature_cols]
        pred = model.predict(future_X)[0]
        pred = max(0, pred)  # Ensure non-negative predictions
        future_preds.append(pred)
    
    future_df['Prediction'] = future_preds
    
    # Get feature importances
    if model_type == 'lightgbm':
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return model, future_df, test_preds, y_test, mape, rmse, feature_importance