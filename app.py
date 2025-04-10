import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import timedelta
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import base64
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎮 Retro Retail Forecaster",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for retro gaming aesthetic
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&display=swap');
    
    /* Main Theme */
    .main {
        background-color: #0A1419;
        color: #F8F8F8;
    }
    
    /* Override default Streamlit background */
    .stApp {
        background-color: #0A1419;
    }
    
    /* Sidebar - comprehensive targeting */
    [data-testid="stSidebar"] {
        background-color: #FFD0DC !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #FFD0DC !important;
    }
    
    .sidebar .sidebar-content, 
    [data-testid="stSidebarUserContent"],
    .css-1d391kg, .css-hxt7ib, .e1fqkh3o1 {
        background-color: #FFD0DC !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'VT323', monospace;
        text-transform: uppercase;
        color: #33FF33;
        text-shadow: 2px 2px 0px #005500;
        letter-spacing: 2px;
    }
    
    h1 {
        font-size: 3rem !important;
        border-bottom: 4px solid #33FF33;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h2 {
        font-size: 2rem !important;
        border-left: 4px solid #FF3366;
        padding-left: 10px;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: #33CCFF;
        border-bottom: 2px solid #33CCFF;
        display: inline-block;
    }
    
    /* Specific header overrides for sidebar */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        border-left: none;
        border-bottom: 2px solid #0A1419;
        color: #0A1419 !important;
        text-shadow: 1px 1px 0px #FF3366 !important;
    }
    
    /* Text */
    p, div, span, label {
        font-family: 'Space Mono', monospace;
    }
    
    /* Widgets in sidebar */
    [data-testid="stSidebar"] .stRadio > label, 
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stDateInput > label {
        color: #0A1419 !important;
        font-weight: bold !important;
    }
    
    /* Improve sidebar widget contrast */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        background-color: rgba(10, 20, 25, 0.1) !important;
    }
    
    /* Widget buttons in sidebar */
    [data-testid="stSidebar"] button {
        background-color: #0A1419 !important;
        color: #33FF33 !important;
        border: 2px solid #33FF33 !important;
    }
    
    /* Widgets */
    .stSelectbox, .stSlider, .stButton > button {
        border: 2px solid #33FF33 !important;
        border-radius: 0px !important;
        background-color: #0A1419 !important;
    }
    
    /* Button */
    .stButton > button {
        font-family: 'VT323', monospace !important;
        font-size: 1.2rem !important;
        color: #33FF33 !important;
        text-shadow: 1px 1px 0px #005500 !important;
        padding: 2px 15px !important;
        transition: all 0.1s !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0px 0px 8px #33FF33 !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
    
    /* Data frames */
    .dataframe {
        font-family: 'Space Mono', monospace !important;
        border: 2px solid #FF3366 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #0F1D24 !important;
        border: 2px solid #FFCC00 !important;
        border-radius: 0px !important;
        box-shadow: 3px 3px 0px #996600 !important;
        padding: 10px !important;
    }
    
    /* Card container for metrics */
    .metric-container {
        background-color: #0F1D24;
        border: 2px solid #FFCC00;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 4px 4px 0px #996600;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #FFD0DC !important;
        border-right: 3px solid #33CCFF !important;
    }
    
    /* Sidebar text color */
    .sidebar .sidebar-content * {
        color: #0A1419 !important;
    }
    
    /* Sidebar headers */
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #0A1419 !important;
        text-shadow: 1px 1px 0px #FF3366 !important;
    }
    
    /* Sidebar widgets */
    .sidebar .stRadio > label, 
    .sidebar .stCheckbox > label,
    .sidebar .stSelectbox > label,
    .sidebar .stSlider > label {
        color: #0A1419 !important;
        font-weight: bold !important;
    }
    
    /* Plot background */
    .js-plotly-plot {
        border: 2px solid #33CCFF !important;
        background-color: #0A1419 !important;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load CSS
load_css()

# Function to generate synthetic retail data
def generate_synthetic_data(start_date='2022-01-01', end_date='2023-01-31'):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    store_ids = [1, 2, 3, 4, 5]
    product_ids = [101, 102, 103, 104, 105]
    
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
    return df

# Function to engineer features for ML model
def engineer_features(df, store_id, product_id):
    # Filter data for specific store and product
    data = df[(df['StoreID'] == store_id) & (df['ProductID'] == product_id)].copy()
    
    # Check if we have data for this combination
    if len(data) == 0:
        st.error(f"No data available for Store ID {store_id} and Product ID {product_id}. Please select another combination.")
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
    
    # Create holiday flags (simplified for demo)
    holidays = ['2022-01-01', '2022-12-25', '2022-07-04', '2022-11-24', '2022-11-25',
                '2023-01-01', '2023-12-25', '2023-07-04', '2023-11-23', '2023-11-24']
    data['Holiday'] = data['Date'].isin(pd.to_datetime(holidays)).astype(int)
    
    # Drop NaN values (from lag and rolling features)
    data = data.dropna()
    
    return data

# Function to train and forecast using Prophet
def prophet_forecast(data, forecast_days=30, include_holidays=True, include_promotions=True, include_oos=True):
    # Prepare data for Prophet
    prophet_data = data[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    
    # Initialize model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Add holidays if included
    if include_holidays and 'Holiday' in data.columns:
        holidays_df = data[data['Holiday'] == 1][['Date']].rename(columns={'Date': 'ds'})
        holidays_df['holiday'] = 'custom_holiday'
        model.add_country_holidays(country_name='US')
    
    # Add regressor for promotions
    if include_promotions and 'Promotion' in data.columns:
        prophet_data['promotion'] = data['Promotion']
        model.add_regressor('promotion')
    
    # Add regressor for OOS
    if include_oos and 'OOS' in data.columns:
        prophet_data['oos'] = data['OOS']
        model.add_regressor('oos')
    
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

# Function to train and forecast using ML model (LightGBM or Random Forest)
def ml_forecast(data, model_type='lightgbm', forecast_days=30, include_holidays=True, include_promotions=True, include_oos=True):
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

# Main app layout
def main():
    # Title with retro gaming aesthetic
    st.markdown("""
    <div style="text-align:center">
        <h1>🎮 RETRO RETAIL FORECASTER 🎮</h1>
        <p style="font-family:'VT323', monospace; font-size:1.5rem; margin-top:-20px; color:#FFCC00;">
            LEVEL UP YOUR INVENTORY MANAGEMENT!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center">
            <h2 style="color: #0A1419;">GAME CONTROLS 🕹️</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Data generation/upload section
        st.markdown('<h3 style="color: #0A1419;">DATA SOURCES</h3>', unsafe_allow_html=True)
        data_option = st.radio(
            "Choose your data source:",
            ["Generate Synthetic Data", "Upload CSV File"],
            key="data_source"
        )
        
        if data_option == "Generate Synthetic Data":
            start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
            end_date = st.date_input("End Date", datetime.date(2023, 1, 31))
            
            if st.button("🎲 GENERATE DATA 🎲", key="generate_data"):
                with st.spinner("Generating data..."):
                    df = generate_synthetic_data(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    st.session_state['data'] = df
                    st.success("Data generated successfully!")
        else:
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                # Ensure Date column is in datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                st.session_state['data'] = df
                st.success("Data loaded successfully!")
        
        # Only show the rest of the controls if data is loaded
        if 'data' in st.session_state:
            # Store and Product selection
            st.markdown('<h3 style="color: #0A1419;">SELECTION</h3>', unsafe_allow_html=True)
            
            df = st.session_state['data']
            store_ids = sorted(df['StoreID'].unique())
            product_ids = sorted(df['ProductID'].unique())
            
            selected_store = st.selectbox("Select Store", options=store_ids, key="store")
            selected_product = st.selectbox("Select Product", options=product_ids, key="product")
            
            # Model selection
            st.markdown('<h3 style="color: #0A1419;">MODEL SELECT</h3>', unsafe_allow_html=True)
            model_type = st.radio(
                "Choose your forecasting model:",
                ["Prophet", "LightGBM", "Random Forest"],
                key="model_type"
            )
            
            # Feature toggles
            st.markdown('<h3 style="color: #0A1419;">POWER-UPS</h3>', unsafe_allow_html=True)
            include_holidays = st.toggle("Holiday Effects", value=True, key="holidays")
            include_promotions = st.toggle("Promotion Effects", value=True, key="promotions")
            include_oos = st.toggle("Out-of-Stock Detection", value=True, key="oos")
            
            # Forecast days
            forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1, key="forecast_days")
            
            # Run forecast button
            if st.button("🚀 RUN FORECAST 🚀", key="run_forecast"):
                with st.spinner("Running forecast..."):
                    # Engineer features
                    engineered_data = engineer_features(df, selected_store, selected_product)
                    
                    # Check if data is available
                    if len(engineered_data) == 0:
                        st.error("Cannot run forecast with no data. Please select a different store/product combination.")
                    else:
                        st.session_state['engineered_data'] = engineered_data
                        
                        try:
                            # Run selected model
                            if model_type == "Prophet":
                                model, forecast = prophet_forecast(
                                    engineered_data,
                                    forecast_days=forecast_days,
                                    include_holidays=include_holidays,
                                    include_promotions=include_promotions,
                                    include_oos=include_oos
                                )
                                st.session_state['model_results'] = {
                                    'model': model,
                                    'forecast': forecast,
                                    'type': 'prophet'
                                }
                            else:  # LightGBM or Random Forest
                                model, future_df, test_preds, y_test, mape, rmse, feature_importance = ml_forecast(
                                    engineered_data,
                                    model_type=model_type.lower(),
                                    forecast_days=forecast_days,
                                    include_holidays=include_holidays,
                                    include_promotions=include_promotions,
                                    include_oos=include_oos
                                )
                                st.session_state['model_results'] = {
                                    'model': model,
                                    'future_df': future_df,
                                    'test_preds': test_preds,
                                    'y_test': y_test,
                                    'mape': mape,
                                    'rmse': rmse,
                                    'feature_importance': feature_importance,
                                    'type': model_type.lower()
                                }
                            
                            st.success("Forecast completed!")
                        except Exception as e:
                            st.error(f"An error occurred while running the forecast: {str(e)}")
                            st.info("Try selecting different parameters or another store/product combination.")
    
    # Main content area
    if 'data' in st.session_state:
        # Tab layout for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🔮 Forecast Results", "📈 Performance Metrics"])
        
        with tab1:
            st.markdown("<h2>DATA EXPLORER</h2>", unsafe_allow_html=True)
            
            df = st.session_state['data']
            
            # Display summary statistics
            st.markdown("<h3>DATASET STATS</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
            with col3:
                st.metric("Unique Products", f"{df['ProductID'].nunique()}")
            
            # Display sample data
            st.markdown("<h3>SAMPLE DATA</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), height=300)
            
            # Display time series of sales for selected store/product
            st.markdown("<h3>SALES HISTORY</h3>", unsafe_allow_html=True)
            
            if 'store' in st.session_state and 'product' in st.session_state:
                filtered_df = df[(df['StoreID'] == st.session_state['store']) & 
                                 (df['ProductID'] == st.session_state['product'])]
                
                if not filtered_df.empty:
                    # Create a Plotly time series chart with retro styling
                    fig = px.line(
                        filtered_df, 
                        x='Date', 
                        y='UnitsSold',
                        title=f"Sales History for Store {st.session_state['store']} - Product {st.session_state['product']}"
                    )
                    
                    # Update layout for retro gaming aesthetic
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0A1419",
                        paper_bgcolor="#0A1419",
                        font=dict(family="VT323", size=16, color="#33FF33"),
                        title=dict(font=dict(family="VT323", size=24, color="#33FF33")),
                        xaxis=dict(
                            title="Date",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        ),
                        yaxis=dict(
                            title="Units Sold",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        )
                    )
                    
                    # Add promotion markers
                    if 'Promotion' in filtered_df.columns:
                        promo_dates = filtered_df[filtered_df['Promotion'] == 1]['Date']
                        promo_sales = filtered_df[filtered_df['Promotion'] == 1]['UnitsSold']
                        
                        if not promo_dates.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=promo_dates,
                                    y=promo_sales,
                                    mode='markers',
                                    marker=dict(
                                        size=10,
                                        color='#FF3366',
                                        symbol='star',
                                        line=dict(width=2, color='#FFFFFF')
                                    ),
                                    name='Promotion'
                                )
                            )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display sales by day of week
                    st.markdown("<h3>SALES BY DAY OF WEEK</h3>", unsafe_allow_html=True)
                    
                    # Add day of week
                    filtered_df['DayOfWeek'] = filtered_df['Date'].dt.dayofweek
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    filtered_df['DayName'] = filtered_df['DayOfWeek'].apply(lambda x: days[x])
                    
                    dow_sales = filtered_df.groupby('DayName')['UnitsSold'].mean().reindex(days)
                    
                    fig = px.bar(
                        x=dow_sales.index,
                        y=dow_sales.values,
                        title="Average Sales by Day of Week",
                        labels={'x': 'Day of Week', 'y': 'Average Units Sold'}
                    )
                    
                    # Update layout for retro gaming aesthetic
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0A0A1A",
                        paper_bgcolor="#0A0A1A",
                        font=dict(family="VT323", size=16, color="#33FF33"),
                        title=dict(font=dict(family="VT323", size=24, color="#33FF33")),
                        xaxis=dict(
                            title="Day of Week",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        ),
                        yaxis=dict(
                            title="Average Units Sold",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        )
                    )
                    
                    # Update bar colors with retro palette
                    fig.update_traces(
                        marker_color=['#33FF33', '#33CCFF', '#FF3366', '#FFCC00', '#CC66FF', '#FF6633', '#00FFCC'],
                        marker_line_color='#FFFFFF',
                        marker_line_width=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for the selected store and product combination.")
        
        with tab2:
            st.markdown("<h2>FORECAST RESULTS</h2>", unsafe_allow_html=True)
            
            if 'model_results' in st.session_state:
                results = st.session_state['model_results']
                model_type = results['type']
                
                if model_type == 'prophet':
                    model = results['model']
                    forecast = results['forecast']
                    
                    # Display forecast plot
                    st.markdown("<h3>SALES FORECAST</h3>", unsafe_allow_html=True)
                    
                    # Get historical data
                    historical_data = st.session_state['engineered_data']
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data['Date'],
                            y=historical_data['UnitsSold'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='#33FF33', width=2)
                        )
                    )
                    
                    # Add forecast
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#33CCFF', width=3)
                        )
                    )
                    
                    # Add confidence intervals
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
                            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'].iloc[::-1]]),
                            fill='toself',
                            fillcolor='rgba(51, 204, 255, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="Sales Forecast with Prophet",
                        template="plotly_dark",
                        plot_bgcolor="#0A1419",
                        paper_bgcolor="#0A1419",
                        font=dict(family="VT323", size=16, color="#33FF33"),
                        title_font=dict(family="VT323", size=24, color="#33FF33"),
                        legend=dict(
                            font=dict(family="VT323", size=14, color="#FFCC00"),
                            bgcolor="#0F1D24",
                            bordercolor="#33CCFF"
                        ),
                        xaxis=dict(
                            title="Date",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        ),
                        yaxis=dict(
                            title="Units Sold",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display components (trend, weekly, yearly)
                    st.markdown("<h3>FORECAST COMPONENTS</h3>", unsafe_allow_html=True)
                    
                    # Plot trend component
                    fig_trend = model.plot_components(forecast)
                    
                    # Customize the figure for retro aesthetics
                    for ax in fig_trend.axes:
                        ax.set_facecolor('#0A0A1A')
                        ax.grid(color='#333366', linestyle='--', alpha=0.7)
                        ax.spines['bottom'].set_color('#33CCFF')
                        ax.spines['left'].set_color('#33CCFF')
                        ax.spines['top'].set_color('#33CCFF')
                        ax.spines['right'].set_color('#33CCFF')
                        ax.tick_params(colors='#FFCC00')
                        ax.title.set_color('#33FF33')
                        ax.yaxis.label.set_color('#33CCFF')
                        ax.xaxis.label.set_color('#33CCFF')
                        
                        # Update line colors if present
                        for line in ax.get_lines():
                            line.set_color('#33FF33')
                            line.set_linewidth(2)
                    
                    fig_trend.set_facecolor('#0A0A1A')
                    st.pyplot(fig_trend)
                    
                else:  # LightGBM or Random Forest
                    model = results['model']
                    future_df = results['future_df']
                    test_preds = results['test_preds']
                    y_test = results['y_test']
                    mape = results['mape']
                    rmse = results['rmse']
                    feature_importance = results['feature_importance']
                    
                    # Get historical data
                    historical_data = st.session_state['engineered_data']
                    
                    # Display forecast plot
                    st.markdown("<h3>SALES FORECAST</h3>", unsafe_allow_html=True)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data['Date'],
                            y=historical_data['UnitsSold'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='#33FF33', width=2)
                        )
                    )
                    
                    # Add forecast
                    fig.add_trace(
                        go.Scatter(
                            x=future_df['Date'],
                            y=future_df['Prediction'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#33CCFF', width=3)
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Sales Forecast with {model_type}",
                        template="plotly_dark",
                        plot_bgcolor="#0A1419",
                        paper_bgcolor="#0A1419",
                        font=dict(family="VT323", size=16, color="#33FF33"),
                        title_font=dict(family="VT323", size=24, color="#33FF33"),
                        legend=dict(
                            font=dict(family="VT323", size=14, color="#FFCC00"),
                            bgcolor="#0F1D24",
                            bordercolor="#33CCFF"
                        ),
                        xaxis=dict(
                            title="Date",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        ),
                        yaxis=dict(
                            title="Units Sold",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display feature importance
                    st.markdown("<h3>FEATURE IMPORTANCE</h3>", unsafe_allow_html=True)
                    
                    # Sort feature importance
                    feature_importance = feature_importance.sort_values('Importance', ascending=True).tail(10)
                    
                    # Create feature importance bar chart
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Top 10 Feature Importance - {model_type}"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0A0A1A",
                        paper_bgcolor="#0A0A1A",
                        font=dict(family="VT323", size=16, color="#33FF33"),
                        title_font=dict(family="VT323", size=24, color="#33FF33"),
                        xaxis=dict(
                            title="Importance",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        ),
                        yaxis=dict(
                            title="Feature",
                            title_font=dict(family="VT323", size=18, color="#33CCFF"),
                            tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                            gridcolor="#333366",
                            linecolor="#33CCFF"
                        )
                    )
                    
                    # Update bar colors with pixel gradient
                    fig.update_traces(
                        marker_color=px.colors.sequential.Plasma,
                        marker_line_color='#FFFFFF',
                        marker_line_width=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run a forecast to see results here!")
        
        with tab3:
            st.markdown("<h2>PERFORMANCE METRICS</h2>", unsafe_allow_html=True)
            
            if 'model_results' in st.session_state:
                results = st.session_state['model_results']
                model_type = results['type']
                
                # Display metrics in a retro-styled card layout
                st.markdown("""
                <div style="font-family: 'VT323', monospace; text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #33FF33; text-shadow: 2px 2px #005500;">MODEL PERFORMANCE SCOREBOARD</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if model_type == 'prophet':
                    # Calculate metrics for Prophet
                    forecast = results['forecast']
                    historical_data = st.session_state['engineered_data']
                    
                    # Merge forecast with historical data - use only the dates that exist in historical data
                    # Convert to date for proper merging
                    forecast_copy = forecast.copy()
                    forecast_copy['ds'] = pd.to_datetime(forecast_copy['ds']).dt.date
                    historical_copy = historical_data.copy()
                    historical_copy['Date'] = pd.to_datetime(historical_copy['Date']).dt.date
                    
                    merged_data = pd.merge(
                        historical_copy[['Date', 'UnitsSold']],
                        forecast_copy[['ds', 'yhat']],
                        left_on='Date',
                        right_on='ds',
                        how='inner'
                    )
                    
                    # Check if there's data to calculate metrics
                    if len(merged_data) > 0:
                        # Calculate metrics
                        mape = mean_absolute_percentage_error(merged_data['UnitsSold'], merged_data['yhat'])
                        rmse = np.sqrt(mean_squared_error(merged_data['UnitsSold'], merged_data['yhat']))
                    else:
                        # Handle case with no overlapping data
                        st.warning("No overlapping dates between historical data and forecast. Using training data metrics instead.")
                        
                        try:
                            # Use the stored training data for in-sample predictions
                            if hasattr(model, 'training_data'):
                                # Get in-sample forecast on the training data
                                in_sample_forecast = model.predict(model.training_data[['ds'] + model.extra_regressors_added])
                                
                                # Merge with actual values
                                in_sample_merged = pd.merge(
                                    model.training_data[['ds', 'y']],
                                    in_sample_forecast[['ds', 'yhat']],
                                    on='ds',
                                    how='inner'
                                )
                                
                                if len(in_sample_merged) > 0:
                                    mape = mean_absolute_percentage_error(in_sample_merged['y'], in_sample_merged['yhat'])
                                    rmse = np.sqrt(mean_squared_error(in_sample_merged['y'], in_sample_merged['yhat']))
                                else:
                                    st.error("No matching data found for metrics calculation")
                                    mape = 0.0
                                    rmse = 0.0
                            else:
                                st.error("Training data not available for metrics calculation")
                                mape = 0.0
                                rmse = 0.0
                        except Exception as e:
                            st.error(f"Error calculating metrics: {str(e)}")
                            mape = 0.0
                            rmse = 0.0
                    
                    # Create scatter plot of actual vs predicted if data is available
                    if len(merged_data) > 0:
                        fig = px.scatter(
                            merged_data,
                            x='UnitsSold',
                            y='yhat',
                            title="Actual vs Predicted Values",
                            labels={'UnitsSold': 'Actual', 'yhat': 'Predicted'}
                        )
                        
                        # Add diagonal line for perfect predictions
                        max_val = max(merged_data['UnitsSold'].max(), merged_data['yhat'].max())
                        fig.add_trace(
                            go.Scatter(
                                x=[0, max_val],
                                y=[0, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='#FF3366', width=2, dash='dash')
                            )
                        )
                    else:
                        # Create a basic placeholder chart
                        fig = go.Figure()
                        fig.add_annotation(
                            text="No overlapping data available for comparison",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(family="VT323", size=20, color="#FF3366")
                        )
                        fig.update_layout(
                            title="Actual vs Predicted Values",
                            xaxis_title="Actual",
                            yaxis_title="Predicted"
                        )
                    
                else:  # LightGBM or Random Forest
                    mape = results['mape']
                    rmse = results['rmse']
                    test_preds = results['test_preds']
                    y_test = results['y_test']
                    
                    # Create scatter plot of actual vs predicted
                    fig = px.scatter(
                        x=y_test,
                        y=test_preds,
                        title="Actual vs Predicted Values",
                        labels={'x': 'Actual', 'y': 'Predicted'}
                    )
                    
                    # Add diagonal line for perfect predictions
                    max_val = max(y_test.max(), test_preds.max())
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='#FF3366', width=2, dash='dash')
                        )
                    )
                
                # Update layout for retro gaming aesthetic
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0A1419",
                    paper_bgcolor="#0A1419",
                    font=dict(family="VT323", size=16, color="#33FF33"),
                    title_font=dict(family="VT323", size=24, color="#33FF33"),
                    legend=dict(
                        font=dict(family="VT323", size=14, color="#FFCC00"),
                        bgcolor="#0F1D24",
                        bordercolor="#33CCFF"
                    ),
                    xaxis=dict(
                        title="Actual Units Sold",
                        title_font=dict(family="VT323", size=18, color="#33CCFF"),
                        tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                        gridcolor="#333366",
                        linecolor="#33CCFF"
                    ),
                    yaxis=dict(
                        title="Predicted Units Sold",
                        title_font=dict(family="VT323", size=18, color="#33CCFF"),
                        tickfont=dict(family="VT323", size=14, color="#FFCC00"),
                        gridcolor="#333366",
                        linecolor="#33CCFF"
                    )
                )
                
                # Update marker style
                fig.update_traces(
                    marker=dict(
                        size=10,
                        color='#33CCFF',
                        symbol='square',
                        line=dict(width=1, color='#FFFFFF')
                    ),
                    selector=dict(mode='markers')
                )
                
                # Display metrics in retro-styled cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: #33FF33; text-align: center; margin-top: 0;">MAPE</h3>
                        <div style="font-family: 'VT323', monospace; font-size: 3rem; text-align: center; color: #FFCC00;">
                            {mape:.2%}
                        </div>
                        <p style="font-family: 'Space Mono', monospace; font-size: 0.8rem; text-align: center; color: #33CCFF;">
                            Mean Absolute Percentage Error
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="color: #33FF33; text-align: center; margin-top: 0;">RMSE</h3>
                        <div style="font-family: 'VT323', monospace; font-size: 3rem; text-align: center; color: #FFCC00;">
                            {rmse:.2f}
                        </div>
                        <p style="font-family: 'Space Mono', monospace; font-size: 0.8rem; text-align: center; color: #33CCFF;">
                            Root Mean Square Error
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add some retro-styled text explaining the metrics
                st.markdown("""
                <div style="background-color: #0F1D24; border: 2px solid #33CCFF; padding: 20px; margin-top: 20px; box-shadow: 4px 4px 0px #1A1A4D;">
                    <h3 style="color: #33FF33; text-align: center; margin-top: 0;">METRICS GUIDE</h3>
                    <p style="font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #FFCC00;">
                        <span style="color: #33FF33;">MAPE</span> (Mean Absolute Percentage Error): Shows the average percentage difference between predicted and actual values. Lower is better!
                    </p>
                    <p style="font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #FFCC00;">
                        <span style="color: #33FF33;">RMSE</span> (Root Mean Square Error): Measures the standard deviation of prediction errors. Lower values indicate better accuracy!
                    </p>
                    <p style="font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #FFCC00;">
                        The <span style="color: #FF3366;">diagonal line</span> represents perfect predictions. Points closer to this line indicate more accurate forecasts.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Run a forecast to see performance metrics here!")
    
    else:
        # Display welcome screen if no data is loaded
        st.markdown("""
        <div style="text-align:center; padding: 50px 20px; background-color: #0F1D24; border: 4px solid #33FF33; margin: 20px 0;">
            <h2 style="color: #33FF33; text-shadow: 3px 3px 0px #005500;">WELCOME TO THE RETAIL FORECASTER!</h2>
            <p style="font-family: 'VT323', monospace; font-size: 1.5rem; color: #FFCC00;">
                Start your journey by generating data or uploading a CSV file using the controls in the sidebar.
            </p>
            <div style="margin: 40px 0; font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #33CCFF; line-height: 1.6;">
                This retro-themed forecasting tool will help you:
                <ul style="text-align: left; list-style-type: none; padding-left: 20px;">
                    <li>📊 Explore historical sales patterns</li>
                    <li>🎮 Select stores and products for analysis</li>
                    <li>🧠 Choose between different forecasting models</li>
                    <li>🔮 Generate accurate demand predictions</li>
                    <li>🏆 Evaluate model performance</li>
                </ul>
            </div>
            <p style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #FF3366; text-shadow: 2px 2px 0px #330011;">
                PRESS START TO BEGIN YOUR ADVENTURE!
            </p>
            <div style="font-size: 2rem; margin-top: 30px; animation: blink 1s infinite;">
                ⬅️ SELECT OPTIONS FROM THE SIDEBAR
            </div>
        </div>
        
        <style>
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()