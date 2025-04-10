import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Define retro gaming color scheme
RETRO_COLORS = {
    'background': '#0A1419',
    'text': '#F8F8F8',
    'primary': '#33FF33',  # Neon green
    'secondary': '#33CCFF',  # Cyan
    'accent': '#FF3366',  # Pink
    'highlight': '#FFCC00',  # Gold
    'dark': '#0F1D24',
    'grid': '#333366'
}

def apply_retro_style(fig):
    """
    Apply retro gaming style to a Plotly figure.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to style
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Styled figure
    """
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=RETRO_COLORS['background'],
        paper_bgcolor=RETRO_COLORS['background'],
        font=dict(family="VT323", size=16, color=RETRO_COLORS['primary']),
        title_font=dict(family="VT323", size=24, color=RETRO_COLORS['primary']),
        legend=dict(
            font=dict(family="VT323", size=14, color=RETRO_COLORS['highlight']),
            bgcolor=RETRO_COLORS['dark'],
            bordercolor=RETRO_COLORS['secondary']
        ),
        xaxis=dict(
            title_font=dict(family="VT323", size=18, color=RETRO_COLORS['secondary']),
            tickfont=dict(family="VT323", size=14, color=RETRO_COLORS['highlight']),
            gridcolor=RETRO_COLORS['grid'],
            linecolor=RETRO_COLORS['secondary']
        ),
        yaxis=dict(
            title_font=dict(family="VT323", size=18, color=RETRO_COLORS['secondary']),
            tickfont=dict(family="VT323", size=14, color=RETRO_COLORS['highlight']),
            gridcolor=RETRO_COLORS['grid'],
            linecolor=RETRO_COLORS['secondary']
        )
    )
    return fig

def create_sales_chart(data, title="Sales History"):
    """
    Create a time series chart of historical sales.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing Date and UnitsSold columns
    title : str
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Sales history chart
    """
    fig = px.line(
        data, 
        x='Date', 
        y='UnitsSold',
        title=title
    )
    
    # Add promotion markers if available
    if 'Promotion' in data.columns:
        promo_dates = data[data['Promotion'] == 1]['Date']
        promo_sales = data[data['Promotion'] == 1]['UnitsSold']
        
        if not promo_dates.empty:
            fig.add_trace(
                go.Scatter(
                    x=promo_dates,
                    y=promo_sales,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=RETRO_COLORS['accent'],
                        symbol='star',
                        line=dict(width=2, color='#FFFFFF')
                    ),
                    name='Promotion'
                )
            )
    
    return apply_retro_style(fig)

def create_forecast_chart(historical_data, forecast_data, model_type="Prophet", is_prophet=True):
    """
    Create a chart showing historical data and forecast.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing historical sales data
    forecast_data : pandas.DataFrame
        DataFrame containing forecast data
    model_type : str
        Type of model used for forecasting
    is_prophet : bool
        Whether the forecast is from a Prophet model or ML model
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Forecast chart
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['UnitsSold'],
            mode='lines',
            name='Historical Sales',
            line=dict(color=RETRO_COLORS['primary'], width=2)
        )
    )
    
    # Add forecast
    if is_prophet:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color=RETRO_COLORS['secondary'], width=3)
            )
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_data['ds'], forecast_data['ds'].iloc[::-1]]),
                y=pd.concat([forecast_data['yhat_upper'], forecast_data['yhat_lower'].iloc[::-1]]),
                fill='toself',
                fillcolor=f"rgba(51, 204, 255, 0.2)",
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Prediction'],
                mode='lines',
                name='Forecast',
                line=dict(color=RETRO_COLORS['secondary'], width=3)
            )
        )
    
    fig.update_layout(
        title=f"Sales Forecast with {model_type}"
    )
    
    return apply_retro_style(fig)

def create_feature_importance_chart(feature_importance):
    """
    Create a bar chart of feature importances.
    
    Parameters:
    -----------
    feature_importance : pandas.DataFrame
        DataFrame with Feature and Importance columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Feature importance chart
    """
    # Sort and get top 10 features
    feature_importance = feature_importance.sort_values('Importance', ascending=True).tail(10)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Feature Importance"
    )
    
    # Update bar colors with pixel gradient
    fig.update_traces(
        marker_color=px.colors.sequential.Plasma,
        marker_line_color='#FFFFFF',
        marker_line_width=1
    )
    
    return apply_retro_style(fig)

def create_actual_vs_predicted_chart(y_true, y_pred):
    """
    Create a scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Actual vs Predicted chart
    """
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    fig = px.scatter(
        df,
        x='Actual',
        y='Predicted',
        title="Actual vs Predicted Values"
    )
    
    # Add diagonal line for perfect predictions
    max_val = max(df['Actual'].max(), df['Predicted'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color=RETRO_COLORS['accent'], width=2, dash='dash')
        )
    )
    
    # Update marker style
    fig.update_traces(
        marker=dict(
            size=10,
            color=RETRO_COLORS['secondary'],
            symbol='square',
            line=dict(width=1, color='#FFFFFF')
        ),
        selector=dict(mode='markers')
    )
    
    return apply_retro_style(fig)