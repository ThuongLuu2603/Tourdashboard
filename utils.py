"""
Utility functions for Vietravel Business Intelligence Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


def format_currency(value):
    """Format number as Vietnamese currency (VND)"""
    if pd.isna(value):
        return "0 ₫"
    
    # Convert to billions for readability if > 1 billion
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f} tỷ ₫"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f} triệu ₫"
    else:
        return f"{value:,.0f} ₫"


def format_number(value):
    """Format number with thousand separators"""
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"


def format_percentage(value):
    """Format number as percentage"""
    if pd.isna(value):
        return "0%"
    return f"{value:.1f}%"


def calculate_completion_rate(actual, planned):
    """Calculate completion rate percentage"""
    if planned == 0 or pd.isna(planned):
        return 0
    return (actual / planned) * 100


def get_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0 or pd.isna(previous):
        return 0
    return ((current - previous) / previous) * 100


def filter_data_by_date(df, start_date, end_date, date_column='booking_date'):
    """Filter dataframe by date range"""
    mask = (df[date_column] >= pd.to_datetime(start_date)) & \
           (df[date_column] <= pd.to_datetime(end_date))
    return df[mask].copy()


def filter_confirmed_bookings(df):
    """Filter only confirmed bookings (exclude cancelled/postponed)"""
    return df[df['status'] == 'Đã xác nhận'].copy()


def calculate_kpis(tours_df, plans_df, start_date, end_date):
    """
    Calculate key performance indicators for the dashboard
    
    Args:
        tours_df: DataFrame with tour bookings
        plans_df: DataFrame with plans
        start_date: Start date for calculation
        end_date: End date for calculation
        
    Returns:
        dict: Dictionary with calculated KPIs
    """
    # Filter data for current period
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    # Calculate actual metrics
    actual_revenue = confirmed_data['revenue'].sum()
    actual_gross_profit = confirmed_data['gross_profit'].sum()
    actual_customers = confirmed_data['num_customers'].sum()
    
    # Filter plans for the same period
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    planned_revenue = period_plans['planned_revenue'].sum()
    planned_gross_profit = period_plans['planned_gross_profit'].sum()
    planned_customers = period_plans['planned_customers'].sum()
    
    # Calculate same period last year
    last_year_start = start_dt - timedelta(days=365)
    last_year_end = end_dt - timedelta(days=365)
    last_year_data = filter_data_by_date(tours_df, last_year_start, last_year_end)
    last_year_confirmed = filter_confirmed_bookings(last_year_data)
    
    ly_revenue = last_year_confirmed['revenue'].sum()
    ly_gross_profit = last_year_confirmed['gross_profit'].sum()
    ly_customers = last_year_confirmed['num_customers'].sum()
    
    # Completion rates
    revenue_completion = calculate_completion_rate(actual_revenue, planned_revenue)
    customer_completion = calculate_completion_rate(actual_customers, planned_customers)
    
    # Growth rates
    revenue_growth = get_growth_rate(actual_revenue, ly_revenue)
    profit_growth = get_growth_rate(actual_gross_profit, ly_gross_profit)
    customer_growth = get_growth_rate(actual_customers, ly_customers)
    
    return {
        'actual_revenue': actual_revenue,
        'actual_gross_profit': actual_gross_profit,
        'actual_customers': actual_customers,
        'planned_revenue': planned_revenue,
        'planned_gross_profit': planned_gross_profit,
        'planned_customers': planned_customers,
        'ly_revenue': ly_revenue,
        'ly_gross_profit': ly_gross_profit,
        'ly_customers': ly_customers,
        'revenue_completion': revenue_completion,
        'customer_completion': customer_completion,
        'revenue_growth': revenue_growth,
        'profit_growth': profit_growth,
        'customer_growth': customer_growth
    }


def create_gauge_chart(value, title, max_value=150, threshold=100):
    """Create a gauge chart for completion rate"""
    
    # Determine color based on value
    if value >= threshold:
        color = "#00CC96"  # Green
    elif value >= threshold * 0.8:
        color = "#FFA500"  # Orange
    else:
        color = "#EF553B"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        number = {'suffix': "%", 'font': {'size': 32}},
        gauge = {
            'axis': {'range': [None, max_value], 'ticksuffix': "%"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 0.5], 'color': "#FFE5E5"},
                {'range': [threshold * 0.5, threshold * 0.8], 'color': "#FFF4E5"},
                {'range': [threshold * 0.8, threshold], 'color': "#E5F5E5"},
                {'range': [threshold, max_value], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_bar_chart(data, x, y, title, orientation='v', color=None):
    """Create a bar chart"""
    
    if orientation == 'h':
        fig = px.bar(data, x=y, y=x, orientation='h', title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxis(title="")
        fig.update_yaxis(title="")
    else:
        fig = px.bar(data, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxis(title="")
        fig.update_yaxis(title="")
    
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def create_pie_chart(data, values, names, title):
    """Create a pie chart"""
    
    fig = px.pie(data, values=values, names=names, title=title,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400,
        showlegend=True
    )
    
    return fig


def create_line_chart(data, x, y, title, color=None):
    """Create a line chart"""
    
    fig = px.line(data, x=x, y=y, title=title, color=color,
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_xaxis(title="")
    fig.update_yaxis(title="")
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def get_top_routes(tours_df, n=10, metric='revenue'):
    """
    Get top N routes by specified metric
    
    Args:
        tours_df: DataFrame with tour data
        n: Number of top routes to return
        metric: Metric to rank by ('revenue', 'customers', 'gross_profit')
        
    Returns:
        DataFrame with top routes
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if metric == 'revenue':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('revenue', ascending=False).head(n)
    elif metric == 'customers':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('num_customers', ascending=False).head(n)
    else:  # gross_profit
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('gross_profit', ascending=False).head(n)
    
    # Calculate profit margin
    grouped['profit_margin'] = (grouped['gross_profit'] / grouped['revenue'] * 100).round(2)
    
    return grouped


def calculate_operational_metrics(tours_df):
    """
    Calculate operational metrics
    
    Args:
        tours_df: DataFrame with tour data
        
    Returns:
        dict: Dictionary with operational metrics
    """
    # Average occupancy rate
    confirmed = filter_confirmed_bookings(tours_df)
    total_booked = confirmed['num_customers'].sum()
    total_capacity = confirmed['tour_capacity'].sum()
    avg_occupancy = (total_booked / total_capacity * 100) if total_capacity > 0 else 0
    
    # Cancellation/postponement rate
    total_bookings = len(tours_df)
    cancelled_postponed = len(tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])])
    cancel_rate = (cancelled_postponed / total_bookings * 100) if total_bookings > 0 else 0
    
    # Returning customer rate
    customer_counts = tours_df.groupby('customer_id').size()
    returning_customers = len(customer_counts[customer_counts >= 2])
    total_unique_customers = len(customer_counts)
    returning_rate = (returning_customers / total_unique_customers * 100) if total_unique_customers > 0 else 0
    
    return {
        'avg_occupancy': avg_occupancy,
        'cancel_rate': cancel_rate,
        'returning_rate': returning_rate
    }


def get_low_margin_tours(tours_df, threshold=5):
    """
    Get tours with profit margin below threshold
    
    Args:
        tours_df: DataFrame with tour data
        threshold: Profit margin threshold (default 5%)
        
    Returns:
        DataFrame with low margin tours
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    # Group by route and calculate average margin
    route_margins = confirmed.groupby('route').agg({
        'gross_profit': 'sum',
        'revenue': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    
    route_margins['profit_margin'] = (route_margins['gross_profit'] / route_margins['revenue'] * 100)
    
    # Filter low margin routes
    low_margin = route_margins[route_margins['profit_margin'] < threshold].sort_values('profit_margin')
    
    return low_margin


def get_unit_performance(tours_df, plans_df, start_date, end_date):
    """
    Calculate performance by business unit
    
    Args:
        tours_df: DataFrame with tour data
        plans_df: DataFrame with plans
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with unit performance
    """
    # Filter current period data
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    # Actual by unit
    actual_by_unit = confirmed_data.groupby('business_unit').agg({
        'revenue': 'sum',
        'gross_profit': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    actual_by_unit.columns = ['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers']
    
    # Plans by unit
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    plan_by_unit = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_gross_profit': 'sum',
        'planned_customers': 'sum'
    }).reset_index()
    
    # Merge and calculate completion
    performance = actual_by_unit.merge(plan_by_unit, on='business_unit', how='left')
    performance['revenue_completion'] = (performance['actual_revenue'] / performance['planned_revenue'] * 100)
    performance['customer_completion'] = (performance['actual_customers'] / performance['planned_customers'] * 100)
    performance['profit_margin'] = (performance['actual_profit'] / performance['actual_revenue'] * 100)
    
    return performance
