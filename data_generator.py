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
    if pd.isna(value) or value is None:
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
    if pd.isna(value) or value is None:
        return "0"
    return f"{int(value):,}"


def format_percentage(value):
    """Format number as percentage"""
    if pd.isna(value) or value is None:
        return "0%"
    # Giữ 1 chữ số thập phân, bảo vệ giá trị
    return f"{max(0, value):.1f}%" 


def calculate_completion_rate(actual, planned):
    """Calculate completion rate percentage"""
    if planned == 0 or pd.isna(planned) or planned is None:
        return 0
    return (actual / planned) * 100


def get_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0 or pd.isna(previous) or previous is None:
        return 0
    return ((current - previous) / previous) * 100


def filter_data_by_date(df, start_date, end_date, date_column='booking_date'):
    """Filter dataframe by date range"""
    
    # Kiểm tra df có rỗng không
    if df.empty or date_column not in df.columns:
        return pd.DataFrame(columns=df.columns)
        
    mask = (df[date_column] >= pd.to_datetime(start_date)) & \
           (df[date_column] <= pd.to_datetime(end_date))
    return df[mask].copy()


def filter_confirmed_bookings(df):
    """Filter only confirmed bookings (exclude cancelled/postponed)"""
    if 'status' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df['status'] == 'Đã xác nhận'].copy()


def calculate_kpis(tours_df, plans_df, start_date, end_date):
    """
    Calculate key performance indicators for the dashboard
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
                (plans_df['month'].between(start_dt.month, end_dt.month)) # Đã sửa: dùng .between
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

# ... (các hàm khác)

def get_route_detailed_table(tours_df, plans_df, start_date, end_date):
    """Calculates detailed route performance metrics."""
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)

    if confirmed_data.empty: 
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'planned_revenue', 'revenue_completion'])

    actual_by_route = confirmed_data.groupby('route').agg(
        revenue=('revenue', 'sum'),
        num_customers=('num_customers', 'sum'),
        gross_profit=('gross_profit', 'sum')
    ).reset_index()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    plan_mask = (plans_df['year'] == start_dt.year) & (plans_df['month'].between(start_dt.month, end_dt.month))
    period_plans = plans_df[plan_mask]
    
    plan_by_route = period_plans.groupby('route').agg(
        planned_revenue=('planned_revenue', 'sum')
    ).reset_index()

    table = actual_by_route.merge(plan_by_route, on='route', how='left').fillna(0)
    
    table['profit_margin'] = np.where(table['revenue'] > 0, (table['gross_profit'] / table['revenue'] * 100), 0)
    table['revenue_completion'] = np.where(table['planned_revenue'] > 0, (table['revenue'] / table['planned_revenue'] * 100), 0)

    return table

def get_unit_detailed_table(tours_df, plans_df, start_date, end_date):
    """Calculates detailed business unit performance metrics."""
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)

    if confirmed_data.empty: 
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'avg_revenue_per_customer'])

    actual_by_unit = confirmed_data.groupby('business_unit').agg(
        revenue=('revenue', 'sum'),
        num_customers=('num_customers', 'sum'),
        gross_profit=('gross_profit', 'sum')
    ).reset_index()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    plan_mask = (plans_df['year'] == start_dt.year) & (plans_df['month'].between(start_dt.month, end_dt.month))
    period_plans = plans_df[plan_mask]
    
    plan_by_unit = period_plans.groupby('business_unit').agg(
        planned_revenue=('planned_revenue', 'sum')
    ).reset_index()

    table = actual_by_unit.merge(plan_by_unit, on='business_unit', how='left').fillna(0)
    
    table['profit_margin'] = np.where(table['revenue'] > 0, (table['gross_profit'] / table['revenue'] * 100), 0)
    table['avg_revenue_per_customer'] = np.where(table['num_customers'] > 0, (table['revenue'] / table['num_customers']), 0)
    
    return table

# ... (Vui lòng tham khảo code utils.py đầy đủ ở phần trước. Tôi chỉ hiển thị các hàm đã sửa lỗi)