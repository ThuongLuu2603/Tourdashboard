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
    # Sửa: Bảo vệ giá trị âm hoặc NaN
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


def create_gauge_chart(value, title, max_value=150, threshold=100, unit_breakdown=None):
    """Create a gauge chart for completion rate with hover info for business units"""
    
    # Determine color based on value
    if value >= threshold:
        color = "#00CC96"  # Green
    elif value >= threshold * 0.8:
        color = "#FFA500"  # Orange
    else:
        color = "#EF553B"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 11}},
        number = {'suffix': "%", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, max_value], 'ticksuffix': "%", 'tickfont': {'size': 9}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 0.5], 'color': "#FFE5E5"},
                {'range': [threshold * 0.5, threshold * 0.8], 'color': "#FFF4E5"},
                {'range': [threshold * 0.8, threshold], 'color': "#E5F5E5"},
                {'range': [threshold, max_value], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    # Add invisible scatter trace for hover info with business unit breakdown
    if unit_breakdown is not None and not unit_breakdown.empty:
        hover_text = "Chi tiết theo đơn vị:<br>"
        for _, row in unit_breakdown.iterrows():
            hover_text += f"<br>{row['business_unit']}: {row['completion']:.1f}%"
        
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.1],
            mode='markers',
            marker=dict(size=100, color='rgba(0,0,0,0)', opacity=0),
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        height=160,
        margin=dict(l=5, r=5, t=30, b=5),
        hovermode='closest'
    )
    
    return fig


def create_bar_chart(data, x, y, title, orientation='v', color=None):
    """Create a bar chart"""
    
    if orientation == 'h':
        fig = px.bar(data, x=y, y=x, orientation='h', title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    else:
        fig = px.bar(data, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    
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
    
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def get_top_routes(tours_df, n=10, metric='revenue'):
    """
    Get top N routes by specified metric
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty: 
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin'])

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
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    grouped['profit_margin'] = np.where(
        grouped['revenue'] > 0,
        (grouped['gross_profit'] / grouped['revenue'] * 100).round(2),
        0
    )
    
    return grouped


def get_route_unit_breakdown(tours_df, route_name):
    """
    Get breakdown by business unit for a specific route
    """
    confirmed = filter_confirmed_bookings(tours_df)
    route_data = confirmed[confirmed['route'] == route_name]
    
    if route_data.empty:
        return pd.DataFrame(columns=['business_unit', 'revenue', 'percentage'])
    
    unit_breakdown = route_data.groupby('business_unit').agg({
        'revenue': 'sum'
    }).reset_index()
    
    total_revenue = unit_breakdown['revenue'].sum()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_breakdown['percentage'] = np.where(
        total_revenue > 0,
        (unit_breakdown['revenue'] / total_revenue * 100).round(1),
        0
    )
    unit_breakdown = unit_breakdown.sort_values('revenue', ascending=False)
    
    return unit_breakdown


def calculate_operational_metrics(tours_df):
    """
    Calculate operational metrics
    """
    # Average occupancy rate
    confirmed = filter_confirmed_bookings(tours_df)
    total_booked = confirmed['num_customers'].sum()
    total_capacity = confirmed['tour_capacity'].sum()
    # Hàm này đã có bảo vệ chia cho 0
    avg_occupancy = (total_booked / total_capacity * 100) if total_capacity > 0 else 0
    
    # Cancellation/postponement rate
    total_bookings = len(tours_df)
    cancelled_postponed = len(tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])])
    # Hàm này đã có bảo vệ chia cho 0
    cancel_rate = (cancelled_postponed / total_bookings * 100) if total_bookings > 0 else 0
    
    # Returning customer rate
    customer_counts = tours_df.groupby('customer_id').size()
    returning_customers = len(customer_counts[customer_counts >= 2])
    total_unique_customers = len(customer_counts)
    # Hàm này đã có bảo vệ chia cho 0
    returning_rate = (returning_customers / total_unique_customers * 100) if total_unique_customers > 0 else 0
    
    return {
        'avg_occupancy': avg_occupancy,
        'cancel_rate': cancel_rate,
        'returning_rate': returning_rate
    }


def get_low_margin_tours(tours_df, threshold=5):
    """
    Get tours with profit margin below threshold
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty:
        return pd.DataFrame(columns=['route', 'gross_profit', 'revenue', 'num_customers', 'profit_margin'])

    # Group by route and calculate average margin
    route_margins = confirmed.groupby('route').agg({
        'gross_profit': 'sum',
        'revenue': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    route_margins['profit_margin'] = np.where(
        route_margins['revenue'] > 0,
        (route_margins['gross_profit'] / route_margins['revenue'] * 100),
        0
    )
    
    # Filter low margin routes
    low_margin = route_margins[route_margins['profit_margin'] < threshold].sort_values('profit_margin')
    
    return low_margin


def get_unit_performance(tours_df, plans_df, start_date, end_date):
    """
    Calculate performance by business unit
    """
    # Filter current period data
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty: 
        return pd.DataFrame(columns=['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers', 'planned_revenue', 'planned_gross_profit', 'planned_customers', 'revenue_completion', 'customer_completion', 'profit_margin'])

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
    performance = actual_by_unit.merge(plan_by_unit, on='business_unit', how='left').fillna(0)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    performance['revenue_completion'] = np.where(
        performance['planned_revenue'] > 0,
        (performance['actual_revenue'] / performance['planned_revenue'] * 100),
        0
    )
    performance['customer_completion'] = np.where(
        performance['planned_customers'] > 0,
        (performance['actual_customers'] / performance['planned_customers'] * 100),
        0
    )
    performance['profit_margin'] = np.where(
        performance['actual_revenue'] > 0,
        (performance['actual_profit'] / performance['actual_revenue'] * 100),
        0
    )
    
    return performance


def get_unit_breakdown(tours_df, plans_df, start_date, end_date, metric='revenue'):
    """
    Get completion rate breakdown by business unit for a specific metric (Dùng cho Gauge Chart Hover)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    results = []
    for unit in sorted(confirmed_data['business_unit'].unique()):
        unit_data = confirmed_data[confirmed_data['business_unit'] == unit]
        unit_plans = period_plans[period_plans['business_unit'] == unit]
        
        if metric == 'revenue':
            actual = unit_data['revenue'].sum()
            planned = unit_plans['planned_revenue'].sum()
        elif metric == 'profit':
            actual = unit_data['gross_profit'].sum()
            planned = unit_plans['planned_gross_profit'].sum()
        else:  # customers
            actual = unit_data['num_customers'].sum()
            planned = unit_plans['planned_customers'].sum()
        
        completion = calculate_completion_rate(actual, planned)
        results.append({
            'business_unit': unit,
            'completion': completion
        })
    
    return pd.DataFrame(results)


def get_segment_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by segment (FIT/GIT/Inbound) for a specific metric
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty:
        return pd.DataFrame(columns=['segment', 'value', 'percentage'])
    
    if metric == 'revenue':
        segment_data = confirmed_data.groupby('segment')['revenue'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    elif metric == 'customers':
        segment_data = confirmed_data.groupby('segment')['num_customers'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    else:  # profit
        segment_data = confirmed_data.groupby('segment')['gross_profit'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    
    total_value = segment_data['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_data['percentage'] = np.where(
        total_value > 0,
        (segment_data['value'] / total_value * 100).round(1),
        0
    )
    segment_data = segment_data.sort_values('value', ascending=False)
    
    return segment_data


def get_segment_unit_breakdown(tours_df, start_date, end_date, segment_name, metric='revenue'):
    """
    Get business unit breakdown for a specific segment (for hover tooltips)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    segment_data = confirmed_data[confirmed_data['segment'] == segment_name]
    
    if segment_data.empty:
        return pd.DataFrame(columns=['business_unit', 'value', 'percentage'])
    
    if metric == 'revenue':
        unit_breakdown = segment_data.groupby('business_unit')['revenue'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    elif metric == 'customers':
        unit_breakdown = segment_data.groupby('business_unit')['num_customers'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    else:  # profit
        unit_breakdown = segment_data.groupby('business_unit')['gross_profit'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    
    total_value = unit_breakdown['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_breakdown['percentage'] = np.where(
        total_value > 0,
        (unit_breakdown['value'] / total_value * 100).round(1),
        0
    )
    unit_breakdown = unit_breakdown.sort_values('value', ascending=False)
    
    return unit_breakdown


def create_forecast_chart(tours_df, plans_df, start_date, end_date):
    """
    Create forecast chart combining actual (bars) and plan (line) with projection
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    # Group by month for the period
    confirmed_data['month'] = pd.to_datetime(confirmed_data['booking_date']).dt.to_period('M')
    monthly_actual = confirmed_data.groupby('month').agg({
        'revenue': 'sum'
    }).reset_index()
    monthly_actual['month_str'] = monthly_actual['month'].astype(str)
    
    # Get monthly plans
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    monthly_plan = period_plans.groupby('month').agg({
        'planned_revenue': 'sum'
    }).reset_index()
    monthly_plan['month_str'] = monthly_plan['month'].apply(lambda x: f"{start_dt.year}-{x:02d}")
    
    # Create combined chart
    fig = go.Figure()
    
    # Actual bars
    fig.add_trace(go.Bar(
        x=monthly_actual['month_str'],
        y=monthly_actual['revenue'],
        name='Thực hiện',
        marker_color='#636EFA',
        hovertemplate='Thực hiện: %{y:,.0f} ₫<extra></extra>'
    ))
    
    # Plan line
    fig.add_trace(go.Scatter(
        x=monthly_plan['month_str'],
        y=monthly_plan['planned_revenue'],
        name='Kế hoạch',
        mode='lines+markers',
        line=dict(color='#EF553B', width=2, dash='dash'),
        hovertemplate='Kế hoạch: %{y:,.0f} ₫<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=40, b=30),
        hovermode='x unified'
    )
    
    return fig


def create_trend_chart(tours_df, start_date, end_date, metrics=['revenue', 'customers', 'profit']):
    """
    Create a multi-line trend chart showing trends over time
    """
    # Filter confirmed bookings in period
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    # Calculate period length in days
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # Determine grouping granularity
    if period_length <= 7:
        # Daily granularity for week or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('D')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].dt.strftime('%d/%m')
        x_title = "Ngày"
    elif period_length <= 60:
        # Weekly granularity for 2 months or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].apply(lambda x: f"T{x.week}")
        x_title = "Tuần"
    else:
        # Monthly granularity for longer periods
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].astype(str)
        x_title = "Tháng"
    
    monthly_data = period_data
    
    # Create figure
    fig = go.Figure()
    
    if 'revenue' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['revenue'],
            name='Doanh thu',
            mode='lines+markers',
            line=dict(color='#636EFA', width=2),
            yaxis='y1'
        ))
    
    if 'customers' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['num_customers'],
            name='Lượt khách',
            mode='lines+markers',
            line=dict(color='#00CC96', width=2),
            yaxis='y2'
        ))
    
    if 'profit' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['gross_profit'],
            name='Lợi nhuận',
            mode='lines+markers',
            line=dict(color='#AB63FA', width=2),
            yaxis='y1'
        ))
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis=dict(title="Doanh thu / Lợi nhuận (₫)", side='left'),
        yaxis2=dict(title="Lượt khách", overlaying='y', side='right'),
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=40, b=30),
        hovermode='x unified'
    )
    
    return fig


def calculate_marketing_metrics(tours_df, start_date, end_date):
    """
    Calculate marketing and sales cost metrics (OPEX)
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    total_revenue = period_tours['revenue'].sum()
    total_opex = period_tours['opex'].sum()
    total_marketing = period_tours['marketing_cost'].sum()
    total_sales = period_tours['sales_cost'].sum()
    
    opex_ratio = (total_opex / total_revenue * 100) if total_revenue > 0 else 0
    
    return {
        'total_opex': total_opex,
        'total_marketing': total_marketing,
        'total_sales': total_sales,
        'total_revenue': total_revenue,
        'opex_ratio': opex_ratio
    }


def calculate_cac_by_channel(tours_df, start_date, end_date):
    """
    Calculate Customer Acquisition Cost (CAC) by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    channel_metrics = period_tours.groupby('sales_channel').agg({
        'opex': 'sum',
        'customer_id': 'nunique',  # Unique customers
        'revenue': 'sum'
    }).reset_index()
    
    channel_metrics.columns = ['sales_channel', 'total_opex', 'unique_customers', 'revenue']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    channel_metrics['cac'] = np.where(
        channel_metrics['unique_customers'] > 0,
        channel_metrics['total_opex'] / channel_metrics['unique_customers'],
        0
    )
    channel_metrics['cac'] = channel_metrics['cac'].fillna(0)
    
    return channel_metrics


def calculate_clv_by_segment(tours_df):
    """
    Calculate Customer Lifetime Value (CLV) by segment
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    
    # Calculate CLV = Total revenue from repeat customers / Number of customers
    segment_metrics = confirmed_tours.groupby('segment').agg({
        'customer_id': 'nunique',
        'revenue': 'sum',
        'booking_id': 'count'
    }).reset_index()
    
    segment_metrics.columns = ['segment', 'unique_customers', 'total_revenue', 'total_bookings']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['avg_bookings_per_customer'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_bookings'] / segment_metrics['unique_customers'],
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['clv'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_revenue'] / segment_metrics['unique_customers'],
        0
    )
    segment_metrics['clv'] = segment_metrics['clv'].fillna(0)
    
    return segment_metrics


def get_channel_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if metric == 'revenue':
        channel_data = period_tours.groupby('sales_channel').agg({
            'revenue': 'sum',
            'num_customers': 'sum'
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    elif metric == 'customers':
        channel_data = period_tours.groupby('sales_channel').agg({
            'num_customers': 'sum',
            'revenue': 'sum' # Giữ revenue để tính Avg Rev per customer
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    else:  # profit
        channel_data = period_tours.groupby('sales_channel').agg({
            'gross_profit': 'sum'
        }).reset_index()
        return channel_data


def create_profit_margin_chart_with_color(data, x_col, y_col, title):
    """
    Create horizontal bar chart with continuous color scale (temperature/heatmap style)
    """
    # Use continuous color scale based on margin values
    fig = go.Figure(go.Bar(
        x=data[x_col],
        y=data[y_col],
        orientation='h',
        marker=dict(
            color=data[x_col],
            colorscale='RdYlGn',  # Red-Yellow-Green temperature scale
            showscale=True,
            colorbar=dict(
                title=dict(text="Tỷ suất LN (%)", side="right"),
                tickmode="linear",
                tick0=0,
                dtick=2,
                len=0.7
            ),
            cmin=data[x_col].min(),
            cmax=data[x_col].max()
        ),
        text=data[x_col].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='%{y}<br>Tỷ suất: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tỷ suất lợi nhuận (%)",
        yaxis_title="",
        height=max(300, len(data) * 30),
        margin=dict(l=30, r=100, t=50, b=30),
        showlegend=False
    )
    
    return fig


def get_route_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by route with plan comparison
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if period_tours.empty: 
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'planned_revenue', 'revenue_completion'])

    # Actual data
    route_actual = period_tours.groupby('route').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    route_actual['profit_margin'] = np.where(
        route_actual['revenue'] > 0,
        (route_actual['gross_profit'] / route_actual['revenue'] * 100),
        0
    )
    
    # Plan data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    route_plan = period_plans.groupby('route').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # Merge
    route_table = route_actual.merge(route_plan, on='route', how='left').fillna(0)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    route_table['revenue_completion'] = np.where(
        route_table['planned_revenue'] > 0,
        (route_table['revenue'] / route_table['planned_revenue'] * 100),
        0
    )
    route_table['customers_completion'] = np.where(
        route_table['planned_customers'] > 0,
        (route_table['num_customers'] / route_table['planned_customers'] * 100),
        0
    )
    route_table['profit_completion'] = np.where(
        route_table['planned_gross_profit'] > 0,
        (route_table['gross_profit'] / route_table['planned_gross_profit'] * 100),
        0
    )
    
    return route_table


def get_unit_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by business unit
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if period_tours.empty: 
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'avg_revenue_per_customer'])

    # Actual data
    unit_actual = period_tours.groupby('business_unit').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['profit_margin'] = np.where(
        unit_actual['revenue'] > 0,
        (unit_actual['gross_profit'] / unit_actual['revenue'] * 100),
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['avg_revenue_per_customer'] = np.where(
        unit_actual['num_customers'] > 0,
        (unit_actual['revenue'] / unit_actual['num_customers']),
        0
    )
    
    # Plan data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    unit_plan = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # Merge
    unit_table = unit_actual.merge(unit_plan, on='business_unit', how='left').fillna(0)
    
    return unit_table