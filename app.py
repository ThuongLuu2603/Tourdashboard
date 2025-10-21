"""
Vietravel Business Intelligence Dashboard
Comprehensive tour sales performance, revenue, profit margins, and operational metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pytz


# Import custom modules
from data_generator import load_or_generate_data
from data_generator import load_or_generate_data
from utils import (
    format_currency, format_number, format_percentage,
    calculate_kpis, create_gauge_chart, create_bar_chart,
    create_pie_chart, create_line_chart, get_top_routes,
    calculate_operational_metrics, get_low_margin_tours,
    get_unit_performance, filter_data_by_date, filter_confirmed_bookings
)

# Page configuration
st.set_page_config(
    page_title="Vietravel BI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data
# Trong app.py, thay thế từ dòng 32 trở đi (khối if 'data_loaded' not in st.session_state:)

if 'data_loaded' not in st.session_state:
    with st.spinner('Đang tải dữ liệu...'):
        
        # SỬ DỤNG DỮ LIỆU CỨNG ĐỂ LOẠI TRỪ LỖI data_generator.py
        current_year = datetime.now().year
        
        # Dữ liệu tour giả
        tours_df = pd.DataFrame({
            'booking_id': [f"BK{i:06d}" for i in range(1, 5)],
            'customer_id': [f"KH{i:03d}" for i in range(1, 5)],
            'booking_date': [datetime(current_year, 10, 15)] * 4,
            'route': ['DH & ĐBSH', 'Thái Lan', 'Châu Âu', 'Phú Quốc'],
            'business_unit': ['Miền Bắc', 'Trụ sở & ĐNB', 'Trụ sở & ĐNB', 'Miền Tây'],
            'sales_channel': ['Online', 'Trực tiếp VPGD', 'Đại lý', 'Online'],
            'num_customers': [5, 2, 8, 3],
            'tour_capacity': [20, 25, 30, 15],
            'price_per_person': [5000000, 10000000, 50000000, 6000000],
            'revenue': [25000000, 20000000, 400000000, 18000000],
            'cost': [20000000, 16000000, 320000000, 14000000],
            'gross_profit': [5000000, 4000000, 80000000, 4000000],
            'gross_profit_margin': [20.0, 20.0, 20.0, 22.22],
            'status': ['Đã xác nhận', 'Đã xác nhận', 'Đã xác nhận', 'Đã xác nhận']
        })
        
        # Dữ liệu plans giả
        plans_df = pd.DataFrame({
            'year': [current_year] * 4,
            'month': [10] * 4,
            'business_unit': ['Miền Bắc', 'Trụ sở & ĐNB', 'Miền Tây', 'Miền Trung'],
            'route': ['DH & ĐBSH', 'Thái Lan', 'Phú Quốc', 'Nam Trung Bộ'],
            'planned_customers': [100, 50, 70, 60],
            'planned_revenue': [500000000, 200000000, 300000000, 150000000],
            'planned_gross_profit': [100000000, 40000000, 60000000, 30000000]
        })
        
        # Dữ liệu lịch sử giả
        historical_df = tours_df.copy() 
        historical_df['booking_date'] = historical_df['booking_date'] - timedelta(days=365)

        # Lưu vào session state
        st.session_state.tours_df = tours_df
        st.session_state.plans_df = plans_df
        st.session_state.historical_df = historical_df
        st.session_state.data_loaded = True

# Load data from session state (giữ nguyên các dòng tiếp theo)
tours_df = st.session_state.tours_df
plans_df = st.session_state.plans_df
historical_df = st.session_state.historical_df

# Dashboard Title
st.title("📊 VIETRAVEL - DASHBOARD KINH DOANH TOUR")
st.markdown("---")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Bộ lọc dữ liệu")
    
    # Date range selector
    st.subheader("Khoảng thời gian")
    
    # Quick date range options
    date_option = st.selectbox(
        "Chọn kỳ báo cáo",
        ["Tháng này", "Tháng trước", "Quý này", "Năm nay", "Tùy chỉnh"]
    )
    
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(vietnam_tz).replace(tzinfo=None)
    
    if date_option == "Tháng này":
        start_date = datetime(today.year, today.month, 1)
        end_date = today
    elif date_option == "Tháng trước":
        first_day_this_month = datetime(today.year, today.month, 1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        start_date = datetime(last_day_last_month.year, last_day_last_month.month, 1)
        end_date = last_day_last_month
    elif date_option == "Quý này":
        quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * quarter - 2, 1)
        end_date = today
    elif date_option == "Năm nay":
        start_date = datetime(today.year, 1, 1)
        end_date = today
    else:  # Tùy chỉnh
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày",
                value=datetime(today.year, today.month, 1)
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày",
                value=today
            )
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    
    st.markdown(f"**Kỳ báo cáo:** {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    
    # Business unit filter
    st.subheader("Đơn vị kinh doanh")
    business_units = ["Tất cả"] + sorted(tours_df['business_unit'].unique().tolist())
    selected_unit = st.selectbox("Chọn đơn vị", business_units)
    
    # Route filter
    st.subheader("Tuyến tour")
    if selected_unit != "Tất cả":
        routes = ["Tất cả"] + sorted(
            tours_df[tours_df['business_unit'] == selected_unit]['route'].unique().tolist()
        )
    else:
        routes = ["Tất cả"] + sorted(tours_df['route'].unique().tolist())
    selected_route = st.selectbox("Chọn tuyến", routes)
    
    # Top N selector
    st.subheader("Thiết lập hiển thị")
    top_n = st.slider("Top N tuyến tour", min_value=5, max_value=15, value=10)
    
    st.markdown("---")
    
    # Refresh data button
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.session_state.data_loaded = False
        st.rerun()

# Filter data based on selections
filtered_tours = filter_data_by_date(tours_df, start_date, end_date)

if selected_unit != "Tất cả":
    filtered_tours = filtered_tours[filtered_tours['business_unit'] == selected_unit]

if selected_route != "Tất cả":
    filtered_tours = filtered_tours[filtered_tours['route'] == selected_route]

# Calculate KPIs
kpis = calculate_kpis(tours_df, plans_df, start_date, end_date)

# ============================================================
# SECTION I: OVERALL PERFORMANCE INDICATORS
# ============================================================
st.header("I. 📈 CHỈ SỐ HIỆU SUẤT TỔNG QUAN")

# KPI Cards - Row 1
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💰 DOANH THU THỰC HIỆN",
        value=format_currency(kpis['actual_revenue']),
        delta=f"{format_percentage(kpis['revenue_growth'])} so với cùng kỳ"
    )
    with st.expander("Chi tiết"):
        st.write(f"**Kế hoạch:** {format_currency(kpis['planned_revenue'])}")
        st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_revenue'])}")

with col2:
    st.metric(
        label="💵 LỢI NHUẬN GỘP",
        value=format_currency(kpis['actual_gross_profit']),
        delta=f"{format_percentage(kpis['profit_growth'])} so với cùng kỳ"
    )
    with st.expander("Chi tiết"):
        st.write(f"**Kế hoạch:** {format_currency(kpis['planned_gross_profit'])}")
        st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_gross_profit'])}")

with col3:
    st.metric(
        label="👥 TỔNG LƯỢT KHÁCH",
        value=format_number(kpis['actual_customers']),
        delta=f"{format_percentage(kpis['customer_growth'])} so với cùng kỳ"
    )
    with st.expander("Chi tiết"):
        st.write(f"**Kế hoạch:** {format_number(kpis['planned_customers'])}")
        st.write(f"**Cùng kỳ năm trước:** {format_number(kpis['ly_customers'])}")

st.markdown("---")

# Completion Rate Gauges
st.subheader("Tỷ lệ hoàn thành kế hoạch")

col1, col2 = st.columns(2)

with col1:
    fig_revenue = create_gauge_chart(
        kpis['revenue_completion'],
        "Tỷ lệ Hoàn thành Doanh thu (%)"
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    fig_customers = create_gauge_chart(
        kpis['customer_completion'],
        "Tỷ lệ Hoàn thành Lượt khách (%)"
    )
    st.plotly_chart(fig_customers, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION II: DETAILED ANALYSIS & TOUR ROUTES
# ============================================================
st.header("II. 🔍 CHỈ SỐ PHÂN TÍCH CHI TIẾT & TUYẾN TOUR")

# Top Routes Analysis
st.subheader(f"Top {top_n} Tuyến Tour")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Theo Doanh thu", 
    "👥 Theo Lượt khách", 
    "💹 Theo Lợi nhuận", 
    "📈 Tốc độ đạt KH"
])

with tab1:
    st.markdown("#### Doanh thu thực hiện theo tuyến Tour")
    top_revenue = get_top_routes(filtered_tours, n=top_n, metric='revenue')
    
    if not top_revenue.empty:
        # Bar chart
        fig = create_bar_chart(
            top_revenue,
            x='route',
            y='revenue',
            title=f"Top {top_n} Tuyến Tour - Doanh thu cao nhất",
            orientation='v'
        )
        fig.update_traces(
            text=[format_currency(v) for v in top_revenue['revenue']],
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        display_df = top_revenue.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x}%")
        
        display_df.columns = ['Tuyến Tour', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 'Tỷ suất LN (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Không có dữ liệu để hiển thị")

with tab2:
    st.markdown("#### Lượt khách theo tuyến Tour")
    top_customers = get_top_routes(filtered_tours, n=top_n, metric='customers')
    
    if not top_customers.empty:
        # Bar chart
        fig = create_bar_chart(
            top_customers,
            x='route',
            y='num_customers',
            title=f"Top {top_n} Tuyến Tour - Lượt khách nhiều nhất",
            orientation='v'
        )
        fig.update_traces(
            text=[format_number(v) for v in top_customers['num_customers']],
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        display_df = top_customers.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x}%")
        
        display_df.columns = ['Tuyến Tour', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 'Tỷ suất LN (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Không có dữ liệu để hiển thị")

with tab3:
    st.markdown("#### Lợi nhuận gộp (%) theo tuyến Tour")
    top_profit = get_top_routes(filtered_tours, n=top_n, metric='gross_profit')
    
    if not top_profit.empty:
        # Horizontal bar chart for profit margin
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_profit['route'],
            x=top_profit['profit_margin'],
            orientation='h',
            text=[f"{v:.1f}%" for v in top_profit['profit_margin']],
            textposition='outside',
            marker=dict(
                color=top_profit['profit_margin'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Tỷ suất LN (%)")
            )
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Tuyến Tour - Tỷ suất Lợi nhuận cao nhất",
            xaxis_title="Tỷ suất Lợi nhuận (%)",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        display_df = top_profit.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x}%")
        
        display_df.columns = ['Tuyến Tour', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 'Tỷ suất LN (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Không có dữ liệu để hiển thị")

with tab4:
    st.markdown("#### Tốc độ đạt kế hoạch (%) theo tuyến Tour")
    
    # Calculate completion rate by route
    confirmed_filtered = filter_confirmed_bookings(filtered_tours)
    actual_by_route = confirmed_filtered.groupby('route').agg({
        'revenue': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    
    # Get plans for the period
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    plan_by_route = period_plans.groupby('route').agg({
        'planned_revenue': 'sum'
    }).reset_index()
    
    # Merge and calculate
    completion_by_route = actual_by_route.merge(plan_by_route, on='route', how='left')
    completion_by_route['completion_rate'] = (
        completion_by_route['revenue'] / completion_by_route['planned_revenue'] * 100
    )
    completion_by_route = completion_by_route.sort_values('completion_rate', ascending=False).head(top_n)
    
    if not completion_by_route.empty:
        # Bar chart
        fig = go.Figure()
        
        colors = ['#00CC96' if x >= 100 else '#FFA500' if x >= 80 else '#EF553B' 
                  for x in completion_by_route['completion_rate']]
        
        fig.add_trace(go.Bar(
            x=completion_by_route['route'],
            y=completion_by_route['completion_rate'],
            text=[f"{v:.1f}%" for v in completion_by_route['completion_rate']],
            textposition='outside',
            marker_color=colors
        ))
        
        # Add 100% reference line
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                      annotation_text="Mục tiêu 100%")
        
        fig.update_layout(
            title="Tốc độ đạt kế hoạch Doanh thu theo Tuyến Tour",
            xaxis_title="",
            yaxis_title="Tỷ lệ hoàn thành (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        display_df = completion_by_route.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['planned_revenue'] = display_df['planned_revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['completion_rate'] = display_df['completion_rate'].apply(lambda x: f"{x:.1f}%")
        
        display_df.columns = ['Tuyến Tour', 'Doanh thu TH', 'Lượt khách', 'Doanh thu KH', 'Tỷ lệ đạt (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Không có dữ liệu để hiển thị")

st.markdown("---")

# Sales Channel Analysis
st.subheader("Lượt khách theo Kênh bán")

confirmed_filtered = filter_confirmed_bookings(filtered_tours)
channel_data = confirmed_filtered.groupby('sales_channel').agg({
    'num_customers': 'sum',
    'revenue': 'sum'
}).reset_index()

if not channel_data.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for customers by channel
        fig = create_pie_chart(
            channel_data,
            values='num_customers',
            names='sales_channel',
            title="Phân bổ Lượt khách theo Kênh bán"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart for revenue by channel
        fig = create_pie_chart(
            channel_data,
            values='revenue',
            names='sales_channel',
            title="Phân bổ Doanh thu theo Kênh bán"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    display_df = channel_data.copy()
    display_df['num_customers'] = display_df['num_customers'].apply(format_number)
    display_df['revenue'] = display_df['revenue'].apply(format_currency)
    display_df['avg_revenue_per_customer'] = (channel_data['revenue'] / channel_data['num_customers']).apply(format_currency)
    
    display_df.columns = ['Kênh bán', 'Lượt khách', 'Doanh thu', 'Doanh thu TB/khách']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Không có dữ liệu để hiển thị")

st.markdown("---")

# Business Unit Performance
st.subheader("Hiệu suất theo Đơn vị Kinh doanh")

unit_performance = get_unit_performance(tours_df, plans_df, start_date, end_date)

if not unit_performance.empty:
    # Bar chart comparing actual vs planned
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Doanh thu Thực hiện',
        x=unit_performance['business_unit'],
        y=unit_performance['actual_revenue'],
        text=[format_currency(v) for v in unit_performance['actual_revenue']],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Doanh thu Kế hoạch',
        x=unit_performance['business_unit'],
        y=unit_performance['planned_revenue'],
        text=[format_currency(v) for v in unit_performance['planned_revenue']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="So sánh Doanh thu Thực hiện vs Kế hoạch theo Đơn vị",
        xaxis_title="",
        yaxis_title="Doanh thu",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    display_df = unit_performance.copy()
    display_df['actual_revenue'] = display_df['actual_revenue'].apply(format_currency)
    display_df['planned_revenue'] = display_df['planned_revenue'].apply(format_currency)
    display_df['actual_customers'] = display_df['actual_customers'].apply(format_number)
    display_df['revenue_completion'] = display_df['revenue_completion'].apply(lambda x: f"{x:.1f}%")
    display_df['customer_completion'] = display_df['customer_completion'].apply(lambda x: f"{x:.1f}%")
    display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
    
    display_df = display_df[[
        'business_unit', 'actual_revenue', 'planned_revenue', 'revenue_completion',
        'actual_customers', 'customer_completion', 'profit_margin'
    ]]
    
    display_df.columns = [
        'Đơn vị', 'Doanh thu TH', 'Doanh thu KH', 'Tỷ lệ DT (%)',
        'Lượt khách', 'Tỷ lệ Khách (%)', 'Tỷ suất LN (%)'
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Không có dữ liệu để hiển thị")

st.markdown("---")

# ============================================================
# SECTION III: OPERATIONAL MANAGEMENT & ALERTS
# ============================================================
st.header("III. ⚙️ CHỈ SỐ QUẢN LÝ HOẠT ĐỘNG & CẢNH BÁO")

# Operational Metrics
operational_metrics = calculate_operational_metrics(filtered_tours)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📊 TỶ LỆ LẤP ĐẦY BÌNH QUÂN",
        value=format_percentage(operational_metrics['avg_occupancy'])
    )
    
    # Occupancy gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = operational_metrics['avg_occupancy'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00CC96"},
            'steps': [
                {'range': [0, 50], 'color': "#FFE5E5"},
                {'range': [50, 70], 'color': "#FFF4E5"},
                {'range': [70, 90], 'color': "#E5F5E5"},
                {'range': [90, 100], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric(
        label="❌ TỶ LỆ KHÁCH HỦY/HOÃN",
        value=format_percentage(operational_metrics['cancel_rate'])
    )
    
    # Cancellation gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = operational_metrics['cancel_rate'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [None, 30]},
            'bar': {'color': "#EF553B"},
            'steps': [
                {'range': [0, 5], 'color': "#D4F1D4"},
                {'range': [5, 10], 'color': "#E5F5E5"},
                {'range': [10, 20], 'color': "#FFF4E5"},
                {'range': [20, 30], 'color': "#FFE5E5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 15
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.metric(
        label="🔄 TỶ LỆ KHÁCH HÀNG QUAY LẠI",
        value=format_percentage(operational_metrics['returning_rate'])
    )
    
    # Returning customer gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = operational_metrics['returning_rate'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [None, 50]},
            'bar': {'color': "#AB63FA"},
            'steps': [
                {'range': [0, 10], 'color': "#FFE5E5"},
                {'range': [10, 20], 'color': "#FFF4E5"},
                {'range': [20, 30], 'color': "#E5F5E5"},
                {'range': [30, 50], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Low Margin Tours Alert
st.subheader("🚨 Cảnh báo Lợi nhuận Gộp Dưới Ngưỡng An toàn")

low_margin_tours = get_low_margin_tours(filtered_tours, threshold=5)

if not low_margin_tours.empty:
    st.warning(f"⚠️ Phát hiện {len(low_margin_tours)} tuyến tour có tỷ suất lợi nhuận < 5%")
    
    # Alert table
    display_df = low_margin_tours.copy()
    display_df['revenue'] = display_df['revenue'].apply(format_currency)
    display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
    display_df['num_customers'] = display_df['num_customers'].apply(format_number)
    display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.2f}%")
    
    display_df.columns = ['Tuyến Tour', 'Lợi nhuận gộp', 'Doanh thu', 'Lượt khách', 'Tỷ suất LN (%)']
    
    # Highlight low margins in red
    def highlight_low_margin(row):
        return ['background-color: #FFE5E5' if 'Tỷ suất LN' in col else '' for col in row.index]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Bar chart of low margin tours
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=low_margin_tours['route'],
        y=low_margin_tours['profit_margin'],
        text=[f"{v:.2f}%" for v in low_margin_tours['profit_margin']],
        textposition='outside',
        marker_color='#EF553B'
    ))
    
    fig.add_hline(y=5, line_dash="dash", line_color="red", 
                  annotation_text="Ngưỡng an toàn 5%")
    
    fig.update_layout(
        title="Tuyến Tour có Tỷ suất Lợi nhuận Thấp",
        xaxis_title="",
        yaxis_title="Tỷ suất Lợi nhuận (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ Tất cả tuyến tour đều có tỷ suất lợi nhuận trên ngưỡng an toàn")

st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)
