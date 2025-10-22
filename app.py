"""
Vietravel Business Intelligence Dashboard
Comprehensive tour sales performance, revenue, profit margins, and operational metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz # Cần thiết cho Timezone handling
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from data_generator import load_or_generate_data
from utils import (
    # Các hàm Format và Core Logic
    format_currency, format_number, format_percentage,
    calculate_completion_rate, get_growth_rate, filter_data_by_date, filter_confirmed_bookings,
    
    # Các hàm KPI và Chart
    calculate_kpis, 
    create_gauge_chart, create_bar_chart, create_pie_chart, create_line_chart,
    
    # Các hàm Top/Breakdown
    get_top_routes, get_route_unit_breakdown, get_unit_breakdown,
    get_segment_breakdown, get_segment_unit_breakdown, get_channel_breakdown,
    
    # Các hàm Operational và Detailed Tables
    calculate_operational_metrics, get_low_margin_tours, get_unit_performance, 
    get_route_detailed_table, get_unit_detailed_table,
    
    # Các hàm Marketing/CLV/Forecast
    create_forecast_chart, create_trend_chart, 
    calculate_marketing_metrics, calculate_cac_by_channel, calculate_clv_by_segment, 
    create_profit_margin_chart_with_color
)

# Page configuration
st.set_page_config(
    page_title="Vietravel BI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce padding and whitespace
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    with st.spinner('Đang tải dữ liệu...'):
        # KHÔI PHỤC LẠI LỆNH GỌI DATA GENERATOR
        tours_df, plans_df, historical_df = load_or_generate_data()
        st.session_state.tours_df = tours_df
        st.session_state.plans_df = plans_df
        st.session_state.historical_df = historical_df
        st.session_state.data_loaded = True

# Load data from session state
tours_df = st.session_state.tours_df
plans_df = st.session_state.plans_df
historical_df = st.session_state.historical_df

# Dashboard Title
st.title("📊 VIETRAVEL - DASHBOARD KINH DOANH TOUR")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Bộ lọc dữ liệu")
    
    # Date range selector
    st.subheader("Khoảng thời gian")
    
    # Quick date range options
    date_option = st.selectbox(
        "Chọn kỳ báo cáo",
        ["Tuần", "Tháng", "Quý", "Năm", "Tùy chỉnh"]
    )
    
    # Xử lý Timezone an toàn
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(vietnam_tz).replace(tzinfo=None) # Naive datetime
    
    if date_option == "Tuần":
        # 7 ngày gần nhất
        start_date = today - timedelta(days=6)
        start_date = datetime(start_date.year, start_date.month, start_date.day)
        end_date = today
    elif date_option == "Tháng":
        # Tháng hiện tại
        start_date = datetime(today.year, today.month, 1)
        end_date = today
    elif date_option == "Quý":
        # Quý hiện tại
        quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * quarter - 2, 1)
        end_date = today
    elif date_option == "Năm":
        # Năm hiện tại
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
    
    # Segment filter
    st.subheader("Phân khúc")
    segments = ["Tất cả"] + sorted(tours_df['segment'].unique().tolist())
    selected_segment = st.selectbox("Chọn phân khúc", segments)
    
    # Top N selector
    st.subheader("Thiết lập hiển thị")
    top_n = st.slider("Top N tuyến tour", min_value=5, max_value=15, value=10)
    
    st.markdown("---")
    
    # Refresh data button
    if st.button("🔄 Làm mới dữ liệu", width='stretch'):
        st.session_state.data_loaded = False
        st.rerun()

# Filter data based on selections (dimensional filters only, NOT date)
# Date filtering will be done inside calculate_kpis to preserve YoY data
tours_filtered_dimensional = tours_df.copy()
filtered_plans = plans_df.copy()

if selected_unit != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['business_unit'] == selected_unit]
    filtered_plans = filtered_plans[filtered_plans['business_unit'] == selected_unit]

if selected_route != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['route'] == selected_route]
    filtered_plans = filtered_plans[filtered_plans['route'] == selected_route]

if selected_segment != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['segment'] == selected_segment]
    filtered_plans = filtered_plans[filtered_plans['segment'] == selected_segment]

# Calculate KPIs using dimensionally filtered data (calculate_kpis will handle date filtering)
kpis = calculate_kpis(tours_filtered_dimensional, filtered_plans, start_date, end_date)

# Also create a date+dimension filtered version for charts that don't need historical data
filtered_tours = filter_data_by_date(tours_filtered_dimensional, start_date, end_date)

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2 = st.tabs([
    "📊 Tổng quan",
    "🔍 Chi tiết"
])

# ============================================================
# TAB 1: TỔNG QUAN (5 VÙNG THEO SPEC)
# ============================================================
with tab1:
    # ========== VÙNG 1: CHỈ SỐ TỔNG QUAN ==========
    st.markdown("###  Vùng 1: Chỉ số Tổng quan")
    
    # Row 1: 3 KPI Cards 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 DOANH THU TỔNG",
            value=format_currency(kpis['actual_revenue']),
            delta=f"{format_percentage(kpis['revenue_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_revenue'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['revenue_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_revenue'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['revenue_growth'])}")
    
    with col2:
        st.metric(
            label="💵 LỢI NHUẬN GỘP",
            value=format_currency(kpis['actual_gross_profit']),
            delta=f"{format_percentage(kpis['profit_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_gross_profit'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_gross_profit'])}")
            profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
            st.write(f"**Hoàn thành:** {format_percentage(profit_completion)}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_gross_profit'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['profit_growth'])}")
    
    with col3:
        st.metric(
            label="👥 LƯỢT KHÁCH TỔNG",
            value=format_number(kpis['actual_customers']),
            delta=f"{format_percentage(kpis['customer_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_number(kpis['planned_customers'])}")
            st.write(f"**Thực hiện:** {format_number(kpis['actual_customers'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['customer_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_number(kpis['ly_customers'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['customer_growth'])}")
    
    # Row 2: Marketing/Sales Cost and Trend Chart
    st.markdown("")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Calculate marketing metrics
        marketing_metrics = calculate_marketing_metrics(filtered_tours, start_date, end_date)
        st.metric(
            label="💳 CHI PHÍ MARKETING/BÁN HÀNG",
            value=f"{format_percentage(marketing_metrics['opex_ratio'])}",
            delta=f"{format_currency(marketing_metrics['total_opex'])} OPEX"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Chi phí Marketing:** {format_currency(marketing_metrics['total_marketing'])}")
            st.write(f"**Chi phí Bán hàng:** {format_currency(marketing_metrics['total_sales'])}")
            st.write(f"**Tổng OPEX:** {format_currency(marketing_metrics['total_opex'])}")
            st.write(f"**Doanh thu:** {format_currency(marketing_metrics['total_revenue'])}")
            st.write(f"**Tỷ lệ OPEX/DT:** {format_percentage(marketing_metrics['opex_ratio'])}")
    
    with col2:
        st.markdown("<div style='font-size: 14px; font-weight: bold; margin-bottom: 10px;'>📊 Xu hướng Doanh thu / Lượt khách / Lợi nhuận theo thời gian</div>", unsafe_allow_html=True)
        fig_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue', 'customers', 'profit'])
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ========== VÙNG 2: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### Vùng 2: Tốc độ đạt Kế hoạch")
    
    # Row: 3 Gauge charts + 1 Forecast chart
    col1, col2, col3, col4 = st.columns(4)
    
    # Get unit breakdown data for hover tooltips
    revenue_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='revenue')
    profit_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='profit')
    customers_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='customers')
    
    with col1:
        fig_revenue = create_gauge_chart(
            kpis['revenue_completion'],
            "Đạt KH Doanh thu",
            unit_breakdown=revenue_breakdown
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
        fig_profit = create_gauge_chart(
            profit_completion,
            "Đạt KH Lợi nhuận",
            unit_breakdown=profit_breakdown
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with col3:
        fig_customers = create_gauge_chart(
            kpis['customer_completion'],
            "Đạt KH Lượt khách",
            unit_breakdown=customers_breakdown
        )
        st.plotly_chart(fig_customers, use_container_width=True)
    
    with col4:
        st.markdown("<div style='font-size: 11px; font-weight: bold; text-align: center; margin-bottom: 5px;'>Dự báo hoàn thành KH</div>", unsafe_allow_html=True)
        fig_forecast = create_forecast_chart(filtered_tours, filtered_plans, start_date, end_date)
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    
    # ========== VÙNG 3: PHÂN THEO PHÂN KHÚC ==========
    st.markdown("### Vùng 3: Phân theo Phân khúc (FIT / GIT / Inbound)")
    SEGMENT_COLORS = ['#3CB371', '#6495ED', '#FFA07A']
    
    # Row: 3 Pie charts for segments
    col1, col2, col3 = st.columns(3)
    
    # Get segment breakdown data
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_profit = get_segment_breakdown(filtered_tours, start_date, end_date, metric='profit')
    
    with col1:
        st.markdown("#### 💰 Doanh thu theo phân khúc")
        if not segment_revenue.empty:
            # Prepare hover with unit breakdown
            hovertext = []
            for seg in segment_revenue['segment']:
                unit_breakdown = get_segment_unit_breakdown(filtered_tours, start_date, end_date, seg, 'revenue')
                if not unit_breakdown.empty:
                    # Chuyển đổi sang format tiền tệ cho hover
                    breakdown_text = "<br>".join([
                        f"{row['business_unit']}: {format_currency(row['value'])} ({row['percentage']:.1f}%)"
                        for _, row in unit_breakdown.iterrows()
                    ])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=segment_revenue['segment'],
                values=segment_revenue['value'],
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>' +
                             'Doanh thu: %{value:,.0f} ₫<br>' +
                             'Tỉ lệ: %{percent}<br><br>' +
                             '<b>Theo đơn vị:</b><br>' +
                             '%{customdata}' + 
                             '<extra></extra>',
                marker=dict(colors=SEGMENT_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### 👥 Lượt khách theo phân khúc")
        if not segment_customers.empty:
            # Prepare hover with unit breakdown
            hovertext = []
            for seg in segment_customers['segment']:
                unit_breakdown = get_segment_unit_breakdown(filtered_tours, start_date, end_date, seg, 'customers')
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([
                        f"{row['business_unit']}: {format_number(row['value'])} ({row['percentage']:.1f}%)"
                        for _, row in unit_breakdown.iterrows()
                    ])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=segment_customers['segment'],
                values=segment_customers['value'],
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>' +
                             'Lượt khách: %{value:,.0f}<br>' +
                             'Tỉ lệ: %{percent}<br><br>' +
                             '<b>Theo đơn vị:</b><br>' +
                             '%{customdata}' +
                             '<extra></extra>',
                marker=dict(colors=SEGMENT_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    with col3:
        st.markdown("#### 💵 Lợi nhuận theo phân khúc")
        if not segment_profit.empty:
            # Prepare hover with unit breakdown
            hovertext = []
            for seg in segment_profit['segment']:
                unit_breakdown = get_segment_unit_breakdown(filtered_tours, start_date, end_date, seg, 'profit')
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([
                        f"{row['business_unit']}: {format_currency(row['value'])} ({row['percentage']:.1f}%)"
                        for _, row in unit_breakdown.iterrows()
                    ])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=segment_profit['segment'],
                values=segment_profit['value'],
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>' +
                             'Lợi nhuận: %{value:,.0f} ₫<br>' +
                             'Tỉ lệ: %{percent}<br><br>' +
                             '<b>Theo đơn vị:</b><br>' +
                             '%{customdata}' +
                             '<extra></extra>',
                marker=dict(colors=SEGMENT_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    st.markdown("---")
    
    # ========== VÙNG 4: CÁC BẢNG THÔNG TIN KHÁC ==========
    st.markdown("### Vùng 4: Các bảng thông tin khác")
    
    # Prepare data
    unit_performance = get_unit_performance(tours_filtered_dimensional, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    
    # Row: 3 charts (Tiến độ KH, Top 10, Tỉ trọng)
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.markdown("#### 📊 Tiến độ KH theo khu vực")
        if not unit_performance.empty:
            fig = go.Figure()
            colors = ['#00CC96' if x >= 100 else '#FFA500' if x >= 80 else '#EF553B' 
                      for x in unit_performance['revenue_completion']]
            customdata = [[row['actual_revenue'], row['planned_revenue'], row['revenue_completion']]
                          for _, row in unit_performance.iterrows()]
            fig.add_trace(go.Bar(
                x=unit_performance['business_unit'],
                y=unit_performance['revenue_completion'],
                text=[f"{v:.1f}%" for v in unit_performance['revenue_completion']],
                textposition='outside',
                marker_color=colors,
                customdata=customdata,
                hovertemplate='<b>%{x}</b><br>DT thực hiện: %{customdata[0]:,.0f} ₫<br>DT kế hoạch: %{customdata[1]:,.0f} ₫<br>Tiến độ: %{customdata[2]:.1f}%<extra></extra>'
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="KH 100%")
            fig.update_layout(xaxis_title="", yaxis_title="", height=230, showlegend=False, margin=dict(l=30, r=30, t=10, b=30))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### 🎯 Top 10 Tuyến Tour")
        if not top_revenue.empty:
            fig = go.Figure()
            hovertext = []
            for route in top_revenue['route'][::-1]:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route)
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_currency(row['revenue'])} ({row['percentage']:.1f}%)"
                                                  for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            fig.add_trace(go.Bar(
                y=top_revenue['route'][::-1],
                x=top_revenue['revenue'][::-1],
                orientation='h',
                text=[format_currency(v) for v in top_revenue['revenue'][::-1]],
                textposition='outside',
                marker_color='#636EFA',
                customdata=hovertext,
                hovertemplate='<b>%{y}</b><br>Tổng DT: %{x:,.0f} ₫<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(xaxis_title="", yaxis_title="", height=230, showlegend=False, margin=dict(l=100, r=30, t=10, b=30))
            st.plotly_chart(fig)
    
    with col3:
        st.markdown("#### 📊 Tỉ trọng các tuyến (%)")
        if not top_revenue.empty:
            labels = [route if len(route) <= 12 else route[:10] + ".." for route in top_revenue['route']]
            # Add unit breakdown to hover
            hovertext = []
            for route in top_revenue['route']:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route)
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_currency(row['revenue'])} ({row['percentage']:.1f}%)"
                                                  for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=labels,
                values=top_revenue['revenue'],
                textposition='outside',
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>Doanh thu: %{value:,.0f} ₫<br>Tỉ lệ: %{percent}<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    st.markdown("---")
    
    # ========== VÙNG 5: CHỈ SỐ QUẢN LÝ HOẠT ĐỘNG ==========
    st.markdown("### Vùng 5: Chỉ số Quản lý Hoạt động")
    
    # Calculate operational metrics (use all-time dimensional data for accurate rates)
    ops_metrics = calculate_operational_metrics(tours_filtered_dimensional)
    
    # Row: 3 Operational gauge charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ",
            max_value=100,
            threshold=75
        )
        st.plotly_chart(fig_occ)
    
    with col2:
        fig_cancel = create_gauge_chart(
            ops_metrics['cancel_rate'],
            "Tỷ lệ Khách Hủy/Hoãn",
            max_value=30,
            threshold=10
        )
        st.plotly_chart(fig_cancel)
    
    with col3:
        fig_return = create_gauge_chart(
            ops_metrics['returning_rate'],
            "Tỷ lệ Khách Quay lại",
            max_value=100,
            threshold=30
        )
        st.plotly_chart(fig_return)


# ============================================================
# TAB 2: CHI TIẾT (3 VÙNG THEO SPEC)
# ============================================================
with tab2:
    # ========== VÙNG 1: THEO TUYẾN ==========
    st.markdown("### Vùng 1: Phân tích theo Tuyến")
    
    # Get route data
    route_table = get_route_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')
    
    # Row 1: Top tuyến Tour charts
    st.markdown("#### Top Tuyến Tour")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Theo Doanh thu")
        if not top_revenue.empty:
            fig = create_bar_chart(top_revenue.head(5), 'route', 'revenue', '', orientation='v')
            fig.update_traces(text=[format_currency(v) for v in top_revenue.head(5)['revenue']], textposition='outside')
            fig.update_layout(height=200, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Theo Lượt khách")
        if not top_customers.empty:
            fig = create_bar_chart(top_customers.head(5), 'route', 'num_customers', '', orientation='v')
            fig.update_traces(text=[format_number(v) for v in top_customers.head(5)['num_customers']], textposition='outside')
            fig.update_layout(height=200, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("##### Theo Lợi nhuận")
        if not top_profit.empty:
            fig = create_bar_chart(top_profit.head(5), 'route', 'gross_profit', '', orientation='v')
            fig.update_traces(text=[format_currency(v) for v in top_profit.head(5)['gross_profit']], textposition='outside')
            fig.update_layout(height=200, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("")
    
    # Row 2: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Tuyến")
    if not route_table.empty:
        display_df = route_table.copy()
        display_df = display_df[[
            'route', 'revenue', 'num_customers', 'gross_profit', 
            'profit_margin', 'revenue_completion'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['revenue_completion'] = display_df['revenue_completion'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['Tuyến', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 'Tỷ suất LN (%)', 'Tiến độ KH (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Row 3: Profit margin with color coding
    st.markdown("#### Tỷ suất Lợi nhuận theo Tuyến")
    if not route_table.empty:
        top_10_margin = route_table.nlargest(10, 'profit_margin')[['route', 'profit_margin']]
        fig = create_profit_margin_chart_with_color(top_10_margin, 'profit_margin', 'route', '')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== VÙNG 2: THEO KÊNH BÁN VÀ PHÂN KHÚC ==========
    st.markdown("### Vùng 2: Theo Kênh bán và Phân khúc")
    
    # Get channel and segment data
    channel_revenue = get_channel_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    channel_customers = get_channel_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    cac_data = calculate_cac_by_channel(filtered_tours, start_date, end_date)
    clv_data = calculate_clv_by_segment(tours_filtered_dimensional)
    
    # Row 1: Kênh bán pie charts
    st.markdown("#### Phân bố theo Kênh bán")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not channel_revenue.empty:
            fig = create_pie_chart(channel_revenue, 'revenue', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not channel_customers.empty:
            fig = create_pie_chart(channel_customers, 'num_customers', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col3:
        st.markdown("##### Doanh thu TB/khách")
        if not channel_revenue.empty:
            fig = go.Figure(go.Bar(
                x=channel_revenue['sales_channel'],
                y=channel_revenue['avg_revenue_per_customer'],
                text=[format_currency(v) for v in channel_revenue['avg_revenue_per_customer']],
                textposition='outside',
                marker_color='#636EFA'
            ))
            fig.update_layout(xaxis_title="Doanh thu TB/khách (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig)
    
    # Row 2: Kênh bán detailed table
    if not channel_revenue.empty:
        display_df = channel_revenue.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Kênh bán', 'Doanh thu', 'Lượt khách', 'Doanh thu TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Row 3: Phân khúc pie charts
    st.markdown("#### Phân bố theo Phân khúc")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not segment_revenue.empty:
            fig = create_pie_chart(segment_revenue, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not segment_customers.empty:
            fig = create_pie_chart(segment_customers, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    st.markdown("")
    
    # Row 4: CAC and CLV
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Chi phí Thu hút Khách hàng (CAC) theo Kênh")
        if not cac_data.empty:
            fig = go.Figure(go.Bar(
                y=cac_data['sales_channel'],
                x=cac_data['cac'],
                orientation='h',
                text=[format_currency(v) for v in cac_data['cac']],
                textposition='outside',
                marker_color='#FFA15A'
            ))
            fig.update_layout(xaxis_title="CAC (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Giá trị Trọn đời Khách hàng (CLV) theo Phân khúc")
        if not clv_data.empty:
            fig = go.Figure(go.Bar(
                y=clv_data['segment'],
                x=clv_data['clv'],
                orientation='h',
                text=[format_currency(v) for v in clv_data['clv']],
                textposition='outside',
                marker_color='#00CC96'
            ))
            fig.update_layout(xaxis_title="CLV (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    st.markdown("---")
    
    # ========== VÙNG 3: THEO ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 3: Hiệu suất theo Đơn vị Kinh doanh")
    
    # Get unit data
    unit_table = get_unit_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    
    # Row 1: Revenue vs Plan comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### So sánh Doanh thu Thực hiện và Kế hoạch")
        if not unit_table.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['planned_revenue'],
                name='Kế hoạch',
                marker_color='#FFA15A'
            ))
            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['revenue'],
                name='Thực hiện',
                marker_color='#636EFA'
            ))
            fig.update_layout(xaxis_title="", yaxis_title="Doanh thu (₫)", height=300, barmode='group', margin=dict(l=30, r=30, t=10, b=80))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Tỷ suất Lợi nhuận Gộp theo Đơn vị")
        if not unit_table.empty:
            unit_margin = unit_table[['business_unit', 'profit_margin']].copy()
            fig = create_profit_margin_chart_with_color(unit_margin, 'profit_margin', 'business_unit', '')
            st.plotly_chart(fig)
    
    # Row 2: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Đơn vị")
    if not unit_table.empty:
        display_df = unit_table.copy()
        display_df = display_df[[
            'business_unit', 'revenue', 'num_customers', 'gross_profit',
            'profit_margin', 'avg_revenue_per_customer'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Đơn vị', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 'Tỷ suất LN (%)', 'DT TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)


st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)