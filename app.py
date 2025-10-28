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
# Cần import make_subplots ở đây để dùng trong app.py nếu cần cho chart phức tạp
from plotly.subplots import make_subplots 
from admin_ui import render_admin_ui

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
    get_unit_breakdown_simple,
    
    # Các hàm Operational và Detailed Tables
    calculate_operational_metrics, get_low_margin_tours, get_unit_performance, 
    get_route_detailed_table, get_unit_detailed_table,
    
    # Các hàm Marketing/CLV/Forecast
    create_forecast_chart, create_trend_chart, 
    calculate_marketing_metrics, calculate_cac_by_channel, calculate_clv_by_segment, 
    create_profit_margin_chart_with_color,
    calculate_partner_performance,
    
    # Các hàm Đối tác mới (ĐÃ THÊM)
    calculate_partner_kpis, calculate_partner_revenue_metrics, create_partner_trend_chart,
    calculate_partner_breakdown_by_type,calculate_service_inventory, calculate_service_cancellation_metrics,
    calculate_partner_revenue_by_type,

    # CHỨC NĂNG MỚI CHO TAB 2
    calculate_booking_metrics, 
    create_cancellation_trend_chart, 
    create_demographic_pie_chart,
    create_ratio_trend_chart
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
    
    # Bổ sung Filter cho Tab 3
    st.markdown("---")
    st.subheader("Bộ lọc Đối tác")
    partners = ["Tất cả"] + sorted(tours_df['partner'].unique().tolist())
    selected_partner = st.selectbox("Chọn Đối tác", partners)
    
    service_types = ["Tất cả"] + sorted(tours_df['service_type'].unique().tolist())
    selected_service = st.selectbox("Chọn Loại dịch vụ", service_types)

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

# Áp dụng bộ lọc đối tác cho Tab 3
partner_filtered_df = tours_filtered_dimensional.copy()
if selected_partner != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['partner'] == selected_partner]
if selected_service != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['service_type'] == selected_service]

# Calculate KPIs using dimensionally filtered data (calculate_kpis will handle date filtering)
kpis = calculate_kpis(tours_filtered_dimensional, filtered_plans, start_date, end_date)


# Also create a date+dimension filtered version for charts that don't need historical data
filtered_tours = filter_data_by_date(tours_filtered_dimensional, start_date, end_date)

# TÍNH TOÁN BOOKING METRICS CHO TAB 2 (ĐÃ DI CHUYỂN)
booking_metrics = calculate_booking_metrics(tours_df, start_date, end_date)


if 'show_admin_ui' not in st.session_state:
    st.session_state.show_admin_ui = False

# Nút mở/đóng UI Admin (đặt ở khu vực trên cùng)
col_toggle, col_empty = st.columns([1, 4])

with col_toggle:
    if st.session_state.show_admin_ui:
        if st.button("<< Quay lại Dashboard Chính", type="secondary"):
            st.session_state.show_admin_ui = False
            st.rerun()
    else:
        if st.button("🔧 Mở UI Nhập liệu/Sửa Hợp đồng (Admin)", type="secondary"):
            st.session_state.show_admin_ui = True
            st.rerun()

# ----------------------------------------------------
# KHU VỰC HIỂN THỊ UI ADMIN LỚN
# ----------------------------------------------------
if st.session_state.show_admin_ui:
    render_admin_ui() # <--- GỌI HÀM TỪ FILE admin_ui.py







# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard theo dõi Kinh Doanh",
    "🔍 Dashboard theo dõi sản phẩm",
    "🤝 Dashboard theo dõi Đối tác" 
])

# ============================================================
# TAB 1: TỔNG QUAN (5 VÙNG THEO SPEC)
# ============================================================
with tab1:
    # ========== VÙNG 1: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### Vùng 1: Tốc độ đạt Kế hoạch")
    
    # Row: 3 Gauge charts + 1 Forecast chart
    col1, col2, col3 = st.columns(3)
    
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
    
# ========== BIỂU ĐỒ DỰ BÁO HOÀN THÀNH KẾ HOẠCH (SỬA LỖI 4 ĐỐI SỐ) ==========
# Hàng 2: Tiến độ KH theo Khu vực (1 cột) | Dự báo Hoàn thành KH (2 cột)
    st.markdown("#### Phân tích Tiến độ & Dự báo")
    col1, col2 = st.columns([1, 2]) # Tỉ lệ 1:2
    
    # Lấy dữ liệu cần thiết cho Hàng 2
    unit_performance = get_unit_performance(tours_filtered_dimensional, filtered_plans, start_date, end_date)
    
    with col1:
        st.markdown("##### 📊 Tiến độ KH theo Khu vực")
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
            fig.update_layout(xaxis_title="", yaxis_title="Tiến độ (%)", height=300, showlegend=False, margin=dict(l=30, r=30, t=10, b=30))
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu tiến độ cho khu vực kinh doanh được chọn.")
    
    with col2:
        st.markdown("##### 📈 Dự báo Hoàn thành Kế hoạch")
        fig_forecast = create_forecast_chart(
            filtered_tours, 
            filtered_plans, 
            start_date, 
            end_date,
            date_option
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    


    # ========== VÙNG 2: CHỈ SỐ TỔNG QUAN ==========
    st.markdown("###  Vùng 2: Các Chỉ số")
    
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

    # Row 3 (MỚI): Doanh thu trung bình/Khách (AOV)
    st.markdown("")
    col1, col2 = st.columns([1, 2]) # Vẫn dùng tỉ lệ 1:2 để căn chỉnh

    # Tính toán AOV
    aov = kpis['actual_revenue'] / kpis['actual_customers'] if kpis['actual_customers'] > 0 else 0
    ly_aov = kpis['ly_revenue'] / kpis['ly_customers'] if kpis['ly_customers'] > 0 else 0
    aov_growth = get_growth_rate(aov, ly_aov)

    with col1:
        st.metric(
            label="💵 DOANH THU TB/KHÁCH (AOV)",
            value=format_currency(aov),
            delta=f"{format_percentage(aov_growth)} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**AOV Cùng kỳ:** {format_currency(ly_aov)}")
            st.write(f"**Tăng trưởng AOV:** {format_percentage(aov_growth)}")
            st.write(f"**Doanh thu Tổng:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Lượt khách Tổng:** {format_number(kpis['actual_customers'])}")

    # Col 2 (trống) để căn chỉnh
    with col2:
        st.empty() 
    st.markdown("---")
    
    
    # ========== VÙNG 3: PHÂN THEO PHÂN KHÚC & ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 3: Phân theo Phân khúc & Đơn vị Kinh doanh")
    SEGMENT_COLORS = ['#3CB371', '#6495ED', '#FFA07A']
    BU_COLORS = ['#3CB371', '#6495ED', '#FFA07A', '#FF6347']
    
    # ----------------------------------------------------
    # PHẦN 1: PHÂN KHÚC (GIỮ NGUYÊN LOGIC)
    # ----------------------------------------------------
    st.markdown("#### Phân bổ theo Phân khúc (FIT / GIT / Inbound)")
    col1, col2, col3 = st.columns(3)
    
    # Get segment breakdown data
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_profit = get_segment_breakdown(filtered_tours, start_date, end_date, metric='profit')
    
    with col1:
        st.markdown("##### 💰 Doanh thu theo phân khúc")
        if not segment_revenue.empty:
            # (GIỮ NGUYÊN CODE TẠO BIỂU ĐỒ PIE SEGMENT)
            hovertext = []
            for seg in segment_revenue['segment']:
                unit_breakdown = get_segment_unit_breakdown(filtered_tours, start_date, end_date, seg, 'revenue')
                if not unit_breakdown.empty:
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
        st.markdown("##### 👥 Lượt khách theo phân khúc")
        if not segment_customers.empty:
            # (GIỮ NGUYÊN CODE TẠO BIỂU ĐỒ PIE SEGMENT)
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
        st.markdown("##### 💵 Lợi nhuận theo phân khúc")
        if not segment_profit.empty:
            # (GIỮ NGUYÊN CODE TẠO BIỂU ĐỒ PIE SEGMENT)
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


    # ----------------------------------------------------
    # PHẦN 2: PHÂN BỔ THEO ĐƠN VỊ KINH DOANH (ĐÃ THÊM)
    # ----------------------------------------------------
    st.markdown("#### Phân bổ theo Khu vực Đơn vị Kinh doanh")
    col1, col2, col3 = st.columns(3)
    
    # Lấy dữ liệu phân bổ theo Đơn vị Kinh doanh (CẦN get_unit_breakdown_simple trong utils.py)
    bu_revenue = get_unit_breakdown_simple(filtered_tours, metric='revenue')
    bu_customers = get_unit_breakdown_simple(filtered_tours, metric='customers')
    bu_profit = get_unit_breakdown_simple(filtered_tours, metric='profit')
    
    with col1:
        st.markdown("##### 💰 Doanh thu theo Khu vực")
        if not bu_revenue.empty:
            fig = go.Figure(go.Pie(
                labels=bu_revenue['business_unit'],
                values=bu_revenue['value'],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Doanh thu: %{value:,.0f} ₫<br>Tỉ lệ: %{percent}<extra></extra>',
                marker=dict(colors=BU_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
            
    with col2:
        st.markdown("##### 👥 Lượt khách theo Khu vực")
        if not bu_customers.empty:
            fig = go.Figure(go.Pie(
                labels=bu_customers['business_unit'],
                values=bu_customers['value'],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Lượt khách: %{value:,.0f}<br>Tỉ lệ: %{percent}<extra></extra>',
                marker=dict(colors=BU_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
            
    with col3:
        st.markdown("##### 💵 Lợi nhuận theo Khu vực")
        if not bu_profit.empty:
            fig = go.Figure(go.Pie(
                labels=bu_profit['business_unit'],
                values=bu_profit['value'],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Lợi nhuận: %{value:,.0f} ₫<br>Tỉ lệ: %{percent}<extra></extra>',
                marker=dict(colors=BU_COLORS)
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    st.markdown("---")
    
# ========== VÙNG 4: CÁC BẢNG THÔNG TIN KHÁC ==========
    st.markdown("### Vùng 4: Các bảng thông tin khác")
    
    # Chuẩn bị dữ liệu cho cả 3 chỉ số
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')
    
    # --- HÀNG 1: DOANH THU (Đã có sẵn, điều chỉnh lại) ---
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("##### 🎯 Top 10 Tuyến Tour (Doanh thu)")
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
        else:
            st.info("Không có dữ liệu Top 10 Tuyến Tour.")
    
    with col2:
        st.markdown("##### 📊 Tỉ trọng các tuyến (%) (Doanh thu)")
        if not top_revenue.empty:
            labels = [route if len(route) <= 12 else route[:10] + ".." for route in top_revenue['route']]
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
        else:
            st.info("Không có dữ liệu tỉ trọng tuyến.")

    st.markdown("---")
    
    # --- HÀNG 2: LƯỢT KHÁCH (ĐÃ THÊM) ---
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("##### 🎯 Top 10 Tuyến Tour (Lượt khách)")
        if not top_customers.empty:
            fig = go.Figure()
            # Lấy breakdown theo Đơn vị cho hover (giá trị là Lượt khách)
            hovertext = []
            for route in top_customers['route'][::-1]:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route, metric='customers') # <--- Metric = customers
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_number(row['num_customers'])} ({row['percentage']:.1f}%)"
                                                    for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            fig.add_trace(go.Bar(
                y=top_customers['route'][::-1],
                x=top_customers['num_customers'][::-1],
                orientation='h',
                text=[format_number(v) for v in top_customers['num_customers'][::-1]],
                textposition='outside',
                marker_color='#FF97FF', # Màu khác cho Lượt khách
                customdata=hovertext,
                hovertemplate='<b>%{y}</b><br>Tổng LK: %{x:,.0f}<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(xaxis_title="", yaxis_title="", height=230, showlegend=False, margin=dict(l=100, r=30, t=10, b=30))
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu Top 10 Tuyến Tour.")
            
    with col2:
        st.markdown("##### 📊 Tỉ trọng các tuyến (%) (Lượt khách)")
        if not top_customers.empty:
            labels = [route if len(route) <= 12 else route[:10] + ".." for route in top_customers['route']]
            
            hovertext = []
            for route in top_customers['route']:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route, metric='customers') # <--- Metric = customers
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_number(row['num_customers'])} ({row['percentage']:.1f}%)"
                                                    for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=labels,
                values=top_customers['num_customers'],
                textposition='outside',
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>Lượt khách: %{value:,.0f}<br>Tỉ lệ: %{percent}<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu tỉ trọng tuyến.")

    st.markdown("---")

    # --- HÀNG 3: LỢI NHUẬN (ĐÃ THÊM) ---
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### 🎯 Top 10 Tuyến Tour (Lợi nhuận)")
        if not top_profit.empty:
            fig = go.Figure()
            hovertext = []
            for route in top_profit['route'][::-1]:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route, metric='profit') # <--- Metric = profit
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_currency(row['gross_profit'])} ({row['percentage']:.1f}%)"
                                                    for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            fig.add_trace(go.Bar(
                y=top_profit['route'][::-1],
                x=top_profit['gross_profit'][::-1],
                orientation='h',
                text=[format_currency(v) for v in top_profit['gross_profit'][::-1]],
                textposition='outside',
                marker_color='#FFA15A', # Màu khác cho Lợi nhuận
                customdata=hovertext,
                hovertemplate='<b>%{y}</b><br>Tổng LN: %{x:,.0f} ₫<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(xaxis_title="", yaxis_title="", height=230, showlegend=False, margin=dict(l=100, r=30, t=10, b=30))
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu Top 10 Tuyến Tour.")

    with col2:
        st.markdown("##### 📊 Tỉ trọng các tuyến (%) (Lợi nhuận)")
        if not top_profit.empty:
            labels = [route if len(route) <= 12 else route[:10] + ".." for route in top_profit['route']]
            
            hovertext = []
            for route in top_profit['route']:
                unit_breakdown = get_route_unit_breakdown(filtered_tours, route, metric='profit') # <--- Metric = profit
                if not unit_breakdown.empty:
                    breakdown_text = "<br>".join([f"{row['business_unit']}: {format_currency(row['gross_profit'])} ({row['percentage']:.1f}%)"
                                                    for _, row in unit_breakdown.iterrows()])
                    hovertext.append(breakdown_text)
                else:
                    hovertext.append("")
            
            fig = go.Figure(go.Pie(
                labels=labels,
                values=top_profit['gross_profit'],
                textposition='outside',
                textinfo='label+percent',
                customdata=hovertext,
                hovertemplate='<b>%{label}</b><br>Lợi nhuận: %{value:,.0f} ₫<br>Tỉ lệ: %{percent}<br><br><b>Theo đơn vị:</b><br>%{customdata}<extra></extra>'
            ))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu tỉ trọng tuyến.")

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
        st.plotly_chart(fig_occ, key="gauge_tab1")
    
    with col2:
        fig_cancel = create_gauge_chart(
            ops_metrics['cancel_rate'],
            "Tỷ lệ Khách Hủy/Hoãn",
            max_value=30,
            threshold=10,
            is_inverse_metric=True
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
# ========== VÙNG 1: TÓM TẮT HIỆU SUẤT BOOKING (ĐÃ THÊM KPI VÀ TRENDS) ==========
    st.markdown("### Vùng 1: Tóm tắt Hiệu suất Booking")
    
    # --- Hàng 1: KPI Cấp cao ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Số lượng khách đã đặt",
            value=format_number(booking_metrics['total_booked_customers'])
        )

    with col2:
        st.metric(
            label="💰 Tổng Doanh thu",
            value=format_currency(kpis['actual_revenue'])
        )
    with col3:
        st.markdown("##### 📈 Tỷ lệ Lấp đầy BQ")
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ",
            max_value=100, 
            threshold=75,
            is_inverse_metric=False
        )
        st.plotly_chart(fig_occ, use_container_width=True, key="gauge_tab2")
    with col4:
        st.empty()

    st.markdown("---")


    # --- Hàng 2: Tỷ lệ Thành công (Gauge & Trend) ---
    st.markdown("#### 🟢 Hiệu suất Booking Thành công")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ booking thành công (Gauge Chart)
        fig_success = create_gauge_chart(
            booking_metrics['success_rate'],
            "Tỷ lệ booking thành công",
            max_value=100, 
            threshold=90
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        # Xu hướng tỷ lệ booking thành công (Line Chart)
        fig_success_trend = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                     metric='success_rate', 
                                                     title='Xu hướng Tỷ lệ Booking Thành công (Theo ngày/tuần)')
        st.plotly_chart(fig_success_trend, use_container_width=True)

    st.markdown("---")


    # --- Hàng 3: Tỷ lệ Hủy/Đổi (Gauge & Trend) ---
    st.markdown("#### 🔴 Hiệu suất Khách Hàng Hủy/Đổi")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ khách hàng hủy tour hoặc thay đổi (Gauge Chart)
        fig_cancel = create_gauge_chart(
            booking_metrics['cancel_change_rate'],
            "Tỷ lệ Khách Hủy/Đổi",
            max_value=30, 
            threshold=15, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_cancel, use_container_width=True)
        
    with col2:
        # Xu hướng tỷ lệ khách hàng hủy tour (Line Chart)
        fig_cancel_trend_ratio = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                           metric='cancellation_rate', 
                                                           title='Xu hướng Tỷ lệ Khách Hủy/Đổi (Theo ngày/tuần)')
        st.plotly_chart(fig_cancel_trend_ratio, use_container_width=True)

    st.markdown("---")


    # ========== VÙNG 2: THEO TUYẾN ==========
    st.markdown("### Vùng 2: Phân tích theo Tuyến")
    
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

    # Row 2: Profit margin with color coding
    st.markdown("#### Tỷ suất Lợi nhuận theo Tuyến")
    if not route_table.empty:
        top_10_margin = route_table.nlargest(10, 'profit_margin')[['route', 'profit_margin']]
        fig = create_profit_margin_chart_with_color(top_10_margin, 'profit_margin', 'route', '')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # Row 3: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Tuyến")
    if not route_table.empty:
        display_df = route_table.copy()
        display_df = display_df[[
            'route', 'revenue', 'num_customers', 'gross_profit', 
            'profit_margin', 'revenue_completion', 'occupancy_rate', 'cancel_rate'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['revenue_completion'] = display_df['revenue_completion'].apply(lambda x: f"{x:.1f}%")
        display_df['occupancy_rate'] = display_df['occupancy_rate'].apply(lambda x: f"{x:.1f}%")
        display_df['cancel_rate'] = display_df['cancel_rate'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['Tuyến', 'Doanh thu', 'Lượt khách', 'Lợi nhuận gộp', 
                      'Tỷ suất LN (%)', 'Tiến độ KH (%)', 'Tỷ lệ Lấp đầy (%)', 'Tỷ lệ Hủy/Đổi (%)']

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    

    
    # ========== VÙNG 3: THEO KÊNH BÁN VÀ PHÂN KHÚC ==========
    st.markdown("### Vùng 3: Theo Kênh bán và Phân khúc")
    
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

# ========== VÙNG 4: XU HƯỚNG VÀ NHÂN KHẨU HỌC (MỚI) ==========
    st.markdown("### Vùng 4: Xu hướng và Nhân khẩu học")

    # Hàng 1: 2 Biểu đồ Xu hướng (Revenue Trend, Cancellation Trend)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Xu hướng Doanh thu theo thời kỳ")
        # Xu hướng doanh thu theo từng thời kỳ (Line Chart)
        fig_rev_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue'])
        st.plotly_chart(fig_rev_trend, use_container_width=True)
        
    with col2:
        st.markdown("##### Xu hướng Khách hàng hủy/đổi tour")
        # Xu hướng khách hàng hủy tour (Line Chart)
        fig_cancel_trend = create_cancellation_trend_chart(tours_df, start_date, end_date)
        st.plotly_chart(fig_cancel_trend, use_container_width=True)

    # Hàng 2: 2 Biểu đồ Tỷ trọng (Age, Nationality)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Tỷ trọng Doanh thu theo Độ tuổi")
        # Tỷ trọng doanh thu khách hàng theo độ tuổi (Pie Chart)
        # Giả định cột customer_age_group tồn tại
        fig_age_pie = create_demographic_pie_chart(filtered_tours, 'customer_age_group', '')
        st.plotly_chart(fig_age_pie, use_container_width=True)

    with col2:
        st.markdown("##### Tỷ trọng Doanh thu theo Quốc tịch")
        # Tỷ trọng doanh thu khách hàng theo quốc tịch (Pie Chart)
        # Giả định cột customer_nationality tồn tại
        fig_nat_pie = create_demographic_pie_chart(filtered_tours, 'customer_nationality', '')
        st.plotly_chart(fig_nat_pie, use_container_width=True)
        
    st.markdown("---")



    # ========== VÙNG 5: THEO ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 5: Hiệu suất theo Đơn vị Kinh doanh")
    
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


# ============================================================
# TAB 3: ĐỐI TÁC (TÁI CẤU TRÚC HOÀN CHỈNH)
# ============================================================
with tab3:
    st.title("🤝 Dashboard Quản lý Dịch vụ và Đối tác")
    
    # Lấy dữ liệu đã lọc theo Đối tác/Dịch vụ
    # Giả định các hàm tính toán đã được định nghĩa trong utils.py hoặc được import
    partner_filtered_data = filter_data_by_date(partner_filtered_df, start_date, end_date)
    partner_kpis = calculate_partner_kpis(partner_filtered_data)
    partner_revenue_metrics = calculate_partner_revenue_metrics(partner_filtered_data)
    service_cancel_metrics = calculate_service_cancellation_metrics(partner_filtered_data)
    service_inventory_total = calculate_service_inventory(partner_filtered_data)['total_units'].sum()
    partner_performance = calculate_partner_performance(partner_filtered_data) 
    
    # Dữ liệu phân tích chi tiết theo loại (cho Expander Vùng 1)
    active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
    expiring_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Sắp hết hạn")
    
    # --- VÙNG 1: TỔNG QUAN KPIs VÀ CẢNH BÁO (ĐÃ THÊM CHI TIẾT DỊCH VỤ) ---
    st.markdown("### 🎯 Vùng 1: Tổng quan Đối tác & Cảnh báo Hợp đồng")
    
    # Hàng 1: 4 KPI Cards tập trung
    col1, col2, col3, col4 = st.columns(4)
    
    # Tổng đối tác Đang triển khai
    with col1:
        st.metric(
            label="🤝 Tổng đối tác Đang triển khai",
            delta=" Tăng 2",
            value=format_number(partner_kpis['total_active_partners'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Đang triển khai"):
            for _, row in active_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")
        
    # Hợp đồng Sắp hết hạn (Cảnh báo)
    with col2:
        expiring_contracts = partner_kpis['contracts_status_count'][partner_kpis['contracts_status_count']['status'] == 'Sắp hết hạn']['count'].sum()
        st.metric(
            label="🚨 Hợp đồng Sắp hết hạn",
            value=format_number(expiring_contracts),
            delta="Cần gia hạn",
            delta_color="inverse"
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Sắp hết hạn"):
            for _, row in expiring_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} hợp đồng")
        
    # Tổng Doanh thu dịch vụ (Revenue)
    with col3:
        st.metric(
            label="💰 Tổng Dịch vụ đang giữ",
            delta=" Tăng 2 tỷ",
            value=format_currency(partner_kpis['total_service_revenue'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        # Giả định hàm calculate_partner_revenue_by_type trả về DataFrame: type, revenue
        revenue_by_type = calculate_partner_revenue_by_type(partner_filtered_data) # <--- Cần hàm này trong utils.py
        with st.expander("Chi tiết: Doanh thu theo Loại DV"):
            for _, row in revenue_by_type.iterrows():
                st.write(f"**{row['service_type']}**: {format_currency(row['revenue'])}")
        
    # Tình trạng Hủy dịch vụ (Gauge Chart)
    with col4:
        st.markdown("##### Tỷ lệ Hủy Dịch vụ")
        fig_service_cancel = create_gauge_chart(
            service_cancel_metrics['cancel_rate'],
            "Tỷ lệ Hủy Dịch vụ",
            max_value=30, 
            threshold=10, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_service_cancel, use_container_width=True)

    st.markdown("---")
    
    
    # --- VÙNG 2: PHÂN TÍCH TÌNH TRẠNG HỢP ĐỒNG & PHÂN TÍCH DỊCH VỤ (ĐÃ SỬA CHÚ THÍCH) ---
    st.markdown("### 📊 Vùng 2: Trạng thái Hợp đồng & Phân tích Dịch vụ")
    
    # Dữ liệu cho biểu đồ tròn (Tỷ trọng Trả trước/Trả sau)
    payment_status_data = partner_filtered_data.groupby('payment_status')['partner'].count().reset_index()
    payment_status_data.columns = ['status', 'count']
    
    col_status, col_price = st.columns([1, 2])
    
    # 1. Biểu đồ: Tỷ trọng Trạng thái Thanh toán (Pie Chart)
    with col_status:
        st.markdown("##### Tỷ trọng Thanh toán Hợp đồng")
        payment_data = payment_status_data[payment_status_data['status'].isin(['Trả trước', 'Trả sau'])].copy()
        total_payment_contracts = payment_data['count'].sum() # TỔNG HỢP ĐỒNG
        
        if not payment_data.empty:
            count_prepaid = payment_data[payment_data['status'] == 'Trả trước']['count'].iloc[0] if 'Trả trước' in payment_data['status'].values else 0
            count_postpaid = payment_data[payment_data['status'] == 'Trả sau']['count'].iloc[0] if 'Trả sau' in payment_data['status'].values else 0
            
            # --- HIỂN THỊ CHÚ THÍCH MỚI ---
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; text-align: center; margin-bottom: 5px;">
                Tổng Hợp đồng: {format_number(total_payment_contracts)}
            </div>
            <div style="font-size: 13px; text-align: center; margin-bottom: 5px;">
                <span style="color: #636EFA;">■ Trả trước:</span> {format_number(count_prepaid)} hợp đồng
                <span style="color: #FFA15A; margin-left: 15px;">■ Trả sau:</span> {format_number(count_postpaid)} hợp đồng
            </div>
            """, unsafe_allow_html=True)
            
            # --- TẠO BIỂU ĐỒ TRÒN (TẮT CHÚ THÍCH TỰ ĐỘNG) ---
            fig_payment_pie = px.pie(
                payment_data, 
                values='count', 
                names='status',
                color_discrete_sequence=['#636EFA', '#FFA15A'],
            )
            
            fig_payment_pie.update_traces(textinfo='percent+label', 
                                            hovertemplate='<b>%{label}</b><br>Số lượng: %{value:,.0f}<br>Tỉ lệ: %{percent}<extra></extra>')
            
            fig_payment_pie.update_layout(
                height=300, # Đã chỉnh height thấp hơn
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            
            st.plotly_chart(fig_payment_pie, use_container_width=True)
        else:
            st.info("Không có dữ liệu hợp đồng Trả trước/Trả sau.")
            
        # Thống kê chi tiết
        active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
        with st.expander("Phân loại Đối tác Đang triển khai"):
             for _, row in active_breakdown.iterrows():
                 st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")

    # 2. Bar Chart: Giá Dịch vụ (Giá TB/Khách)
    with col_price:
        st.markdown("##### Phân tích Giá Dịch vụ (Max, Avg, Min)")
        if not partner_revenue_metrics.empty:
            df_melted = partner_revenue_metrics.melt(
                id_vars='service_type',
                value_vars=['max_price', 'avg_price', 'min_price'],
                var_name='price_type',
                value_name='price_value'
            )
            
            df_melted['price_type'] = df_melted['price_type'].replace({
                'max_price': 'Giá Cao nhất',
                'avg_price': 'Giá Trung bình',
                'min_price': 'Giá Thấp nhất'
            })
            
            fig_price_comp = px.bar(
                df_melted,
                x='price_value',
                y='service_type',
                color='price_type',
                orientation='h',
                title='Giá Dịch vụ theo Loại (Max, Avg, Min)',
                barmode='group'
            )
            fig_price_comp.update_xaxes(title="Giá (₫)")
            fig_price_comp.update_traces(hovertemplate='%{x:,.0f} ₫<extra></extra>')
            fig_price_comp.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, margin=dict(t=30))
            st.plotly_chart(fig_price_comp, use_container_width=True)
        
    st.markdown("---")


    # --- VÙNG 3: XU HƯỚNG VÀ HIỆU QUẢ HỢP TÁC ---
    st.markdown("### 📈 Vùng 3: Xu hướng và Hiệu quả Hợp tác")
    
    # Row 1: Biểu đồ Doanh thu và Số lượng khách theo thời gian
    col_trend, col_scatter = st.columns(2)
    
    with col_trend:
        st.markdown("##### Xu hướng Doanh thu và Lượt khách từ Đối tác")
        fig_partner_trend = create_partner_trend_chart(partner_filtered_df, start_date, end_date)
        st.plotly_chart(fig_partner_trend, use_container_width=True)
    
    with col_scatter:
        st.markdown("##### Đánh giá Hiệu quả Từng Đối tác")
        if not partner_performance.empty:
            # Biểu đồ Bong bóng: X=Doanh thu, Y=Tỷ lệ Phản hồi, Size=Số lượng khách
            fig_scatter = px.scatter(
                partner_performance,
                x='total_revenue',
                y='avg_feedback',
                size='total_customers',
                color='partner',
                hover_name='partner',
                title='Hiệu quả Đối tác (DT vs Phản hồi Tích cực)',
                labels={'total_revenue': 'Doanh thu (₫)', 'avg_feedback': 'Tỷ lệ phản hồi tích cực (%)', 'total_customers': 'Lượt khách'}
            )
            fig_scatter.update_traces(hovertemplate='<b>%{hovertext}</b><br>Doanh thu: %{x:,.0f} ₫<br>Phản hồi: %{y:.1%}<br>Lượt khách: %{marker.size:,.0f}<extra></extra>')
            fig_scatter.update_layout(height=400, showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Bảng chi tiết Doanh thu/Chi phí/Lợi nhuận
    st.markdown("#### Bảng Chi tiết Hợp đồng và Tỷ suất Lợi nhuận")
    
    # Lấy bảng hợp đồng chi tiết
    df_partner_revenue_detail = partner_filtered_data.groupby(['partner', 'service_type', 'payment_status', 'contract_status']).agg(
        total_revenue=('revenue', 'sum'),
        total_service_cost=('service_cost', 'sum'),
        num_bookings=('booking_id', 'count')
    ).reset_index()
    
    df_partner_revenue_detail['profit_margin'] = np.where(
        df_partner_revenue_detail['total_revenue'] > 0,
        ((df_partner_revenue_detail['total_revenue'] - df_partner_revenue_detail['total_service_cost']) / df_partner_revenue_detail['total_revenue']) * 100,
        0
    )
    
    # Áp dụng formatting
    df_partner_revenue_detail['total_revenue'] = df_partner_revenue_detail['total_revenue'].apply(format_currency)
    df_partner_revenue_detail['total_service_cost'] = df_partner_revenue_detail['total_service_cost'].apply(format_currency)
    df_partner_revenue_detail['profit_margin'] = df_partner_revenue_detail['profit_margin'].apply(lambda x: f"{x:.1f}%")

    df_partner_revenue_detail.rename(columns={
        'contract_status': 'Trạng thái HĐ', 
        'service_type': 'Loại DV', 
        'payment_status': 'Tình trạng TT', 
        'total_revenue': 'Doanh thu',
        'total_service_cost': 'Chi phí DV',
        'num_bookings': 'SL HĐ',
        'profit_margin': 'Tỷ suất LN (%)'
    }, inplace=True)
    
    # Hàm highlight_expiring (Giữ nguyên)
    def highlight_expiring(s):
        if s['Trạng thái HĐ'] == 'Sắp hết hạn':
            return ['background-color: #ffe0e0; color: red'] * len(s)
        return [''] * len(s)

    st.dataframe(
        df_partner_revenue_detail[['partner', 'Loại DV', 'Doanh thu', 'Chi phí DV', 'Tỷ suất LN (%)', 'Trạng thái HĐ', 'Tình trạng TT']]
        .style.apply(highlight_expiring, axis=1), 
        use_container_width=True, hide_index=True
    )

st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)