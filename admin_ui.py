import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import random 

def render_admin_ui():
    """
    Renders the dedicated Admin UI for creating new contracts or editing existing ones,
    modifying financial and status data directly in st.session_state.tours_df.
    """
    
    if 'tours_df' not in st.session_state or st.session_state.tours_df.empty:
        st.error("Lỗi: Dữ liệu Tour chưa được tải vào Session State. Vui lòng làm mới trang.")
        return

    tours_df = st.session_state.tours_df
    
    st.header("⚙️ Nhập liệu/Sửa Hợp đồng")

    option = st.radio(
        "Chọn chế độ:",
        ("Sửa Hợp đồng Hiện tại", "Nhập Hợp đồng Mới"),
        index=0
    )

    # Danh sách các lựa chọn mặc định từ dữ liệu hiện có
    unique_route = sorted(tours_df['route'].unique())
    unique_unit = sorted(tours_df['business_unit'].unique())
    unique_channel = sorted(tours_df['sales_channel'].unique())
    unique_segment = sorted(tours_df['segment'].unique())
    
    # Các lựa chọn cho Tab Đối tác (Giả định/Mặc định)
    # Lấy các giá trị đã tồn tại trong DataFrame để đảm bảo chế độ SỬA không bị lỗi
    partner_options = sorted(tours_df['partner'].unique().tolist()) if 'partner' in tours_df.columns and tours_df['partner'].any() else ["Đối tác 1", "Đối tác 2"]
    service_type_options = sorted(tours_df['service_type'].unique().tolist()) if 'service_type' in tours_df.columns and tours_df['service_type'].any() else ["Vé máy bay", "Khách sạn", "Vận chuyển", "Ăn uống"]
    contract_status_options = sorted(tours_df['contract_status'].unique().tolist()) if 'contract_status' in tours_df.columns and tours_df['contract_status'].any() else ["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"]
    payment_status_options = sorted(tours_df['payment_status'].unique().tolist()) if 'payment_status' in tours_df.columns and tours_df['payment_status'].any() else ["Trả trước", "Trả sau", "Chưa thanh toán"]

    # Khởi tạo giá trị cho Form
    selected_contract = ""
    mode_key = "default"
    revenue_val = 0
    profit_val = 0
    status_val = "Đã xác nhận"
    marketing_cost_val = 0
    sales_cost = 0
    
    partner_val = partner_options[0] if partner_options else "N/A"
    service_type_val = service_type_options[0] if service_type_options else "N/A"
    contract_status_val = contract_status_options[0] if contract_status_options else "N/A"
    payment_status_val = payment_status_options[0] if payment_status_options else "N/A"
    service_cost_val = 0
    
    
    if option == "Sửa Hợp đồng Hiện tại":
        # CHẾ ĐỘ SỬA
        contract_ids = tours_df['booking_id'].unique().tolist()
        if not contract_ids:
            st.warning("Không có hợp đồng nào để sửa.")
            st.stop()
            
        selected_contract = st.selectbox("Chọn Mã Hợp đồng để sửa", contract_ids)
        current_contract_data = tours_df[tours_df['booking_id'] == selected_contract].iloc[0]
        mode_key = f"edit_{selected_contract}"
        
        # Lấy giá trị hiện tại từ tours_df
        revenue_val = int(current_contract_data['revenue'])
        profit_val = int(current_contract_data['gross_profit'])
        status_val = current_contract_data['status']
        marketing_cost_val = int(current_contract_data['marketing_cost'])
        sales_cost = float(current_contract_data['sales_cost']) 
        
        # Lấy giá trị đối tác (FIX LỖI: Đảm bảo giá trị hiện tại có trong options)
        partner_val = current_contract_data.get('partner', partner_options[0])
        if partner_val not in partner_options: partner_options.append(partner_val)
        
        service_type_val = current_contract_data.get('service_type', service_type_options[0])
        if service_type_val not in service_type_options: service_type_options.append(service_type_val)
        
        contract_status_val = current_contract_data.get('contract_status', contract_status_options[0])
        if contract_status_val not in contract_status_options: contract_status_options.append(contract_status_val)
        
        payment_status_val = current_contract_data.get('payment_status', payment_status_options[0])
        if payment_status_val not in payment_status_options: payment_status_options.append(payment_status_val)
        
        service_cost_val = current_contract_data.get('service_cost', revenue_val - profit_val)
        
    else:
        # CHẾ ĐỘ NHẬP MỚI
        new_id = f"NEW{datetime.now().strftime('%d%H%M%S')}"
        selected_contract = new_id
        st.text_input("Mã Hợp đồng Mới", value=selected_contract, disabled=True)
        mode_key = "new_contract"
        
        # Giá trị mặc định cho hợp đồng mới
        revenue_val = 15000000
        profit_val = 3000000
        status_val = "Đã xác nhận"
        marketing_cost_val = 150000
        sales_cost = 0
        service_cost_val = revenue_val - profit_val


    # FORM NHẬP LIỆU CHUNG
    with st.container(border=True):
        st.subheader(f"Dữ liệu {selected_contract}")
        
        with st.form(key=mode_key):
            
            # CÁC CỘT ĐƯỢC CHIA LÀM 2 CỘT NHỎ HƠN
            col_a, col_b = st.columns(2)
            
            # ----------------------------------------------------
            # CỘT A: THÔNG TIN ĐỐI TÁC & TOUR CƠ BẢN
            # ----------------------------------------------------
            with col_a:
                st.markdown("##### 1. Thông tin Đối tác & Tour")
                
                # Thông tin Đối tác/Dịch vụ
                input_partner = st.selectbox("Tên Đối tác", options=partner_options, index=partner_options.index(partner_val), key=f"{mode_key}_partner")
                input_service_type = st.selectbox("Loại Dịch vụ", options=service_type_options, index=service_type_options.index(service_type_val), key=f"{mode_key}_service_type")
                
                input_contract_status = st.selectbox("Trạng thái HĐ", options=contract_status_options, index=contract_status_options.index(contract_status_val), key=f"{mode_key}_contract_status")
                input_payment_status = st.selectbox("Tình trạng TT", options=payment_status_options, index=payment_status_options.index(payment_status_val), key=f"{mode_key}_payment_status")
                
                # Thông tin Tour (Chỉ chỉnh sửa ở chế độ nhập mới)
                if option == "Nhập Hợp đồng Mới":
                    new_customer_id = st.text_input("Mã Khách hàng", value=f"KH_A{random.randint(1000, 9999)}", key="new_cust_id")
                    new_route = st.selectbox("Tuyến Tour", options=unique_route, key="new_route")
                    new_unit = st.selectbox("Đơn vị Kinh doanh", options=unique_unit, key="new_unit")
                    new_customers_count = st.number_input("Số lượng Khách", value=4, min_value=1, key="new_cust_count")
                else:
                    st.text_input("Mã Khách hàng", value=current_contract_data['customer_id'], disabled=True)
                    st.text_input("Tuyến Tour", value=current_contract_data['route'], disabled=True)
                    st.text_input("Đơn vị KD", value=current_contract_data['business_unit'], disabled=True)
                    st.number_input("Số lượng Khách", value=int(current_contract_data['num_customers']), min_value=1, disabled=True)


            # ----------------------------------------------------
            # CỘT B: DỮ LIỆU TÀI CHÍNH
            # ----------------------------------------------------
            with col_b:
                st.markdown("##### 2. Dữ liệu Tài chính")
                
                input_revenue = st.number_input("Doanh thu (₫)", value=revenue_val, min_value=0, step=100000, key=f"{mode_key}_rev")
                input_profit = st.number_input("Lợi nhuận gộp (₫)", value=profit_val, min_value=0, step=100000, key=f"{mode_key}_profit")
                
                input_service_cost = st.number_input("Chi phí Dịch vụ (service_cost)", 
                                                     value=int(service_cost_val), 
                                                     min_value=0, step=100000, 
                                                     key=f"{mode_key}_service_cost")

                input_marketing_cost = st.number_input("Chi phí Marketing (₫)", value=marketing_cost_val, min_value=0, step=100000, key=f"{mode_key}_mkt")
                input_status = st.selectbox("Trạng thái Booking", options=["Đã xác nhận", "Đã hủy", "Hoãn"], index=["Đã xác nhận", "Đã hủy", "Hoãn"].index(status_val), key=f"{mode_key}_status")

            # --- NÚT SUBMIT (Phải nằm ngoài col_a, col_b nhưng trong form) ---
            submitted = st.form_submit_button("Lưu & Cập nhật Dashboard", type="primary")

            if submitted:
                # LÔ-GÍC CẬP NHẬT/THÊM MỚI
                
                # 1. Xác định giá trị cuối cùng cho các trường
                if option == "Nhập Hợp đồng Mới":
                    sales_cost_final = input_revenue * 0.05
                    num_cust_final = new_customers_count
                    
                    partner_final = input_partner
                    service_type_final = input_service_type
                    contract_status_final = input_contract_status
                    payment_status_final = input_payment_status
                    route_final = new_route
                    unit_final = new_unit
                    customer_id_final = new_customer_id
                    
                else:
                    sales_cost_final = sales_cost
                    num_cust_final = current_contract_data['num_customers']
                    
                    partner_final = input_partner
                    service_type_final = input_service_type
                    contract_status_final = input_contract_status
                    payment_status_final = input_payment_status
                    route_final = current_contract_data['route']
                    unit_final = current_contract_data['business_unit']
                    customer_id_final = current_contract_data['customer_id']
                
                
                new_opex = input_marketing_cost + sales_cost_final
                
                if input_revenue > 0:
                    price_per_person_final = input_revenue / num_cust_final if num_cust_final > 0 else input_revenue
                    margin_final = (input_profit / input_revenue) * 100
                else:
                    price_per_person_final = 0
                    margin_final = 0
                    
                # 2. Xây dựng Row mới (bao gồm các cột mới cho Đối tác/Dịch vụ)
                new_row = {
                    # Cột cần thiết cho Tab 3 (Đối tác)
                    'partner': partner_final,
                    'service_type': service_type_final,
                    'contract_status': contract_status_final,
                    'payment_status': payment_status_final,
                    'service_cost': input_service_cost, 
                    'feedback_ratio': np.random.uniform(0.7, 0.95), # Giả định giá trị phản hồi
                    
                    # Cột cần thiết cho Tab 1 & 2 (Tour chính)
                    'booking_id': selected_contract,
                    'customer_id': customer_id_final,
                    'booking_date': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                    'route': route_final,
                    'business_unit': unit_final,
                    'sales_channel': current_contract_data['sales_channel'] if option == 'Sửa Hợp đồng Hiện tại' else unique_channel[0],
                    'segment': current_contract_data['segment'] if option == 'Sửa Hợp đồng Hiện tại' else unique_segment[0],
                    'num_customers': num_cust_final,
                    'tour_capacity': current_contract_data['tour_capacity'] if option == 'Sửa Hợp đồng Hiện tại' else 30,
                    'price_per_person': price_per_person_final,
                    'revenue': input_revenue,
                    'cost': input_revenue - input_profit,
                    'gross_profit': input_profit,
                    'gross_profit_margin': margin_final,
                    'status': input_status,
                    'marketing_cost': input_marketing_cost,
                    'sales_cost': sales_cost_final,
                    'opex': new_opex
                }
                
                # 3. Thêm/Sửa Row vào DataFrame
                if option == "Sửa Hợp đồng Hiện tại":
                    idx = st.session_state.tours_df[st.session_state.tours_df['booking_id'] == selected_contract].index[0]
                    for key, val in new_row.items():
                        st.session_state.tours_df.loc[idx, key] = val
                    st.success(f"✅ Hợp đồng {selected_contract} đã được cập nhật!")
                else:
                    new_df = pd.DataFrame([new_row])
                    # Đồng bộ hóa các cột mới vào tours_df gốc nếu chúng chưa tồn tại
                    for col in new_df.columns:
                        if col not in st.session_state.tours_df.columns:
                            st.session_state.tours_df[col] = np.nan
                            
                    st.session_state.tours_df = pd.concat([st.session_state.tours_df, new_df], ignore_index=True)
                    st.success(f"✅ Đã thêm Hợp đồng MỚI: {selected_contract}!")
                
                st.session_state.show_admin_ui = False
                st.rerun()

    st.stop()