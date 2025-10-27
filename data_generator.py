"""
Data generator for Vietravel Business Intelligence Dashboard
Generates realistic mock data for tour sales, customers, and operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import pytz # Thêm pytz

# Initialize Faker with Vietnamese locale
fake = Faker(['vi_VN'])

class VietravelDataGenerator:
    """Generates realistic mock data for Vietravel tour business"""
    
    def __init__(self, seed=42):
        """Initialize the data generator with a seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Define tour routes (tuyến tour)
        self.tour_routes = [
            "DH & ĐBSH",
            "Nam Trung Bộ",
            "Bắc Trung Bộ",
            "Liên Tuyến miền Tây",
            "Phú Quốc",
            "Thái Lan",
            "Trung Quốc",
            "Hàn Quốc",
            "Singapore - Malaysia",
            "Nhật Bản",
            "Châu Âu",
            "Châu Mỹ",
            "Châu Úc",
            "Châu Phi",
            "Tây Bắc",
            "Đông Bắc",
            "Tây Nguyên"
        ]
        
        # Business units (đơn vị kinh doanh)
        self.business_units = [
            "Miền Trung",
            "Miền Tây",
            "Miền Bắc",
            "Trụ sở & ĐNB"
        ]
        
        # Sales channels (kênh bán)
        self.sales_channels = [
            "Online",
            "Trực tiếp VPGD",
            "Đại lý"
        ]
        
        # Segments (phân khúc)
        self.segments = [
            "FIT",  # Free Independent Traveler
            "GIT",  # Group Inclusive Tour
            "Inbound"  # International visitors
        ]
        
        # Map routes to business units
        self.route_to_unit = {
            "DH & ĐBSH": "Miền Bắc",
            "Tây Nguyên": "Miền Tây",
            "Bắc Trung Bộ": "Miền Trung",
            "Phú Quốc": "Miền Tây",
            "Liên Tuyến miền Tây": "Miền Tây",
            "Nam Trung Bộ": "Miền Trung",
            "Đông Bắc": "Miền Bắc",
            "Tây Bắc": "Miền Bắc",
            "Singapore - Malaysia": "Trụ sở & ĐNB",
            "Hàn Quốc": "Trụ sở & ĐNB",
            "Nhật Bản": "Trụ sở & ĐNB",
            "Trung Quốc": "Trụ sở & ĐNB",
            "Thái Lan": "Trụ sở & ĐNB",
            "Châu Âu": "Trụ sở & ĐNB",
            "Châu Mỹ": "Trụ sở & ĐNB",
            "Châu Úc": "Trụ sở & ĐNB",
            "Châu Phi": "Trụ sở & ĐNB"
        }
        
        # Safety margin thresholds by route
        self.safety_margins = {
            route: random.uniform(4, 7) for route in self.tour_routes
        }
        
        # Partner Data
        self.partners = [
                    ("Khách sạn A", "Khách sạn"), ("Khách sạn B", "Khách sạn"), ("Khách sạn C", "Khách sạn"),
                    ("Hàng không X", "Vé máy bay"), ("Hàng không Y", "Vé máy bay"), 
                    ("Vận chuyển 1", "Vận chuyển"), ("Vận chuyển 2", "Vận chuyển"),
                    ("Nhà hàng A", "Ăn uống"), ("Nhà hàng B", "Ăn uống"),
                    ("Điểm tham quan 1", "Điểm tham quan"), ("Đại lý Quốc tế 1", "Đối tác nước ngoài")
                ]

        self.service_types = ["Lưu trú", "Vé máy bay", "Vận chuyển", "Ăn uống", "Tham quan"]
    
    def generate_tour_data(self, start_date, end_date, num_tours=1500):
        """
        Generate tour booking data
        """
        tours = []
        
        # Generate customer IDs to simulate returning customers
        num_customers = int(num_tours * 0.7)  # 70% unique customers
        customer_ids = [f"KH{i:06d}" for i in range(1, num_customers + 1)]
        
        for i in range(num_tours):
            # Random booking date
            booking_date = fake.date_time_between(
                start_date=start_date,
                end_date=end_date
            ).replace(tzinfo=None)
            
            # Tour route and related info
            route = random.choice(self.tour_routes)
            business_unit = self.route_to_unit[route]
            
            # Sales channel with realistic distribution
            channel_weights = [0.35, 0.40, 0.25]  # Online, Direct, Agent
            channel = random.choices(self.sales_channels, weights=channel_weights)[0]
            
            # Segment (phân khúc) based on route and group size
            if route in ["Châu Âu", "Châu Mỹ", "Châu Úc", "Châu Phi", "Nhật Bản", "Hàn Quốc", 
                        "Trung Quốc", "Thái Lan", "Singapore - Malaysia"]:
                segment_weights = [0.25, 0.35, 0.40]
            else:
                segment_weights = [0.35, 0.55, 0.10]
            segment = random.choices(self.segments, weights=segment_weights)[0]
            
            # Number of customers (group size)
            if random.random() < 0.3:
                num_customers_in_booking = random.randint(2, 4)
            else:
                num_customers_in_booking = random.randint(5, 20)
            
            tour_capacity = random.choice([20, 25, 30, 35, 40, 45])
            
            # Price per person (depends on route)
            if route in ["Châu Âu", "Châu Mỹ"]:
                price_per_person = random.randint(45000000, 75000000)
            elif route in ["Châu Úc", "Châu Phi"]:
                price_per_person = random.randint(35000000, 60000000)
            elif route in ["Nhật Bản", "Hàn Quốc"]:
                price_per_person = random.randint(15000000, 35000000)
            elif route in ["Trung Quốc", "Thái Lan", "Singapore - Malaysia"]:
                price_per_person = random.randint(8000000, 18000000)
            else:
                price_per_person = random.randint(3000000, 12000000)
            
            revenue = price_per_person * num_customers_in_booking
            
            cost_ratio = random.uniform(0.85, 0.95)
            cost = revenue * cost_ratio
            gross_profit = revenue - cost
            gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
            
            status_weights = [0.75, 0.15, 0.10]
            status = random.choices(["Đã xác nhận", "Đã hủy", "Hoãn"], weights=status_weights)[0]
            
            if random.random() < 0.25:
                customer_id = random.choice(customer_ids[:int(len(customer_ids) * 0.3)])
            else:
                customer_id = random.choice(customer_ids)
            
            # Marketing and sales costs (OPEX)
            if channel == "Online":
                marketing_cost = revenue * random.uniform(0.02, 0.05)
            else:
                marketing_cost = revenue * random.uniform(0.01, 0.03)
            
            if channel == "Online":
                sales_cost = revenue * random.uniform(0.01, 0.02)
            elif channel == "Trực tiếp VPGD":
                sales_cost = revenue * random.uniform(0.02, 0.04)
            else:
                sales_cost = revenue * random.uniform(0.05, 0.08)
            
            opex = marketing_cost + sales_cost
            
            # Partner and Service Data (cho Tab 3)
            partner_name, partner_type = random.choice(self.partners)
            service_type = partner_type # Tạm thời dùng loại đối tác làm loại dịch vụ
            contract_status = random.choices(["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"], weights=[0.8, 0.1, 0.1])[0]
            payment_status = random.choices(["Trả trước", "Trả sau", "Chưa thanh toán"], weights=[0.6, 0.3, 0.1])[0]
            feedback_ratio = random.uniform(0.7, 0.95)
            service_cost = cost * random.uniform(0.8, 1.2)

            tours.append({
                'booking_id': f"BK{i+1:06d}",
                # ... (Giữ nguyên các trường booking_id đến opex) ...
                'customer_id': customer_id, 'booking_date': booking_date, 'route': route, 
                'business_unit': business_unit, 'sales_channel': channel, 'segment': segment, 
                'num_customers': num_customers_in_booking, 'tour_capacity': tour_capacity, 
                'price_per_person': price_per_person, 'revenue': revenue, 'cost': cost, 
                'gross_profit': gross_profit, 'gross_profit_margin': gross_profit_margin, 
                'status': status, 'marketing_cost': marketing_cost, 'sales_cost': sales_cost, 'opex': opex,
                
                # Thêm trường Đối tác MỚI
                'partner': partner_name,
                'partner_type': partner_type, # <--- TRƯỜNG MỚI ĐỂ PHÂN LOẠI
                'service_type': service_type,
                'contract_status': contract_status,
                'payment_status': payment_status,
                'feedback_ratio': feedback_ratio,
                'service_cost': service_cost 
            })
        
        return pd.DataFrame(tours)
    
    def generate_plan_data(self, year, month=None):
        """
        Generate monthly or yearly plan data
        """
        plans = []
        
        if month:
            periods = [(year, month)]
        else:
            periods = [(year, m) for m in range(1, 13)]
        
        for year, month in periods:
            for business_unit in self.business_units:
                # Get routes for this business unit
                unit_routes = [r for r, u in self.route_to_unit.items() if u == business_unit]
                
                for route in unit_routes:
                    for segment in self.segments:
                        # Seasonality factor
                        if month in [1, 2, 4, 7, 8, 12]:
                            seasonality = random.uniform(1.2, 1.5)
                        elif month in [3, 9, 10]:
                            seasonality = random.uniform(0.9, 1.1)
                        else:
                            seasonality = random.uniform(0.7, 0.9)
                        
                        # Base plan values (distributed by segment)
                        base_customers = random.randint(5, 20)
                        planned_customers = int(base_customers * seasonality)
                        
                        # Revenue plan
                        if route in ["Châu Âu", "Châu Mỹ"]:
                            avg_price = random.randint(30000000, 50000000)
                        elif route in ["Châu Úc", "Châu Phi"]:
                            avg_price = random.randint(25000000, 40000000)
                        elif route in ["Nhật Bản", "Hàn Quốc"]:
                            avg_price = random.randint(15000000, 25000000)
                        elif route in ["Trung Quốc", "Thái Lan", "Singapore - Malaysia"]:
                            avg_price = random.randint(8000000, 12000000)
                        else:
                            avg_price = random.randint(3000000, 7000000)
                        
                        planned_revenue = planned_customers * avg_price
                        
                        # Gross profit plan (20% margin)
                        planned_gross_profit = planned_revenue * 0.20
                        
                        plans.append({
                            'year': year,
                            'month': month,
                            'business_unit': business_unit,
                            'route': route,
                            'segment': segment,
                            'planned_customers': planned_customers,
                            'planned_revenue': planned_revenue,
                            'planned_gross_profit': planned_gross_profit
                        })
        
        return pd.DataFrame(plans)
    
    def generate_historical_data(self, current_date, lookback_years=2):
        """
        Generate historical data for year-over-year comparison
        """
        all_data = []
        
        for year_offset in range(lookback_years + 1):
            year_start = datetime(current_date.year - year_offset, 1, 1)
            year_end = datetime(current_date.year - year_offset, 12, 31)
            
            # Generate tours for this year
            num_tours = random.randint(400, 600)
            yearly_data = self.generate_tour_data(year_start, year_end, num_tours)
            all_data.append(yearly_data)
        
        return pd.concat(all_data, ignore_index=True)


def load_or_generate_data():
    """
    Load or generate data for the dashboard
    """
    generator = VietravelDataGenerator()
    
    # Current date
    current_date = datetime.now()
    current_year = current_date.year
    
    # Generate current year data
    year_start = datetime(current_year, 1, 1)
    year_end = current_date
    tours_df = generator.generate_tour_data(year_start, year_end, num_tours=1500)
    
    # Generate plan data for current year
    plans_df = generator.generate_plan_data(current_year)
    
    # Generate historical data (including last year for YoY comparison)
    historical_df = generator.generate_historical_data(current_date, lookback_years=2)
    
    return tours_df, plans_df, historical_df