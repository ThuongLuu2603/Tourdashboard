"""
Data generator for Vietravel Business Intelligence Dashboard
Generates realistic mock data for tour sales, customers, and operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import pytz # Import pytz

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
    
    def generate_tour_data(self, start_date, end_date, num_tours=500):
        """
        Generate tour booking data
        """
        tours = []
        
        # Generate customer IDs to simulate returning customers
        num_customers = int(num_tours * 0.7)
        customer_ids = [f"KH{i:06d}" for i in range(1, num_customers + 1)]
        
        for i in range(num_tours):
            # Random booking date
            booking_date = fake.date_time_between(
                start_date=start_date,
                end_date=end_date
            ).replace(tzinfo=None) # ĐÃ SỬA: Loại bỏ thông tin múi giờ
            
            # Tour route and related info
            route = random.choice(self.tour_routes)
            business_unit = self.route_to_unit[route]
            
            # Sales channel with realistic distribution
            channel_weights = [0.35, 0.40, 0.25]
            channel = random.choices(self.sales_channels, weights=channel_weights)[0]
            
            # Number of customers (group size)
            if random.random() < 0.3:
                num_customers_in_booking = random.randint(1, 2)
            else:
                num_customers_in_booking = random.randint(3, 15)
            
            # Tour capacity
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
            
            # Revenue
            revenue = price_per_person * num_customers_in_booking
            
            # Cost (to calculate gross profit)
            cost_ratio = random.uniform(0.85, 0.95)
            cost = revenue * cost_ratio
            gross_profit = revenue - cost
            gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
            
            # Status (booking status)
            status_weights = [0.75, 0.15, 0.10]
            status = random.choices(["Đã xác nhận", "Đã hủy", "Hoãn"], weights=status_weights)[0]
            
            # Customer ID (some customers book multiple times)
            if random.random() < 0.25:
                customer_id = random.choice(customer_ids[:int(len(customer_ids) * 0.3)])
            else:
                customer_id = random.choice(customer_ids)
            
            tours.append({
                'booking_id': f"BK{i+1:06d}",
                'customer_id': customer_id,
                'booking_date': booking_date,
                'route': route,
                'business_unit': business_unit,
                'sales_channel': channel,
                'num_customers': num_customers_in_booking,
                'tour_capacity': tour_capacity,
                'price_per_person': price_per_person,
                'revenue': revenue,
                'cost': cost,
                'gross_profit': gross_profit,
                'gross_profit_margin': gross_profit_margin,
                'status': status
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
                    # Seasonality factor
                    if month in [1, 2, 4, 7, 8, 12]:
                        seasonality = random.uniform(1.2, 1.5)
                    elif month in [3, 9, 10]:
                        seasonality = random.uniform(0.9, 1.1)
                    else:
                        seasonality = random.uniform(0.7, 0.9)
                    
                    # Base plan values
                    base_customers = random.randint(50, 200)
                    planned_customers = int(base_customers * seasonality)
                    
                    # Revenue plan
                    if route in ["Châu Âu", "Châu Mỹ"]:
                        avg_price = random.randint(50000000, 70000000)
                    elif route in ["Châu Úc", "Châu Phi"]:
                        avg_price = random.randint(40000000, 55000000)
                    elif route in ["Nhật Bản", "Hàn Quốc"]:
                        avg_price = random.randint(20000000, 30000000)
                    elif route in ["Trung Quốc", "Thái Lan", "Singapore - Malaysia"]:
                        avg_price = random.randint(10000000, 15000000)
                    else:
                        avg_price = random.randint(5000000, 10000000)
                    
                    planned_revenue = planned_customers * avg_price
                    
                    # Gross profit plan (20% margin)
                    planned_gross_profit = planned_revenue * 0.20
                    
                    plans.append({
                        'year': year,
                        'month': month,
                        'business_unit': business_unit,
                        'route': route,
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
    current_date = datetime.now().replace(tzinfo=None) # ĐÃ SỬA: Đảm bảo naive datetime
    current_year = current_date.year
    
    # Generate current year data
    year_start = datetime(current_year, 1, 1)
    year_end = current_date
    tours_df = generator.generate_tour_data(year_start, year_end, num_tours=500)
    
    # Generate plan data for current year
    plans_df = generator.generate_plan_data(current_year)
    
    # Generate historical data (including last year for YoY comparison)
    historical_df = generator.generate_historical_data(current_date, lookback_years=2)
    
    return tours_df, plans_df, historical_df