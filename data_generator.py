"""
Data generator for Vietravel Business Intelligence Dashboard
Generates realistic mock data for tour sales, customers, and operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

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
            "Hà Nội - Hạ Long - Sapa",
            "Hồ Chí Minh - Đà Lạt - Nha Trang",
            "Đà Nẵng - Hội An - Huế",
            "Phú Quốc - Nam Du",
            "Cần Thơ - Miền Tây",
            "Quy Nhơn - Phú Yên",
            "Ninh Bình - Tam Cốc",
            "Hà Nội - Mai Châu - Mộc Châu",
            "Bangkok - Pattaya",
            "Singapore - Malaysia",
            "Seoul - Nami - Everland",
            "Tokyo - Osaka - Kyoto",
            "Bali - Indonesia",
            "Phuket - Krabi",
            "Paris - Thụy Sĩ - Ý"
        ]
        
        # Business units (đơn vị kinh doanh)
        self.business_units = [
            "Miền Bắc",
            "Miền Trung",
            "Miền Nam",
            "Quốc Tế Châu Á",
            "Quốc Tế Châu Âu"
        ]
        
        # Sales channels (kênh bán)
        self.sales_channels = [
            "Online",
            "Trực tiếp VPGD",
            "Đại lý"
        ]
        
        # Map routes to business units
        self.route_to_unit = {
            "Hà Nội - Hạ Long - Sapa": "Miền Bắc",
            "Hồ Chí Minh - Đà Lạt - Nha Trang": "Miền Nam",
            "Đà Nẵng - Hội An - Huế": "Miền Trung",
            "Phú Quốc - Nam Du": "Miền Nam",
            "Cần Thơ - Miền Tây": "Miền Nam",
            "Quy Nhơn - Phú Yên": "Miền Trung",
            "Ninh Bình - Tam Cốc": "Miền Bắc",
            "Hà Nội - Mai Châu - Mộc Châu": "Miền Bắc",
            "Bangkok - Pattaya": "Quốc Tế Châu Á",
            "Singapore - Malaysia": "Quốc Tế Châu Á",
            "Seoul - Nami - Everland": "Quốc Tế Châu Á",
            "Tokyo - Osaka - Kyoto": "Quốc Tế Châu Á",
            "Bali - Indonesia": "Quốc Tế Châu Á",
            "Phuket - Krabi": "Quốc Tế Châu Á",
            "Paris - Thụy Sĩ - Ý": "Quốc Tế Châu Âu"
        }
        
        # Safety margin thresholds by route
        self.safety_margins = {
            route: random.uniform(4, 7) for route in self.tour_routes
        }
    
    def generate_tour_data(self, start_date, end_date, num_tours=500):
        """
        Generate tour booking data
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            num_tours: Number of tour bookings to generate
            
        Returns:
            DataFrame with tour booking data
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
            )
            
            # Tour route and related info
            route = random.choice(self.tour_routes)
            business_unit = self.route_to_unit[route]
            
            # Sales channel with realistic distribution
            channel_weights = [0.35, 0.40, 0.25]  # Online, Direct, Agent
            channel = random.choices(self.sales_channels, weights=channel_weights)[0]
            
            # Number of customers (group size)
            if random.random() < 0.3:  # 30% individual/couple
                num_customers_in_booking = random.randint(1, 2)
            else:  # 70% groups
                num_customers_in_booking = random.randint(3, 15)
            
            # Tour capacity
            tour_capacity = random.choice([20, 25, 30, 35, 40, 45])
            
            # Price per person (depends on route)
            if "Châu Âu" in business_unit:
                price_per_person = random.randint(45000000, 75000000)
            elif "Châu Á" in business_unit and route not in ["Bangkok - Pattaya", "Phuket - Krabi"]:
                price_per_person = random.randint(15000000, 35000000)
            else:
                price_per_person = random.randint(3000000, 12000000)
            
            # Revenue
            revenue = price_per_person * num_customers_in_booking
            
            # Cost (to calculate gross profit)
            # Gross margin typically 15-30% for tours
            cost_ratio = random.uniform(0.70, 0.85)
            cost = revenue * cost_ratio
            gross_profit = revenue - cost
            gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
            
            # Status (booking status)
            status_weights = [0.75, 0.15, 0.10]  # Confirmed, Cancelled, Postponed
            status = random.choices(["Đã xác nhận", "Đã hủy", "Hoãn"], weights=status_weights)[0]
            
            # Customer ID (some customers book multiple times)
            if random.random() < 0.25:  # 25% chance of returning customer
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
        
        Args:
            year: Year for the plan
            month: Optional month (1-12), if None generates yearly plan
            
        Returns:
            DataFrame with plan data by business unit and route
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
                    if month in [1, 2, 4, 7, 8, 12]:  # Peak months
                        seasonality = random.uniform(1.2, 1.5)
                    elif month in [3, 9, 10]:  # Medium months
                        seasonality = random.uniform(0.9, 1.1)
                    else:  # Low months
                        seasonality = random.uniform(0.7, 0.9)
                    
                    # Base plan values
                    base_customers = random.randint(50, 200)
                    planned_customers = int(base_customers * seasonality)
                    
                    # Revenue plan
                    if "Châu Âu" in business_unit:
                        avg_price = random.randint(50000000, 70000000)
                    elif "Châu Á" in business_unit and route not in ["Bangkok - Pattaya", "Phuket - Krabi"]:
                        avg_price = random.randint(20000000, 30000000)
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
        
        Args:
            current_date: Current date reference
            lookback_years: Number of years to look back
            
        Returns:
            DataFrame with historical tour data
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
    
    Returns:
        tuple: (tours_df, plans_df, historical_df)
    """
    generator = VietravelDataGenerator()
    
    # Current date
    current_date = datetime.now()
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
