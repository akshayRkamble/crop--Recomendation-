import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class IrrigationScheduler:
    def __init__(self):
        self.water_requirements = {
            'rice': {'water_need': 1200, 'frequency': 2},  # mm per season, days between irrigation
            'maize': {'water_need': 500, 'frequency': 7},
            'cotton': {'water_need': 700, 'frequency': 5},
            'wheat': {'water_need': 450, 'frequency': 7},
            'sugarcane': {'water_need': 1500, 'frequency': 3},
            'coffee': {'water_need': 1800, 'frequency': 4},
            'banana': {'water_need': 1200, 'frequency': 3},
            'apple': {'water_need': 800, 'frequency': 5},
            'orange': {'water_need': 900, 'frequency': 4},
            'mango': {'water_need': 850, 'frequency': 5},
            # Added more crops
            'chickpea': {'water_need': 350, 'frequency': 8},
            'pigeonpeas': {'water_need': 400, 'frequency': 8},
            'mothbeans': {'water_need': 300, 'frequency': 9},
            'mungbean': {'water_need': 450, 'frequency': 7},
            'blackgram': {'water_need': 500, 'frequency': 6},
            'lentil': {'water_need': 350, 'frequency': 8},
            'pomegranate': {'water_need': 650, 'frequency': 5},
            'grapes': {'water_need': 700, 'frequency': 4},
            'watermelon': {'water_need': 800, 'frequency': 3},
            'muskmelon': {'water_need': 750, 'frequency': 4},
            'papaya': {'water_need': 900, 'frequency': 3},
            'coconut': {'water_need': 1200, 'frequency': 5},
            'jute': {'water_need': 600, 'frequency': 6},
            'kidneybeans': {'water_need': 400, 'frequency': 7},
            # Additional crops
            'soybean': {'water_need': 450, 'frequency': 7},
            'peas': {'water_need': 350, 'frequency': 6},
            'potato': {'water_need': 500, 'frequency': 5},
            'mustard': {'water_need': 400, 'frequency': 8},
            'tomato': {'water_need': 600, 'frequency': 3},
            'onion': {'water_need': 550, 'frequency': 4},
            'garlic': {'water_need': 450, 'frequency': 5},
            'turmeric': {'water_need': 1100, 'frequency': 4},
            'ginger': {'water_need': 1000, 'frequency': 4},
            'cucumber': {'water_need': 500, 'frequency': 3},
            'brinjal': {'water_need': 550, 'frequency': 4},
            'pepper': {'water_need': 600, 'frequency': 4},
            'chilli': {'water_need': 650, 'frequency': 5},
            'cauliflower': {'water_need': 500, 'frequency': 4},
            'cabbage': {'water_need': 450, 'frequency': 5}
        }
    
    def calculate_schedule(self, crop_name, area_hectares, rainfall_mm):
        """Calculate irrigation schedule based on crop water requirements"""
        if crop_name.lower() not in self.water_requirements:
            return None
            
        crop_data = self.water_requirements[crop_name.lower()]
        water_need = crop_data['water_need']
        frequency = crop_data['frequency']
        
        # Calculate water deficit considering rainfall
        water_deficit = max(0, water_need - rainfall_mm)
        irrigation_per_session = water_deficit / (30 / frequency)  # Monthly schedule
        
        # Generate schedule for next 30 days
        schedule = []
        start_date = datetime.now()
        for i in range(30):
            if i % frequency == 0:
                schedule.append({
                    'date': (start_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'water_amount': round(irrigation_per_session, 2),
                    'area': area_hectares
                })
        
        return schedule

class EconomicAnalyzer:
    def __init__(self):
        self.crop_economics = {
            'rice': {'cost_per_hectare': 20000, 'avg_yield': 5.0, 'price_per_kg': 3500},
            'maize': {'cost_per_hectare': 18000, 'avg_yield': 4.0, 'price_per_kg': 4000},
            'cotton': {'cost_per_hectare': 25000, 'avg_yield': 2.5, 'price_per_kg': 7000},
            'wheat': {'cost_per_hectare': 17000, 'avg_yield': 3.0, 'price_per_kg': 5000},
            'sugarcane': {'cost_per_hectare': 22000, 'avg_yield': 30.0, 'price_per_kg': 600},
            'coffee': {'cost_per_hectare': 30000, 'avg_yield': 1.5, 'price_per_kg': 12000},
            'banana': {'cost_per_hectare': 22000, 'avg_yield': 8.0, 'price_per_kg': 2000},
            'apple': {'cost_per_hectare': 30000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'orange': {'cost_per_hectare': 18000, 'avg_yield': 3.0, 'price_per_kg': 6000},
            'mango': {'cost_per_hectare': 20000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'chickpea': {'cost_per_hectare': 12000, 'avg_yield': 1.5, 'price_per_kg': 10000},
            'pigeonpeas': {'cost_per_hectare': 11000, 'avg_yield': 1.2, 'price_per_kg': 12000},
            'mothbeans': {'cost_per_hectare': 10000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'mungbean': {'cost_per_hectare': 12000, 'avg_yield': 1.2, 'price_per_kg': 12000},
            'blackgram': {'cost_per_hectare': 12000, 'avg_yield': 1.2, 'price_per_kg': 12000},
            'lentil': {'cost_per_hectare': 13000, 'avg_yield': 1.3, 'price_per_kg': 12000},
            'pomegranate': {'cost_per_hectare': 35000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'grapes': {'cost_per_hectare': 40000, 'avg_yield': 2.5, 'price_per_kg': 7000},
            'watermelon': {'cost_per_hectare': 12000, 'avg_yield': 4.0, 'price_per_kg': 4000},
            'muskmelon': {'cost_per_hectare': 10000, 'avg_yield': 3.0, 'price_per_kg': 5000},
            'papaya': {'cost_per_hectare': 12000, 'avg_yield': 4.0, 'price_per_kg': 4000},
            'coconut': {'cost_per_hectare': 20000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'jute': {'cost_per_hectare': 8000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'kidneybeans': {'cost_per_hectare': 11000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'soybean': {'cost_per_hectare': 12000, 'avg_yield': 1.2, 'price_per_kg': 12000},
            'peas': {'cost_per_hectare': 12000, 'avg_yield': 1.2, 'price_per_kg': 12000},
            'potato': {'cost_per_hectare': 15000, 'avg_yield': 3.0, 'price_per_kg': 5000},
            'mustard': {'cost_per_hectare': 9000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'tomato': {'cost_per_hectare': 10000, 'avg_yield': 3.0, 'price_per_kg': 5000},
            'onion': {'cost_per_hectare': 10000, 'avg_yield': 3.0, 'price_per_kg': 5000},
            'garlic': {'cost_per_hectare': 11000, 'avg_yield': 2.0, 'price_per_kg': 7500},
            'turmeric': {'cost_per_hectare': 12000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'ginger': {'cost_per_hectare': 15000, 'avg_yield': 2.0, 'price_per_kg': 8000},
            'cucumber': {'cost_per_hectare': 6000, 'avg_yield': 2.0, 'price_per_kg': 7500},
            'brinjal': {'cost_per_hectare': 8000, 'avg_yield': 2.0, 'price_per_kg': 7500},
            'pepper': {'cost_per_hectare': 10000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'chilli': {'cost_per_hectare': 11000, 'avg_yield': 1.0, 'price_per_kg': 15000},
            'cauliflower': {'cost_per_hectare': 6000, 'avg_yield': 2.0, 'price_per_kg': 7500},
            'cabbage': {'cost_per_hectare': 5000, 'avg_yield': 2.0, 'price_per_kg': 7500},
        }
    
    def analyze_crop(self, crop_name, area_hectares):
        """Perform economic analysis for a given crop"""
        if crop_name.lower() not in self.crop_economics:
            return None
            
        crop_data = self.crop_economics[crop_name.lower()]
        total_cost = crop_data['cost_per_hectare'] * area_hectares
        expected_yield = crop_data['avg_yield'] * area_hectares
        revenue = expected_yield * crop_data['price_per_kg']
        profit = revenue - total_cost
        roi = (profit / total_cost) * 100
        
        return {
            'total_cost': round(total_cost, 2),
            'expected_yield': round(expected_yield, 2),
            'expected_revenue': round(revenue, 2),
            'expected_profit': round(profit, 2),
            'roi_percentage': round(roi, 2)
        }

class CropRotationPlanner:
    def __init__(self):
        self.seasonal_crops = {
            'summer': ['rice', 'cotton', 'maize', 'sugarcane', 'watermelon', 'muskmelon', 'papaya', 'banana', 'mungbean', 'cucumber', 'brinjal', 'tomato', 'chilli', 'pepper', 'onion', 'turmeric', 'ginger'],
            'winter': ['wheat', 'potato', 'mustard', 'peas', 'chickpea', 'lentil', 'apple', 'grapes', 'pomegranate', 'cauliflower', 'cabbage', 'garlic', 'onion', 'tomato', 'peas'],
            'monsoon': ['rice', 'maize', 'pulses', 'soybean', 'jute', 'blackgram', 'mothbeans', 'pigeonpeas', 'kidneybeans', 'coconut', 'coffee', 'banana', 'papaya', 'turmeric', 'ginger', 'cucumber']
        }
        
        self.rotation_benefits = {
            'legumes': ['peas', 'pulses', 'soybean', 'chickpea', 'lentil', 'mungbean', 'blackgram', 'mothbeans', 'pigeonpeas', 'kidneybeans'],  # Nitrogen fixing
            'cereals': ['rice', 'wheat', 'maize'],     # Soil structure
            'cash_crops': ['cotton', 'sugarcane', 'jute', 'banana', 'watermelon', 'muskmelon', 'papaya', 'pomegranate', 'grapes', 'apple', 'orange', 'mango', 'coconut', 'coffee', 'potato', 'tomato', 'onion', 'garlic', 'turmeric', 'ginger', 'chilli', 'pepper', 'brinjal', 'cauliflower', 'cabbage', 'cucumber']      # Economic value
        }
    
    def suggest_rotation(self, current_crop, season):
        """Suggest crop rotation based on current crop and season"""
        season = season.lower()
        if season not in self.seasonal_crops:
            return None
            
        # Find crop category
        current_category = None
        for category, crops in self.rotation_benefits.items():
            if current_crop.lower() in crops:
                current_category = category
                break
        
        # Suggest rotation based on soil benefits
        suggested_crops = []
        if current_category == 'legumes':
            # After legumes, suggest cereals or cash crops
            suggested_crops = [crop for crop in self.seasonal_crops[season] 
                            if crop in self.rotation_benefits['cereals'] + self.rotation_benefits['cash_crops']]
        elif current_category == 'cereals':
            # After cereals, suggest legumes
            suggested_crops = [crop for crop in self.seasonal_crops[season] 
                            if crop in self.rotation_benefits['legumes']]
        else:
            # After cash crops, suggest legumes or cereals
            suggested_crops = [crop for crop in self.seasonal_crops[season] 
                            if crop in self.rotation_benefits['legumes'] + self.rotation_benefits['cereals']]
        
        return {
            'current_season': season,
            'suggested_crops': suggested_crops,
            'rotation_benefits': "Improves soil health and nutrient balance"
        }
