# dataset_generator_travel_time.py
"""
School Bus Route Travel Time Prediction Dataset Generator
Optimized for regression ML tasks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy.spatial.distance import euclidean

np.random.seed(42)
random.seed(42)

class TravelTimeDatasetGenerator:
    """
    Generate synthetic dataset for bus route travel time prediction
    Target variable: total_time_minutes
    """
    
    def __init__(self, n_records=2000):
        self.n_records = n_records
        # City boundaries (Kuala Lumpur area)
        self.lat_min, self.lat_max = 3.05, 3.30
        self.lon_min, self.lon_max = 101.60, 101.80
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def calculate_base_travel_time(self, distance, road_type, num_stops):
        """Calculate base travel time with realistic factors"""
        # Base speed by road type (km/h)
        speed_map = {
            'highway': 70,
            'main_road': 45,
            'local_road': 30,
            'residential': 25
        }
        
        base_speed = speed_map.get(road_type, 35)
        
        # Travel time in minutes
        travel_time = (distance / base_speed) * 60
        
        # Add stop time (avg 45 seconds per stop for boarding)
        stop_time = num_stops * 0.75
        
        return travel_time + stop_time
    
    def apply_traffic_factor(self, base_time, traffic, hour, day_of_week):
        """Apply traffic impact on travel time"""
        traffic_multipliers = {
            'free_flow': 1.0,
            'light': 1.15,
            'moderate': 1.45,
            'heavy': 2.0,
            'congested': 2.5
        }
        
        multiplier = traffic_multipliers.get(traffic, 1.2)
        
        # Rush hour penalty (6-8 AM, 4-7 PM)
        if hour in [6, 7] or hour in [16, 17, 18]:
            multiplier *= 1.2
        
        # Weekday penalty
        if day_of_week < 5:  # Monday-Friday
            multiplier *= 1.1
        
        return base_time * multiplier
    
    def apply_weather_factor(self, base_time, weather, season):
        """Apply weather impact"""
        weather_multipliers = {
            'clear': 1.0,
            'cloudy': 1.05,
            'light_rain': 1.2,
            'heavy_rain': 1.5,
            'fog': 1.35,
            'storm': 1.8
        }
        
        multiplier = weather_multipliers.get(weather, 1.0)
        
        # Monsoon season effect
        if season in ['winter', 'monsoon']:
            multiplier *= 1.1
        
        return base_time * multiplier
    
    def apply_driver_factor(self, base_time, experience, age):
        """Apply driver experience and age factors"""
        # Experienced drivers are faster
        if experience > 10:
            exp_factor = 0.95
        elif experience > 5:
            exp_factor = 1.0
        else:
            exp_factor = 1.1
        
        # Older drivers slightly more cautious
        if age > 55:
            age_factor = 1.05
        elif age > 45:
            age_factor = 1.0
        else:
            age_factor = 0.98
        
        return base_time * exp_factor * age_factor
    
    def apply_bus_factor(self, base_time, bus_age, maintenance_score):
        """Apply bus condition factors"""
        # Older buses are slower
        age_penalty = 1 + (bus_age * 0.01)  # 1% per year
        
        # Poor maintenance adds delays
        maintenance_factor = 2 - (maintenance_score / 5)  # Score 1-5
        
        return base_time * age_penalty * (maintenance_factor * 0.1 + 0.95)
    
    def add_random_incidents(self, base_time, route_complexity):
        """Add random delays due to incidents"""
        # Higher complexity = more chance of delays
        incident_probability = route_complexity / 200
        
        if np.random.random() < incident_probability:
            # Random incident delay (1-15 minutes)
            incident_delay = np.random.exponential(5)
            return base_time + min(incident_delay, 15)
        
        return base_time
    
    def generate_route_records(self):
        """Generate complete dataset"""
        records = []
        
        schools = [
            {'id': 'SCH_001', 'lat': 3.15, 'lon': 101.68, 'name': 'International School A'},
            {'id': 'SCH_002', 'lat': 3.12, 'lon': 101.70, 'name': 'National School B'},
            {'id': 'SCH_003', 'lat': 3.20, 'lon': 101.65, 'name': 'Private School C'}
        ]
        
        for route_id in range(1, self.n_records + 1):
            # Date and time features
            days_offset = np.random.randint(0, 365)
            date = datetime(2024, 1, 1) + timedelta(days=days_offset)
            day_of_week = date.weekday()
            month = date.month
            
            # Season
            if month in [12, 1, 2]:
                season = 'winter'
            elif month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            else:
                season = 'monsoon'
            
            # Time of day (morning routes: 5:30 AM - 7:30 AM)
            hour = np.random.choice([5, 6, 7], p=[0.2, 0.5, 0.3])
            minute = np.random.randint(0, 60)
            
            # School selection
            school = random.choice(schools)
            
            # Route characteristics
            num_stops = np.random.randint(5, 20)  # 5-20 stops per route
            
            # Generate stop locations
            stops_lat = np.random.uniform(self.lat_min, self.lat_max, num_stops)
            stops_lon = np.random.uniform(self.lon_min, self.lon_max, num_stops)
            
            # Calculate total distance
            total_distance = 0
            for i in range(num_stops - 1):
                total_distance += self.haversine_distance(
                    stops_lat[i], stops_lon[i],
                    stops_lat[i+1], stops_lon[i+1]
                )
            # Add distance from last stop to school
            total_distance += self.haversine_distance(
                stops_lat[-1], stops_lon[-1],
                school['lat'], school['lon']
            )
            
            # Student count
            total_students = sum(np.random.poisson(6) + 1 for _ in range(num_stops))
            total_students = min(total_students, 50)  # Bus capacity limit
            
            # Road characteristics
            road_type = np.random.choice(
                ['highway', 'main_road', 'local_road', 'residential'],
                p=[0.15, 0.35, 0.35, 0.15]
            )
            
            # Calculate average road quality (1-5)
            road_quality = np.random.uniform(2.5, 5.0)
            
            # Traffic conditions (time-dependent)
            if hour >= 6:
                traffic_probs = [0.1, 0.15, 0.35, 0.30, 0.10]
            else:
                traffic_probs = [0.4, 0.35, 0.20, 0.05, 0.0]
            
            traffic = np.random.choice(
                ['free_flow', 'light', 'moderate', 'heavy', 'congested'],
                p=traffic_probs
            )
            
            # Weather conditions (season-dependent)
            if season == 'monsoon':
                weather_probs = [0.3, 0.2, 0.25, 0.15, 0.05, 0.05]
            else:
                weather_probs = [0.5, 0.3, 0.10, 0.05, 0.03, 0.02]
            
            weather = np.random.choice(
                ['clear', 'cloudy', 'light_rain', 'heavy_rain', 'fog', 'storm'],
                p=weather_probs
            )
            
            # Temperature (affects AC usage, hence speed)
            if season == 'summer':
                temperature = np.random.uniform(28, 38)
            elif season == 'winter':
                temperature = np.random.uniform(22, 30)
            else:
                temperature = np.random.uniform(24, 33)
            
            # Driver characteristics
            driver_id = f"DRV_{np.random.randint(1, 51):03d}"
            driver_experience = np.random.randint(1, 25)  # years
            driver_age = np.random.randint(25, 65)
            
            # Bus characteristics
            bus_id = f"BUS_{np.random.randint(1, 31):03d}"
            bus_age = np.random.randint(0, 15)  # years
            bus_capacity = 50
            maintenance_score = np.random.uniform(2, 5)  # 1-5 scale
            
            # Route complexity score (based on multiple factors)
            route_complexity = (
                num_stops * 3 +
                (total_distance * 2) +
                (1 if traffic in ['heavy', 'congested'] else 0) * 20 +
                (1 if weather in ['heavy_rain', 'storm', 'fog'] else 0) * 15 +
                (1 if road_type == 'residential' else 0) * 10 +
                np.random.randint(0, 20)
            )
            route_complexity = min(route_complexity, 100)
            
            # Historical performance (some routes are historically slower)
            route_zone = f"Zone_{chr(65 + route_id % 5)}"
            zone_performance = {'Zone_A': 0.95, 'Zone_B': 1.0, 'Zone_C': 1.1, 
                              'Zone_D': 1.05, 'Zone_E': 0.98}
            zone_factor = zone_performance.get(route_zone, 1.0)
            
            # --- CALCULATE TRAVEL TIME (TARGET VARIABLE) ---
            
            # Step 1: Base travel time
            base_time = self.calculate_base_travel_time(total_distance, road_type, num_stops)
            
            # Step 2: Apply traffic factor
            time_with_traffic = self.apply_traffic_factor(base_time, traffic, hour, day_of_week)
            
            # Step 3: Apply weather factor
            time_with_weather = self.apply_weather_factor(time_with_traffic, weather, season)
            
            # Step 4: Apply driver factor
            time_with_driver = self.apply_driver_factor(time_with_weather, driver_experience, driver_age)
            
            # Step 5: Apply bus condition factor
            time_with_bus = self.apply_bus_factor(time_with_driver, bus_age, maintenance_score)
            
            # Step 6: Apply zone historical factor
            time_with_zone = time_with_bus * zone_factor
            
            # Step 7: Apply road quality factor
            road_quality_factor = 2 - (road_quality / 5)  # Better roads = faster
            time_with_road = time_with_zone * (0.9 + road_quality_factor * 0.1)
            
            # Step 8: Add random incidents
            final_time = self.add_random_incidents(time_with_road, route_complexity)
            
            # Step 9: Add small random noise
            final_time += np.random.normal(0, 1.5)
            
            # Ensure minimum time
            final_time = max(final_time, 10)
            
            # Calculate derived metrics
            avg_speed = (total_distance / final_time) * 60 if final_time > 0 else 0
            time_per_stop = final_time / num_stops
            time_per_km = final_time / total_distance if total_distance > 0 else 0
            
            # Peak hour flag
            is_peak_hour = 1 if hour in [6, 7] else 0
            
            # Weekend flag
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Create record
            record = {
                # Identifiers
                'route_id': f'ROUTE_{route_id:05d}',
                'school_id': school['id'],
                'school_name': school['name'],
                'bus_id': bus_id,
                'driver_id': driver_id,
                
                # Temporal features
                'date': date.strftime('%Y-%m-%d'),
                'year': date.year,
                'month': month,
                'day': date.day,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'season': season,
                'hour': hour,
                'minute': minute,
                'is_peak_hour': is_peak_hour,
                
                # Route characteristics
                'num_stops': num_stops,
                'total_distance_km': round(total_distance, 3),
                'total_students': total_students,
                'students_per_stop': round(total_students / num_stops, 2),
                'avg_distance_per_stop': round(total_distance / num_stops, 3),
                'route_zone': route_zone,
                'route_complexity_score': route_complexity,
                
                # Road and traffic
                'road_type': road_type,
                'road_quality_score': round(road_quality, 2),
                'traffic_conditions': traffic,
                
                # Weather
                'weather': weather,
                'temperature_celsius': round(temperature, 1),
                
                # Driver factors
                'driver_experience_years': driver_experience,
                'driver_age': driver_age,
                
                # Bus factors
                'bus_age_years': bus_age,
                'bus_capacity': bus_capacity,
                'bus_maintenance_score': round(maintenance_score, 2),
                'occupancy_rate': round(total_students / bus_capacity, 2),
                
                # Geographic
                'school_lat': school['lat'],
                'school_lon': school['lon'],
                'avg_stop_lat': round(np.mean(stops_lat), 4),
                'avg_stop_lon': round(np.mean(stops_lon), 4),
                
                # TARGET VARIABLE
                'total_time_minutes': round(final_time, 2),
                
                # Derived metrics (for analysis)
                'avg_speed_kmh': round(avg_speed, 2),
                'time_per_stop_minutes': round(time_per_stop, 2),
                'time_per_km_minutes': round(time_per_km, 2)
            }
            
            records.append(record)
            
            if route_id % 500 == 0:
                print(f"Generated {route_id}/{self.n_records} records...")
        
        return pd.DataFrame(records)
    
    def save_dataset(self, output_path='data/bus_route_travel_time.csv'):
        """Generate and save dataset"""
        import os
        os.makedirs('data', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        print("="*70)
        print("GENERATING SCHOOL BUS TRAVEL TIME PREDICTION DATASET")
        print("="*70)
        
        df = self.generate_route_records()
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Dataset saved to: {output_path}")
        print(f"✅ Total records: {len(df)}")
        print(f"✅ Total features: {len(df.columns)}")
        print(f"\n📊 Target Variable Statistics:")
        print(f"   Mean travel time: {df['total_time_minutes'].mean():.2f} minutes")
        print(f"   Std deviation: {df['total_time_minutes'].std():.2f} minutes")
        print(f"   Min travel time: {df['total_time_minutes'].min():.2f} minutes")
        print(f"   Max travel time: {df['total_time_minutes'].max():.2f} minutes")
        print(f"   Median: {df['total_time_minutes'].median():.2f} minutes")
        
        print(f"\n📁 Feature Categories:")
        print(f"   - Temporal features: 10")
        print(f"   - Route characteristics: 8")
        print(f"   - Traffic & Road: 4")
        print(f"   - Weather: 2")
        print(f"   - Driver factors: 2")
        print(f"   - Bus factors: 4")
        print(f"   - Geographic: 4")
        
        return df

# Generate dataset
if __name__ == "__main__":
    generator = TravelTimeDatasetGenerator(n_records=2000)
    df = generator.save_dataset()
    
    print("\n" + "="*70)
    print("SAMPLE RECORDS")
    print("="*70)
    print(df.head(10))
    
    print("\n" + "="*70)
    print("DATA TYPES")
    print("="*70)
    print(df.dtypes)