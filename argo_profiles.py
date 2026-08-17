# ============================================
# CODE 1: GENERATE SYNTHETIC ARGO PROFILES
# ============================================
# This generates temperature and salinity profiles at 8 depth levels
# for 1000 synthetic floats (1000 profiles × 8 depths = 8,000 records)

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DEFINE REALISTIC OCEAN PARAMETERS
# ============================================

# Depth levels (same as real Argo float measurements)
DEPTHS = [0, 10, 25, 50, 75, 100, 150, 200]

# Indian Ocean region (Arabian Sea & Bay of Bengal)
LAT_MIN, LAT_MAX = 5, 25   # 5°N to 25°N
LON_MIN, LON_MAX = 65, 85  # 65°E to 85°E

# Real Argo statistics (from INCOIS data)
# Temperature profile: warm surface (28°C), cold deep (8°C)
# Salinity profile: 35 PSU surface, 34.5 PSU deep

# ============================================
# 2. DEFINE GENERATION FUNCTIONS
# ============================================

def thermocline_temp(depth, surface_temp=28, deep_temp=8, thermocline_depth=50):
    """
    Generate realistic temperature profile with thermocline.
    Uses exponential decay for smooth temperature gradient.
    """
    return surface_temp - (surface_temp - deep_temp) * (1 - np.exp(-depth / thermocline_depth))

def halocline_sal(depth, surface_sal=35.0, deep_sal=34.5, halocline_depth=30):
    """
    Generate realistic salinity profile with halocline.
    """
    return surface_sal - (surface_sal - deep_sal) * (1 - np.exp(-depth / halocline_depth))

def add_ocean_variability(value, depth, variability=0.5):
    """
    Add realistic ocean variability (internal waves, eddies, seasonal)
    """
    # Internal wave signature (frequency increases with depth)
    wave_signal = 0.3 * np.sin(depth / 10) * np.random.normal(0, 0.5)
    
    # Seasonal signal (simplified)
    seasonal = 0.5 * np.sin(2 * np.pi * depth / 100)
    
    # Random noise (sensor accuracy: ±0.002°C for temp, ±0.005 for salinity)
    sensor_noise = np.random.normal(0, 0.002 if 'temp' in str(value) else 0.005)
    
    return value + wave_signal + seasonal + sensor_noise

# ============================================
# 3. GENERATE SYNTHETIC PROFILES
# ============================================

def generate_synthetic_argo(n_profiles=1000, random_seed=42):
    """
    Generate synthetic Argo profiles with realistic T/S relationships.
    
    Parameters:
        n_profiles: Number of synthetic profiles to generate
        random_seed: For reproducibility
    
    Returns:
        DataFrame with synthetic Argo data
    """
    np.random.seed(random_seed)
    
    synthetic_data = []
    
    # Realistic TS correlation (negative correlation in thermocline)
    # From real Argo data: correlation ~ -0.3 to -0.7
    ts_correlation = -0.5
    
    for i in range(n_profiles):
        # Random location within Indian Ocean region
        latitude = np.random.uniform(LAT_MIN, LAT_MAX)
        longitude = np.random.uniform(LON_MIN, LON_MAX)
        
        # Random date within last 6 months
        date = pd.Timestamp('2026-01-01') + pd.Timedelta(days=np.random.randint(0, 180))
        
        # Generate profile with some variability
        profile_temp = []
        profile_sal = []
        
        for depth in DEPTHS:
            # Base temperature and salinity
            temp_base = thermocline_temp(depth)
            sal_base = halocline_sal(depth)
            
            # Add regional variability
            region_factor = 0.5 * np.sin(2 * np.pi * latitude / 20) * np.cos(2 * np.pi * longitude / 30)
            
            # Add vertical variability (internal waves)
            wave_factor = 0.3 * np.sin(depth / 15 + i) * np.random.normal(0, 0.3)
            
            # Final temperature with realistic noise
            temp = temp_base + region_factor + wave_factor + np.random.normal(0, 0.15)
            
            # Final salinity (correlated with temperature)
            # When temperature decreases, salinity decreases (in thermocline)
            sal = sal_base + ts_correlation * (temp - temp_base) + np.random.normal(0, 0.05)
            
            # Clip to realistic ranges
            temp = np.clip(temp, 5, 30)
            sal = np.clip(sal, 30, 37)
            
            profile_temp.append(temp)
            profile_sal.append(sal)
        
        # Store profile
        for depth, temp, sal in zip(DEPTHS, profile_temp, profile_sal):
            synthetic_data.append({
                'float_id': f'SYN_{i:04d}',
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth,
                'temperature': temp,
                'salinity': sal,
                'date': date
            })
    
    return pd.DataFrame(synthetic_data)

# ============================================
# 4. GENERATE DATA
# ============================================

print("="*60)
print("GENERATING SYNTHETIC ARGO PROFILES")
print("="*60)

# Generate 1000 profiles (1000 floats × 8 depths = 8,000 records)
synthetic_argo = generate_synthetic_argo(n_profiles=1000, random_seed=42)

# ============================================
# 5. STATISTICS AND VALIDATION
# ============================================

print(f"\n✅ Generated {len(synthetic_argo)} records")
print(f"   Unique floats: {synthetic_argo['float_id'].nunique()}")
print(f"   Depth levels: {sorted(synthetic_argo['depth'].unique())}")

# Temperature statistics by depth
print("\n📊 Temperature Statistics by Depth:")
print(synthetic_argo.groupby('depth')['temperature'].agg(['mean', 'std', 'min', 'max']).round(2))

# Salinity statistics by depth
print("\n📊 Salinity Statistics by Depth:")
print(synthetic_argo.groupby('depth')['salinity'].agg(['mean', 'std', 'min', 'max']).round(2))

# TS correlation
ts_corr = synthetic_argo['temperature'].corr(synthetic_argo['salinity'])
print(f"\n📊 Temperature-Salinity Correlation: {ts_corr:.3f}")

# Spatial distribution
print(f"\n📊 Spatial Distribution:")
print(f"   Latitude: {synthetic_argo['latitude'].min():.1f}° to {synthetic_argo['latitude'].max():.1f}°")
print(f"   Longitude: {synthetic_argo['longitude'].min():.1f}° to {synthetic_argo['longitude'].max():.1f}°")

# ============================================
# 6. SAVE DATA
# ============================================

synthetic_argo.to_csv('synthetic_argo.csv', index=False)
print(f"\n💾 Saved to: synthetic_argo.csv")
print(f"   File size: ~{len(synthetic_argo) * 0.0001:.1f} MB")

# Create a smaller sample for quick testing
synthetic_argo_sample = synthetic_argo[synthetic_argo['float_id'].isin(['SYN_0000', 'SYN_0001', 'SYN_0002'])]
synthetic_argo_sample.to_csv('synthetic_argo_sample.csv', index=False)
print(f"   Sample saved: synthetic_argo_sample.csv (3 profiles)")

print("\n" + "="*60)
print("✅ SYNTHETIC ARGO GENERATION COMPLETE")
print("="*60)