# ============================================
# CODE 3: GENERATE SYNTHETIC BATHYMETRY (CORRECTED)
# ============================================
# This generates underwater topography on a 100×100 grid
# with realistic features (continental slope, abyssal hills)

import numpy as np
import pandas as pd  # <-- ADDED THIS
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DEFINE GRID PARAMETERS
# ============================================

# Indian Ocean region
LAT_MIN, LAT_MAX = 5, 25
LON_MIN, LON_MAX = 65, 85

# Grid resolution
GRID_SIZE = 100

# Bathymetry statistics (from GEBCO)
# Indian Ocean typical depths: 0m to -200m in coastal regions
DEPTH_MEAN = -80   # Mean depth (meters)
DEPTH_STD = 40     # Standard deviation

# ============================================
# 2. DEFINE GENERATION FUNCTIONS
# ============================================

def generate_synthetic_bathymetry(grid_size=100, random_seed=42):
    """
    Generate synthetic bathymetry using K-L expansion method.
    
    Features:
    - Continental slope (shallow north, deeper south)
    - Abyssal hills (Von Karman correlation)
    - Coastline influence
    - Realistic seabed roughness
    
    Returns:
        depth_grid: 2D array of depths (meters)
        X, Y: Coordinate grids
    """
    np.random.seed(random_seed)
    
    # Create coordinate grids
    x = np.linspace(LON_MIN, LON_MAX, grid_size)
    y = np.linspace(LAT_MIN, LAT_MAX, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # ========================================
    # COMPONENT 1: Continental Slope
    # ========================================
    # Depth increases from north (shallow) to south (deeper)
    # Based on real Indian Ocean bathymetry
    
    # Base slope: -10m per degree latitude
    slope_factor = -10 * (Y - LAT_MIN)
    
    # Coastline proximity: shallower near coast (west)
    coast_factor = -30 * np.exp(-((X - LON_MIN) / 5)**2)
    
    # Continental shelf (flat at ~-30m)
    shelf = -30 * np.exp(-((X - LON_MIN) / 3)**2) * (Y < 12)
    
    # ========================================
    # COMPONENT 2: Abyssal Hills (Von Karman)
    # ========================================
    # Using spectral method for realistic seabed roughness
    
    kx = np.fft.fftfreq(grid_size)
    ky = np.fft.fftfreq(grid_size)
    KX, KY = np.meshgrid(kx, ky)
    
    # Von Karman spectrum
    spectrum = 1 / (1 + (KX**2 + KY**2)**1.5)
    spectrum[0, 0] = 0  # Remove zero frequency
    
    # Random phase for realistic roughness
    random_phase = np.exp(2j * np.pi * np.random.rand(grid_size, grid_size))
    hills = np.fft.ifft2(np.sqrt(spectrum) * random_phase).real
    
    # Scale abyssal hills amplitude
    hills = hills * 25  # ±25m variation
    
    # ========================================
    # COMPONENT 3: Submarine Canyons
    # ========================================
    # Create 2-3 canyon features
    canyons = np.zeros((grid_size, grid_size))
    
    canyon_locations = [
        (10, 70, 3, 1),   # (lat, lon, width, depth_factor)
        (15, 75, 2, 0.8),
        (20, 80, 4, 0.6),
    ]
    
    for lat, lon, width, depth_factor in canyon_locations:
        canyon = -20 * depth_factor * np.exp(-((X - lon)**2 + (Y - lat)**2) / (width**2))
        canyons = canyons + canyon
    
    # ========================================
    # COMPONENT 4: Seamounts
    # ========================================
    # Small isolated underwater mountains
    seamounts = np.zeros((grid_size, grid_size))
    
    seamount_locations = [
        (8, 72, 50),   # (lat, lon, height_factor)
        (18, 68, 40),
        (22, 82, 30),
    ]
    
    for lat, lon, height in seamount_locations:
        seamount = -height * np.exp(-((X - lon)**2 + (Y - lat)**2) / 1.5)
        seamounts = seamounts + seamount
    
    # ========================================
    # COMBINE ALL COMPONENTS
    # ========================================
    depth = (slope_factor + coast_factor + shelf + hills + canyons + seamounts)
    
    # ========================================
    # SCALE TO REALISTIC DEPTHS
    # ========================================
    # Clip to realistic ranges
    depth = np.clip(depth, -200, -5)
    
    # Match real bathymetry statistics
    depth = (depth - depth.mean()) / depth.std() * DEPTH_STD + DEPTH_MEAN
    depth = np.clip(depth, -200, -5)
    
    # ========================================
    # ADD REALISTIC NOISE
    # ========================================
    # Small-scale roughness
    roughness = np.random.normal(0, 2, (grid_size, grid_size))
    depth = depth + roughness
    
    # Coastline: shallow water near coast
    depth = depth * (1 - 0.3 * np.exp(-((X - LON_MIN) / 2)**2))
    depth = np.clip(depth, -200, -5)
    
    return depth, X, Y

# ============================================
# 3. GENERATE DATA
# ============================================

print("="*60)
print("GENERATING SYNTHETIC BATHYMETRY")
print("="*60)

# Generate 100×100 grid
bathymetry_synthetic, X, Y = generate_synthetic_bathymetry(grid_size=GRID_SIZE, random_seed=42)

# ============================================
# 4. STATISTICS AND VALIDATION
# ============================================

print(f"\n✅ Generated {bathymetry_synthetic.shape[0]}×{bathymetry_synthetic.shape[1]} grid")
print(f"   Total points: {bathymetry_synthetic.size}")

print(f"\n📊 Bathymetry Statistics:")
print(f"   Mean: {bathymetry_synthetic.mean():.1f} m")
print(f"   Std: {bathymetry_synthetic.std():.1f} m")
print(f"   Min: {bathymetry_synthetic.min():.1f} m")
print(f"   Max: {bathymetry_synthetic.max():.1f} m")

# Depth distribution
deep_water = (bathymetry_synthetic < -100).sum()
shallow_water = (bathymetry_synthetic >= -100).sum()
print(f"\n📊 Depth Distribution:")
print(f"   Deep water (>100m): {deep_water} points ({deep_water/bathymetry_synthetic.size*100:.1f}%)")
print(f"   Shallow water (<100m): {shallow_water} points ({shallow_water/bathymetry_synthetic.size*100:.1f}%)")

# ============================================
# 5. SAVE DATA
# ============================================

# Save as NetCDF
ds_bathy = xr.Dataset({
    'elevation': (['y', 'x'], bathymetry_synthetic),
    'latitude': (['y', 'x'], Y),
    'longitude': (['y', 'x'], X)
})

ds_bathy.to_netcdf('synthetic_bathymetry.nc')
print(f"\n💾 Saved to: synthetic_bathymetry.nc")
print(f"   File size: ~{bathymetry_synthetic.nbytes / 1e6:.1f} MB")

# ============================================
# FIXED: Save as CSV with pandas
# ============================================
bathy_flat = pd.DataFrame({
    'latitude': Y.flatten(),
    'longitude': X.flatten(),
    'depth': bathymetry_synthetic.flatten()
})
bathy_flat.to_csv('synthetic_bathymetry.csv', index=False)
print(f"   Also saved as CSV: synthetic_bathymetry.csv")

# ============================================
# 6. QUICK PLOT (Optional)
# ============================================

try:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(X, Y, bathymetry_synthetic, levels=30, cmap='Blues_r')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Synthetic Bathymetry\n(Underwater Topography)')
    plt.colorbar(im, label='Depth (m)')
    plt.tight_layout()
    plt.savefig('synthetic_bathymetry_plot.png', dpi=150)
    print(f"\n📊 Plot saved: synthetic_bathymetry_plot.png")
    plt.close()
except Exception as e:
    print(f"\n⚠️ Could not generate plot: {e}")

print("\n" + "="*60)
print("✅ SYNTHETIC BATHYMETRY GENERATION COMPLETE")
print("="*60)