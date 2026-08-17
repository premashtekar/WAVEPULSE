# ============================================
# CODE 4: GENERATE SYNTHETIC OCEAN CURRENTS (CORRECTED)
# ============================================
# This generates ocean surface currents (u and v components)
# on a 100×100 grid using deterministic + stochastic methods

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

# Current statistics (from OSCAR data)
U_MEAN = 0.15      # Zonal current mean (m/s)
U_STD = 0.25       # Zonal current std (m/s)
V_MEAN = 0.05      # Meridional current mean (m/s)
V_STD = 0.15       # Meridional current std (m/s)

# ============================================
# 2. DEFINE GENERATION FUNCTIONS
# ============================================

def generate_synthetic_currents(grid_size=100, random_seed=42):
    """
    Generate synthetic ocean currents using SUP model.
    
    Features:
    - Deterministic: Monsoon-driven circulation
    - Stochastic: Fat-tailed statistics (Student's t)
    - Mesoscale eddies
    - Coastal boundary effects
    
    Returns:
        u_grid: Zonal current (m/s)
        v_grid: Meridional current (m/s)
        X, Y: Coordinate grids
    """
    np.random.seed(random_seed)
    
    # Create coordinate grids
    x = np.linspace(LON_MIN, LON_MAX, grid_size)
    y = np.linspace(LAT_MIN, LAT_MAX, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # ========================================
    # COMPONENT 1: Deterministic Circulation
    # ========================================
    # Indian Ocean monsoon-driven circulation
    # Summer monsoon: clockwise in Arabian Sea, anti-clockwise in Bay of Bengal
    
    # Arabian Sea Gyre (clockwise)
    u_arabian = 0.4 * np.sin(2 * np.pi * (Y - LAT_MIN) / 10) * np.exp(-((X - 70)**2) / 20)
    v_arabian = 0.3 * np.cos(2 * np.pi * (X - LON_MIN) / 15) * np.exp(-((X - 70)**2) / 20)
    
    # Bay of Bengal Gyre (anti-clockwise)
    u_bengal = 0.3 * np.sin(2 * np.pi * (Y - LAT_MIN) / 8) * np.exp(-((X - 78)**2) / 15)
    v_bengal = -0.25 * np.cos(2 * np.pi * (X - LON_MIN) / 12) * np.exp(-((X - 78)**2) / 15)
    
    # Equatorial currents
    u_equatorial = 0.2 * np.cos(2 * np.pi * X / 10) * np.exp(-((Y - 8)**2) / 3)
    
    # ========================================
    # COMPONENT 2: Stochastic (Fat-tailed)
    # ========================================
    # Student's t-distribution for fat tails (captures extreme events)
    u_stoch = np.random.standard_t(df=3, size=(grid_size, grid_size)) * 0.08
    v_stoch = np.random.standard_t(df=3, size=(grid_size, grid_size)) * 0.06
    
    # ========================================
    # COMPONENT 3: Mesoscale Eddies
    # ========================================
    # Multiple eddies with varying sizes and strengths
    
    eddies_u = np.zeros((grid_size, grid_size))
    eddies_v = np.zeros((grid_size, grid_size))
    
    eddy_params = [
        (10, 70, 4, 0.20, 1),   # (lat, lon, radius, strength, rotation)
        (15, 75, 3, 0.15, -1),
        (20, 80, 5, 0.18, 1),
        (8, 72, 2, 0.12, -1),
        (18, 68, 3, 0.14, 1),
    ]
    
    for lat, lon, radius, strength, rotation in eddy_params:
        r2 = (X - lon)**2 + (Y - lat)**2
        eddy_mask = np.exp(-r2 / (radius**2))
        
        # Classic eddy structure: u = -∂ψ/∂y, v = ∂ψ/∂x
        # ψ = strength * exp(-r²/R²)
        eddies_u = eddies_u + rotation * strength * (-(Y - lat) / radius**2) * eddy_mask
        eddies_v = eddies_v + rotation * strength * ((X - lon) / radius**2) * eddy_mask
    
    # ========================================
    # COMPONENT 4: Coastal Boundary Effects
    # ========================================
    # Currents stronger along coast (western boundary currents)
    coast_factor = 1 + 0.5 * np.exp(-((X - LON_MIN) / 3)**2)
    
    # ========================================
    # COMPONENT 5: Seasonal Variability
    # ========================================
    # Simple seasonal modulation
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * Y / 20)
    
    # ========================================
    # COMBINE ALL COMPONENTS
    # ========================================
    u = (u_arabian + u_bengal + u_equatorial + u_stoch + eddies_u) * coast_factor * seasonal_factor
    v = (v_arabian + v_bengal + v_stoch + eddies_v) * coast_factor * seasonal_factor
    
    # ========================================
    # SCALE TO REALISTIC STATISTICS
    # ========================================
    u = (u - u.mean()) / u.std() * U_STD + U_MEAN
    v = (v - v.mean()) / v.std() * V_STD + V_MEAN
    
    # ========================================
    # ADD REALISTIC NOISE
    # ========================================
    sensor_noise_u = np.random.normal(0, 0.02, (grid_size, grid_size))
    sensor_noise_v = np.random.normal(0, 0.02, (grid_size, grid_size))
    
    u = u + sensor_noise_u
    v = v + sensor_noise_v
    
    return u, v, X, Y

# ============================================
# 3. GENERATE DATA
# ============================================

print("="*60)
print("GENERATING SYNTHETIC OCEAN CURRENTS")
print("="*60)

# Generate 100×100 grid
u_synthetic, v_synthetic, X, Y = generate_synthetic_currents(grid_size=GRID_SIZE, random_seed=42)

# ============================================
# 4. STATISTICS AND VALIDATION
# ============================================

print(f"\n✅ Generated {u_synthetic.shape[0]}×{u_synthetic.shape[1]} grid")
print(f"   Total points: {u_synthetic.size}")

print(f"\n📊 U-Current (Zonal) Statistics:")
print(f"   Mean: {u_synthetic.mean():.3f} m/s")
print(f"   Std: {u_synthetic.std():.3f} m/s")
print(f"   Min: {u_synthetic.min():.3f} m/s")
print(f"   Max: {u_synthetic.max():.3f} m/s")

print(f"\n📊 V-Current (Meridional) Statistics:")
print(f"   Mean: {v_synthetic.mean():.3f} m/s")
print(f"   Std: {v_synthetic.std():.3f} m/s")
print(f"   Min: {v_synthetic.min():.3f} m/s")
print(f"   Max: {v_synthetic.max():.3f} m/s")

# Current magnitude
current_magnitude = np.sqrt(u_synthetic**2 + v_synthetic**2)
print(f"\n📊 Current Magnitude:")
print(f"   Mean: {current_magnitude.mean():.3f} m/s")
print(f"   Std: {current_magnitude.std():.3f} m/s")

# ============================================
# 5. SAVE DATA
# ============================================

# Save as NetCDF
ds_currents = xr.Dataset({
    'u': (['y', 'x'], u_synthetic),
    'v': (['y', 'x'], v_synthetic),
    'latitude': (['y', 'x'], Y),
    'longitude': (['y', 'x'], X),
    'magnitude': (['y', 'x'], current_magnitude)
})

ds_currents.to_netcdf('synthetic_currents.nc')
print(f"\n💾 Saved to: synthetic_currents.nc")
print(f"   File size: ~{u_synthetic.nbytes * 2 / 1e6:.1f} MB")

# ============================================
# FIXED: Save as CSV with pandas
# ============================================
currents_flat = pd.DataFrame({
    'latitude': Y.flatten(),
    'longitude': X.flatten(),
    'u_current': u_synthetic.flatten(),
    'v_current': v_synthetic.flatten(),
    'magnitude': current_magnitude.flatten()
})
currents_flat.to_csv('synthetic_currents.csv', index=False)
print(f"   Also saved as CSV: synthetic_currents.csv")

# ============================================
# 6. QUICK PLOT (Optional)
# ============================================

try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # U-current
    im1 = axes[0].contourf(X, Y, u_synthetic, levels=30, cmap='RdBu_r')
    axes[0].set_xlabel('Longitude (°E)')
    axes[0].set_ylabel('Latitude (°N)')
    axes[0].set_title('Zonal Current (u)')
    plt.colorbar(im1, ax=axes[0], label='m/s')
    
    # V-current
    im2 = axes[1].contourf(X, Y, v_synthetic, levels=30, cmap='RdBu_r')
    axes[1].set_xlabel('Longitude (°E)')
    axes[1].set_ylabel('Latitude (°N)')
    axes[1].set_title('Meridional Current (v)')
    plt.colorbar(im2, ax=axes[1], label='m/s')
    
    plt.tight_layout()
    plt.savefig('synthetic_currents_plot.png', dpi=150)
    print(f"\n📊 Plot saved: synthetic_currents_plot.png")
    plt.close()
    
    # Also plot streamlines
    fig, ax = plt.subplots(figsize=(8, 6))
    skip = 4
    ax.streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                  u_synthetic[::skip, ::skip], v_synthetic[::skip, ::skip],
                  color=current_magnitude[::skip, ::skip], cmap='viridis', density=1.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Ocean Current Streamlines')
    plt.colorbar(ax.collections[0], label='Speed (m/s)')
    plt.tight_layout()
    plt.savefig('synthetic_currents_streamlines.png', dpi=150)
    print(f"   Streamline plot saved: synthetic_currents_streamlines.png")
    plt.close()
except Exception as e:
    print(f"\n⚠️ Could not generate plots: {e}")

print("\n" + "="*60)
print("✅ SYNTHETIC CURRENTS GENERATION COMPLETE")
print("="*60)