# ============================================
# CODE 2: GENERATE SYNTHETIC ALTIMETRY (CORRECTED)
# ============================================
# This generates sea surface height (SSH) anomalies with
# internal wave signatures on a 100×100 grid

import numpy as np
import pandas as pd  # <-- THIS WAS MISSING
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# ============================================
# 1. DEFINE GRID PARAMETERS
# ============================================

# Indian Ocean region
LAT_MIN, LAT_MAX = 5, 25   # 5°N to 25°N
LON_MIN, LON_MAX = 65, 85  # 65°E to 85°E

# Grid resolution
GRID_SIZE = 100  # 100×100 grid

# SSH statistics (from AVISO data)
SSH_MEAN = 0.0       # Mean SSH anomaly (meters)
SSH_STD = 0.12       # Standard deviation of SSH anomalies (meters)

# ============================================
# 2. DEFINE GENERATION FUNCTIONS
# ============================================

def generate_synthetic_altimetry(grid_size=100, random_seed=42):
    """
    Generate synthetic SSH with internal wave signatures.
    
    Features:
    - Background ocean variability
    - Internal tides (M2 constituent)
    - Internal solitary waves (ISWs)
    - Mesoscale eddies
    - Multiple wave modes
    
    Returns:
        ssh_grid: 2D array of SSH anomalies
        X, Y: Coordinate grids
    """
    np.random.seed(random_seed)
    
    # Create coordinate grids
    x = np.linspace(LON_MIN, LON_MAX, grid_size)
    y = np.linspace(LAT_MIN, LAT_MAX, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # ========================================
    # COMPONENT 1: Background Variability
    # ========================================
    # Correlated random field (ocean mesoscale)
    background = np.random.normal(SSH_MEAN, SSH_STD * 0.4, (grid_size, grid_size))
    
    # Add spatial correlation (smoothing)
    background = gaussian_filter(background, sigma=2)
    
    # ========================================
    # COMPONENT 2: Internal Tides (M2)
    # ========================================
    # M2 internal tide - detected in SSH by altimetry
    # Wavelength: ~4.5° longitude, 3.2° latitude
    internal_tide = 0.12 * np.sin(2 * np.pi * X / 4.5) * np.cos(2 * np.pi * Y / 3.2)
    
    # ========================================
    # COMPONENT 3: Internal Solitary Waves
    # ========================================
    # ISWs from TITE/GAN method
    # Multiple ISW packets
    isw_packets = []
    
    # Packet 1: Main ISW
    isw1_amp = np.random.uniform(0.08, 0.20)
    isw1 = isw1_amp * np.exp(-((X - 72)**2 + (Y - 14)**2) / 5) * np.cos(2 * np.pi * X / 2)
    isw_packets.append(isw1)
    
    # Packet 2: Secondary ISW
    isw2_amp = np.random.uniform(0.05, 0.12)
    isw2 = isw2_amp * np.exp(-((X - 78)**2 + (Y - 18)**2) / 4) * np.sin(2 * np.pi * Y / 3)
    isw_packets.append(isw2)
    
    # Packet 3: Small ISW near coast
    isw3_amp = np.random.uniform(0.03, 0.08)
    isw3 = isw3_amp * np.exp(-((X - 68)**2 + (Y - 10)**2) / 3) * np.cos(2 * np.pi * X / 1.5)
    isw_packets.append(isw3)
    
    isw = sum(isw_packets)
    
    # ========================================
    # COMPONENT 4: Mesoscale Eddies
    # ========================================
    # Warm-core and cold-core eddies
    eddies = []
    
    # Eddy 1: Warm-core (positive SSH)
    eddy1_amp = np.random.uniform(0.05, 0.12)
    eddy1 = eddy1_amp * np.exp(-((X - 70)**2 + (Y - 10)**2) / 15)
    eddies.append(eddy1)
    
    # Eddy 2: Cold-core (negative SSH)
    eddy2_amp = -np.random.uniform(0.04, 0.10)
    eddy2 = eddy2_amp * np.exp(-((X - 80)**2 + (Y - 20)**2) / 12)
    eddies.append(eddy2)
    
    # Eddy 3: Small eddy
    eddy3_amp = np.random.uniform(0.02, 0.06)
    eddy3 = eddy3_amp * np.exp(-((X - 66)**2 + (Y - 8)**2) / 8)
    eddies.append(eddy3)
    
    eddy = sum(eddies)
    
    # ========================================
    # COMPONENT 5: Higher Wave Modes
    # ========================================
    wave_mode_2 = 0.06 * np.sin(4 * np.pi * X / 8) * np.cos(4 * np.pi * Y / 6)
    wave_mode_3 = 0.04 * np.cos(6 * np.pi * X / 10) * np.sin(6 * np.pi * Y / 8)
    
    # ========================================
    # COMPONENT 6: Bathymetry Influence
    # ========================================
    # Shallow water amplifies internal waves
    depth_factor = 1 + 0.2 * np.exp(-((X - 67)**2 + (Y - 9)**2) / 10)
    
    # ========================================
    # COMBINE ALL COMPONENTS
    # ========================================
    ssh = (background + 
           internal_tide + 
           isw + 
           eddy + 
           wave_mode_2 + 
           wave_mode_3)
    
    # Apply bathymetry influence
    ssh = ssh * depth_factor
    
    # Scale to match real SSH statistics
    ssh = (ssh - ssh.mean()) / ssh.std() * SSH_STD + SSH_MEAN
    
    # ========================================
    # ADD REALISTIC NOISE
    # ========================================
    # Sensor noise: ±0.01m (typical altimetry accuracy)
    sensor_noise = np.random.normal(0, 0.01, (grid_size, grid_size))
    ssh = ssh + sensor_noise
    
    return ssh, X, Y

# ============================================
# 3. GENERATE DATA
# ============================================

print("="*60)
print("GENERATING SYNTHETIC ALTIMETRY")
print("="*60)

# Generate 100×100 grid
ssh_synthetic, X, Y = generate_synthetic_altimetry(grid_size=GRID_SIZE, random_seed=42)

# ============================================
# 4. STATISTICS AND VALIDATION
# ============================================

print(f"\n✅ Generated {ssh_synthetic.shape[0]}×{ssh_synthetic.shape[1]} grid")
print(f"   Total points: {ssh_synthetic.size}")

print(f"\n📊 SSH Statistics:")
print(f"   Mean: {ssh_synthetic.mean():.3f} m")
print(f"   Std: {ssh_synthetic.std():.3f} m")
print(f"   Min: {ssh_synthetic.min():.3f} m")
print(f"   Max: {ssh_synthetic.max():.3f} m")

# Count internal wave regions
internal_wave_mask = np.abs(ssh_synthetic - ssh_synthetic.mean()) > (1.5 * SSH_STD)
n_wave_regions = internal_wave_mask.sum()
print(f"\n📊 Internal Wave Regions:")
print(f"   Points with strong SSH anomalies: {n_wave_regions} ({n_wave_regions/ssh_synthetic.size*100:.1f}%)")

# ============================================
# 5. SAVE DATA
# ============================================

# Save as NetCDF
ds_ssh = xr.Dataset({
    'ssh': (['y', 'x'], ssh_synthetic),
    'latitude': (['y', 'x'], Y),
    'longitude': (['y', 'x'], X)
})

ds_ssh.to_netcdf('synthetic_altimetry.nc')
print(f"\n💾 Saved to: synthetic_altimetry.nc")
print(f"   File size: ~{ssh_synthetic.nbytes / 1e6:.1f} MB")

# ============================================
# FIXED: Save as CSV with pandas
# ============================================
ssh_flat = pd.DataFrame({
    'latitude': Y.flatten(),
    'longitude': X.flatten(),
    'ssh': ssh_synthetic.flatten()
})
ssh_flat.to_csv('synthetic_altimetry.csv', index=False)
print(f"   Also saved as CSV: synthetic_altimetry.csv")

# ============================================
# 6. QUICK PLOT (Optional)
# ============================================

try:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(X, Y, ssh_synthetic, levels=50, cmap='RdBu_r')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Synthetic Sea Surface Height Anomalies\n(Internal Waves + Eddies)')
    plt.colorbar(im, label='SSH (m)')
    plt.tight_layout()
    plt.savefig('synthetic_altimetry_plot.png', dpi=150)
    print(f"\n📊 Plot saved: synthetic_altimetry_plot.png")
    plt.close()
except Exception as e:
    print(f"\n⚠️ Could not generate plot: {e}")

print("\n" + "="*60)
print("✅ SYNTHETIC ALTIMETRY GENERATION COMPLETE")
print("="*60)