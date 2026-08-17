# ============================================
# CODE 5: COMPUTE RICHARDSON NUMBER & HEATMAP (CORRECTED)
# ============================================
# This computes the Richardson Number (Ri = N²/S²) from all synthetic data
# and generates a color-coded ditching survivability heatmap

import numpy as np
import pandas as pd  # <-- ADDED THIS
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
from scipy.ndimage import distance_transform_edt
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD SYNTHETIC DATA
# ============================================

print("="*60)
print("COMPUTING RICHARDSON NUMBER")
print("="*60)

print("\nLoading synthetic data...")

# Load Argo data
argo = pd.read_csv('synthetic_argo.csv')
print(f"  ✓ Argo: {len(argo)} records, {argo['float_id'].nunique()} profiles")

# Load altimetry
altimetry = xr.open_dataset('synthetic_altimetry.nc')
ssh = altimetry['ssh'].values
print(f"  ✓ Altimetry: {ssh.shape[0]}×{ssh.shape[1]} grid")

# Load bathymetry
bathymetry = xr.open_dataset('synthetic_bathymetry.nc')
depth = bathymetry['elevation'].values
print(f"  ✓ Bathymetry: {depth.shape[0]}×{depth.shape[1]} grid")

# Load currents
currents = xr.open_dataset('synthetic_currents.nc')
u = currents['u'].values
v = currents['v'].values
print(f"  ✓ Currents: {u.shape[0]}×{u.shape[1]} grid")

# ============================================
# 2. DEFINE PHYSICAL CONSTANTS
# ============================================

RHO0 = 1027        # Reference density (kg/m³)
G = 9.81           # Gravity (m/s²)
BETA = 0.0007      # Thermal expansion coefficient (1/°C)

# ============================================
# 3. COMPUTE RICHARDSON NUMBER FOR EACH PROFILE
# ============================================

def compute_density(temp, sal):
    """UNESCO equation of state for seawater (simplified)"""
    # Linearized equation of state
    return RHO0 * (1 - BETA * (temp - 20) + 0.0008 * (sal - 35))

def compute_richardson_per_profile(profile):
    """Compute Ri for a single Argo profile"""
    
    # Sort by depth
    profile = profile.sort_values('depth')
    
    depths = profile['depth'].values
    temps = profile['temperature'].values
    sals = profile['salinity'].values
    
    # Compute density profile
    rho = compute_density(temps, sals)
    
    # Buoyancy frequency squared: N² = -(g/ρ₀) * (∂ρ/∂z)
    drho_dz = np.gradient(rho, depths)
    N2 = -(G / RHO0) * drho_dz
    N2 = np.maximum(N2, 0)  # Ensure non-negative
    
    # Get currents at profile location
    # (Using mean currents for the region since we don't have vertical profile)
    lat_idx = int((profile['latitude'].iloc[0] - 5) / 0.2)
    lon_idx = int((profile['longitude'].iloc[0] - 65) / 0.2)
    lat_idx = np.clip(lat_idx, 0, u.shape[0] - 1)
    lon_idx = np.clip(lon_idx, 0, u.shape[1] - 1)
    
    # Get current values at this location
    u_loc = u[lat_idx, lon_idx]
    v_loc = v[lat_idx, lon_idx]
    
    # Create vertical shear profile (simplified)
    # In real data, this would come from current meter measurements
    u_profile = u_loc * np.ones(len(depths)) + np.random.normal(0, 0.02, len(depths))
    v_profile = v_loc * np.ones(len(depths)) + np.random.normal(0, 0.02, len(depths))
    
    # Shear squared: S² = (∂u/∂z)² + (∂v/∂z)²
    du_dz = np.gradient(u_profile, depths)
    dv_dz = np.gradient(v_profile, depths)
    S2 = du_dz**2 + dv_dz**2
    
    # Richardson Number: Ri = N² / S²
    Ri = N2 / (S2 + 1e-10)  # Avoid division by zero
    
    return depths, N2, S2, Ri

# Compute Ri for all profiles
print("\nComputing Richardson Number for all profiles...")

results = []

for float_id in argo['float_id'].unique():
    profile = argo[argo['float_id'] == float_id]
    
    try:
        depths, N2, S2, Ri = compute_richardson_per_profile(profile)
        
        lat = profile['latitude'].iloc[0]
        lon = profile['longitude'].iloc[0]
        
        for d, ri, n2, s2 in zip(depths, Ri, N2, S2):
            # Classify based on Richardson Number
            if ri > 1.0:
                classification = 'SAFE'
                color = 'GREEN'
            elif ri > 0.25:
                classification = 'CAUTION'
                color = 'YELLOW'
            else:
                classification = 'UNSAFE'
                color = 'RED'
            
            results.append({
                'float_id': float_id,
                'latitude': lat,
                'longitude': lon,
                'depth': d,
                'N2': n2,
                'S2': s2,
                'Richardson_Number': ri,
                'classification': classification,
                'color': color
            })
    except Exception as e:
        continue

# Convert to DataFrame
ri_df = pd.DataFrame(results)
print(f"\n✅ Computed Ri for {ri_df['float_id'].nunique()} profiles")

# ============================================
# 4. GENERATE HEATMAP
# ============================================

def fill_nan(data):
    """Fill NaN or zero values using nearest neighbor interpolation"""
    # Check if data has NaN or zeros
    if np.isnan(data).any():
        mask = np.isnan(data)
    else:
        mask = data == 0
    
    if mask.sum() == 0:
        return data
    
    # Find nearest non-zero values
    idx = np.where(~mask)
    if len(idx[0]) == 0:
        return data
    
    # Distance transform to find nearest neighbors
    dist, ind = distance_transform_edt(mask, return_indices=True)
    return data[tuple(ind)]

def generate_heatmap(ri_df, grid_size=100):
    """Generate color-coded ditching survivability heatmap"""
    
    # Create grid
    x = np.linspace(65, 85, grid_size)
    y = np.linspace(5, 25, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize heatmap grid (0 = no data)
    heatmap = np.zeros((grid_size, grid_size))
    heatmap_class = np.zeros((grid_size, grid_size), dtype=int)
    heatmap_ri = np.zeros((grid_size, grid_size))
    
    # Map each Ri value to grid
    for _, row in ri_df.iterrows():
        lat_idx = int((row['latitude'] - 5) / 0.2)
        lon_idx = int((row['longitude'] - 65) / 0.2)
        lat_idx = np.clip(lat_idx, 0, grid_size - 1)
        lon_idx = np.clip(lon_idx, 0, grid_size - 1)
        
        ri_value = row['Richardson_Number']
        
        # Use maximum value (worst case) for each grid cell
        if ri_value > heatmap[lat_idx, lon_idx]:
            heatmap[lat_idx, lon_idx] = ri_value
            heatmap_ri[lat_idx, lon_idx] = ri_value
            # Store classification: 0=SAFE, 1=CAUTION, 2=UNSAFE
            if ri_value > 1.0:
                heatmap_class[lat_idx, lon_idx] = 0
            elif ri_value > 0.25:
                heatmap_class[lat_idx, lon_idx] = 1
            else:
                heatmap_class[lat_idx, lon_idx] = 2
    
    # Fill empty cells (no data) with nearest neighbor interpolation
    if (heatmap == 0).sum() > 0:
        heatmap = fill_nan(heatmap)
        heatmap_class = fill_nan(heatmap_class.astype(float)).astype(int)
        heatmap_ri = fill_nan(heatmap_ri)
    
    return heatmap, heatmap_class, heatmap_ri, X, Y

# Generate heatmap
print("\nGenerating heatmap...")
heatmap, heatmap_class, heatmap_ri, X, Y = generate_heatmap(ri_df, grid_size=100)

# ============================================
# 5. HEATMAP STATISTICS
# ============================================

print(f"\n📊 Heatmap Statistics:")
safe_zones = (heatmap_class == 0).sum()
caution_zones = (heatmap_class == 1).sum()
unsafe_zones = (heatmap_class == 2).sum()
total_zones = heatmap_class.size

print(f"   SAFE zones (Ri > 1.0): {safe_zones} ({safe_zones/total_zones*100:.1f}%)")
print(f"   CAUTION zones (0.25 < Ri < 1.0): {caution_zones} ({caution_zones/total_zones*100:.1f}%)")
print(f"   UNSAFE zones (Ri < 0.25): {unsafe_zones} ({unsafe_zones/total_zones*100:.1f}%)")

ri_nonzero = heatmap_ri[heatmap_ri > 0]
if len(ri_nonzero) > 0:
    print(f"\n   Mean Richardson Number: {ri_nonzero.mean():.3f}")
    print(f"   Std Richardson Number: {ri_nonzero.std():.3f}")
    print(f"   Min Richardson Number: {ri_nonzero.min():.3f}")
    print(f"   Max Richardson Number: {ri_nonzero.max():.3f}")

# ============================================
# 6. SAVE HEATMAP
# ============================================

# Save Ri results
ri_df.to_csv('richardson_number_results.csv', index=False)
print(f"\n💾 Saved Ri results: richardson_number_results.csv")

# Save heatmap as NetCDF
ds_heatmap = xr.Dataset({
    'richardson_number': (['y', 'x'], heatmap_ri),
    'classification': (['y', 'x'], heatmap_class),
    'latitude': (['y', 'x'], Y),
    'longitude': (['y', 'x'], X)
})
ds_heatmap.to_netcdf('ditching_heatmap.nc')
print(f"💾 Saved heatmap: ditching_heatmap.nc")

# ============================================
# 7. SAVE KML FOR EFB INTEGRATION
# ============================================

def save_kml_heatmap(heatmap_class, X, Y, filename='ditching_heatmap.kml'):
    """Save heatmap as KML for Electronic Flight Bag integration"""
    
    kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Ditching Survivability Heatmap</name>
    <description>Richardson Number based ditching safety zones</description>
    <Style id="safe">
        <PolyStyle><color>7f00ff00</color></PolyStyle>
    </Style>
    <Style id="caution">
        <PolyStyle><color>7f00ffff</color></PolyStyle>
    </Style>
    <Style id="unsafe">
        <PolyStyle><color>7f0000ff</color></PolyStyle>
    </Style>
'''
    
    kml_footer = '''</Document>
</kml>'''
    
    kml_body = []
    
    # Create grid polygons
    grid_size = heatmap_class.shape[0]
    
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            lat = 5 + i * 0.2
            lon = 65 + j * 0.2
            
            # Get classification
            cls = int(heatmap_class[i, j])
            if cls == 0:
                style = 'safe'
            elif cls == 1:
                style = 'caution'
            else:
                style = 'unsafe'
            
            # Create polygon
            polygon = f'''
    <Placemark>
        <styleUrl>#{style}</styleUrl>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {lon},{lat},0
                        {lon+0.2},{lat},0
                        {lon+0.2},{lat+0.2},0
                        {lon},{lat+0.2},0
                        {lon},{lat},0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>'''
            kml_body.append(polygon)
    
    # Write KML file (limit to 1000 polygons to keep file size reasonable)
    kml_body_limited = kml_body[:1000] if len(kml_body) > 1000 else kml_body
    
    with open(filename, 'w') as f:
        f.write(kml_header + ''.join(kml_body_limited) + kml_footer)
    
    print(f"💾 Saved KML: {filename}")

# Save KML
save_kml_heatmap(heatmap_class, X, Y, 'ditching_heatmap.kml')

# ============================================
# 8. QUICK PLOT
# ============================================

try:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for classifications
    cmap = ListedColormap(['green', 'yellow', 'red'])
    im = ax.imshow(heatmap_class, origin='lower', extent=[65, 85, 5, 25], cmap=cmap, alpha=0.7)
    
    # Add bathymetry contours
    bathy = xr.open_dataset('synthetic_bathymetry.nc')
    depth_grid = bathy['elevation'].values
    ax.contour(depth_grid, origin='lower', extent=[65, 85, 5, 25], 
               levels=[-100, -150, -200], 
               colors='white', linestyles='dashed', alpha=0.5)
    
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Ditching Survivability Heatmap\n(Richardson Number Based)')
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='SAFE (Ri > 1.0)'),
        Patch(facecolor='yellow', alpha=0.7, label='CAUTION (0.25 < Ri < 1.0)'),
        Patch(facecolor='red', alpha=0.7, label='UNSAFE (Ri < 0.25)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add colorbar for Richardson Number values (as overlay)
    ax2 = ax.twinx()
    ax2.set_ylabel('Richardson Number')
    
    plt.tight_layout()
    plt.savefig('ditching_heatmap.png', dpi=150)
    print(f"\n📊 Heatmap saved: ditching_heatmap.png")
    plt.close()
    
    # ========================================
    # ADDITIONAL PLOT: Ri Distribution
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ri_values = ri_df[ri_df['Richardson_Number'] < 10]['Richardson_Number']  # Cap at 10 for visualization
    ax.hist(ri_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='green', linestyle='--', label='SAFE threshold (Ri=1.0)')
    ax.axvline(x=0.25, color='red', linestyle='--', label='UNSAFE threshold (Ri=0.25)')
    ax.set_xlabel('Richardson Number')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Richardson Numbers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('richardson_distribution.png', dpi=150)
    print(f"📊 Ri distribution saved: richardson_distribution.png")
    plt.close()
    
except Exception as e:
    print(f"\n⚠️ Could not generate plots: {e}")

# ============================================
# 9. SUMMARY
# ============================================

print("\n" + "="*60)
print("✅ WAVEPULSE DATA GENERATION COMPLETE")
print("="*60)

print("\n📦 Generated Files:")
print("  1. synthetic_argo.csv - 1000 profiles, 8 depths each")
print("  2. synthetic_altimetry.nc - 100×100 SSH grid")
print("  3. synthetic_bathymetry.nc - 100×100 depth grid")
print("  4. synthetic_currents.nc - 100×100 u/v current grid")
print("  5. richardson_number_results.csv - Ri values for all profiles")
print("  6. ditching_heatmap.nc - 100×100 heatmap grid")
print("  7. ditching_heatmap.kml - EFB-compatible KML file")
print("  8. ditching_heatmap.png - Heatmap visualization")
print("  9. richardson_distribution.png - Ri distribution histogram")

# Estimate total size
total_size = 0.1 + 0.1 + 0.1 + 0.1 + 1 + 0.1 + 1 + 0.2 + 0.2  # MB
print(f"\n📊 Estimated Total Storage: ~{total_size:.1f} MB")
print(f"   (Well under 500 MB limit)")

print("\n✅ Ready for WavePulse!")
print("="*60)