import numpy as np
import matplotlib.pyplot as plt

# --- Set Global Font to Arial ---
# Uses DejaVu Sans as a high-quality fallback if Arial isn't on the system path
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- Shared Parameters ---
N = 512
n_pml = 112
omegas = [16, 32, 64, 128]
# Peak damping values (eta) based on adaptive scaling: eta ~ omega^0.694
etas = [42.5, 68.7, 111.2, 179.9] 

def get_sigma_1d(N, n_pml, eta):
    """Calculates the quadratic damping profile for a given frequency's peak eta."""
    n = np.arange(N)
    sigma = np.zeros(N)
    # Left PML (0 to 112)
    idx_l = n < n_pml
    sigma[idx_l] = eta * ((n_pml - n[idx_l]) / n_pml)**2
    # Right PML (400 to 512)
    idx_r = n > (N - n_pml)
    sigma[idx_r] = eta * ((n[idx_r] - (N - n_pml)) / n_pml)**2
    return n, sigma

# =================================================================
# FIGURE 1: 1D PML Damping Profiles
# =================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
axes1 = axes1.flatten()

for i, (w, eta) in enumerate(zip(omegas, etas)):
    n, sigma = get_sigma_1d(N, n_pml, eta)
    
    # Plotting with light fill for visual weight
    axes1[i].plot(n, sigma, color='black', linewidth=1.5)
    axes1[i].fill_between(n, sigma, color='#E0E0E0', alpha=0.5)
    
    # Formatting - Using raw strings (rf"") to fix SyntaxWarnings
    axes1[i].set_title(rf"Frequency $\omega = {w}$", fontsize=12, fontweight='bold', pad=10)
    axes1[i].set_ylim(0, 200)
    axes1[i].set_xlim(0, 512)
    axes1[i].set_xticks([0, 112, 400, 512])
    axes1[i].grid(True, linestyle=':', alpha=0.4)
    
    if i >= 2: axes1[i].set_xlabel("Grid Index (n)", fontsize=10)
    if i % 2 == 0: axes1[i].set_ylabel(r"Damping Strength $\sigma$", fontsize=10)

fig1.suptitle("1D PML Damping Profiles", fontsize=15, fontweight='bold', y=0.98)
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.savefig('pml_1d_profiles.png')

# =================================================================
# FIGURE 2: 2D PML Spatial Layout
# =================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
axes2 = axes2.flatten()

for i, (w, eta) in enumerate(zip(omegas, etas)):
    _, s1d = get_sigma_1d(N, n_pml, eta)
    # Generate 2D grid: sigma_total(x,y) = sigma(x) + sigma(y)
    S_x, S_y = np.meshgrid(s1d, s1d)
    S_total = S_x + S_y
    
    # Plotting with grayscale colormap
    im = axes2[i].imshow(S_total, cmap='Greys', origin='lower', extent=[0, 512, 0, 512])
    
    # Draw dashed interior boundary for the 288x288 domain
    rect = plt.Rectangle((112, 112), 288, 288, linewidth=1.2, 
                         edgecolor='black', facecolor='none', linestyle='--')
    axes2[i].add_patch(rect)
    
    # Labeling and Ticks
    axes2[i].set_title(rf"$\omega = {w}$ Spatial Distribution", fontsize=12, fontweight='bold', pad=10)
    axes2[i].set_xticks([0, 112, 400, 512])
    axes2[i].set_yticks([0, 112, 400, 512])
    
    # Individual Colorbars (Scaled to each plot's max)
    cbar = fig2.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

fig2.suptitle("2D PML Spatial Extent", fontsize=15, fontweight='bold', y=0.96)
fig2.tight_layout(pad=3.0)
fig2.savefig('pml_2d_layout.png')

print("Success: 'pml_1d_profiles.png' and 'pml_2d_layout.png' generated.")
plt.show()