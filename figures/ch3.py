import numpy as np
import matplotlib.pyplot as plt

# --- Set Global Font to Arial ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- Data ---
omegas = np.array([16, 32, 64, 128])
# Fixed Damping (eta=const): Rapid growth
fixed_kappa = np.array([1.2e4, 5.8e4, 2.9e5, 1.4e6])
# Adaptive Scaling (eta ~ omega^0.694): Stabilized growth
adaptive_kappa = np.array([1.2e4, 2.1e4, 3.8e4, 6.5e4])

# --- Plot Construction ---
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

# Fixed Damping Plot
ax.loglog(omegas, fixed_kappa,
          color='#888888', linestyle='--', linewidth=2.0,
          marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
          label='Fixed PML Damping ($\eta = \mathrm{const.}$)')

# Adaptive Scaling Plot
ax.loglog(omegas, adaptive_kappa,
          color='#000000', linestyle='-', linewidth=2.5,
          marker='s', markersize=8,
          label='Adaptive PML Scaling ($\eta \propto \omega^{0.694}$)')

# --- Overlap & Collision Management ---

# 1. Expand Y-limits to prevent text/line collisions with top/bottom spines
ax.set_ylim(8e3, 5e6)

# 2. Add padding to labels and title
ax.set_title("Stabilizing Algebraic Difficulty via Adaptive Scaling", 
             fontsize=16, fontweight='bold', pad=25)
ax.set_xlabel("Frequency $\omega$ (Dimensionless)", fontsize=13, labelpad=12)
ax.set_ylabel("Condition Number $\kappa(\mathbf{A})$", fontsize=13, labelpad=12)

# 3. Handle Ticks: Use minor_formatter=Null to keep it clean
ax.set_xticks(omegas)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticklabels([str(w) for w in omegas], fontsize=11)

# 4. Legend Placement: Moved to center-right or lower-right with a frame alpha
# to ensure it doesn't obscure the line growth
ax.legend(loc='upper left', fontsize=11, frameon=True, 
          facecolor='white', framealpha=0.9, edgecolor='#CCCCCC', borderpad=1)

# 5. Clean Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10, pad=8)

# 6. Subtle Grid for readability
ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#EEEEEE')

plt.tight_layout()

# Save instruction
# plt.savefig('condition_number_scaling_arial.png', bbox_inches='tight')
plt.show()