import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import sys

# Ensure we can import the module from the current directory
sys.path.append(os.getcwd())
try:
    from apply_filter_and_fluorescence import process_single_layer, run_stack
except ImportError:
    pass

# Setup paths
artifact_dir = r"C:\Users\Blake Mizzell\.gemini\antigravity\brain\6bf2efc1-b060-441d-8f5b-788934fd2ca4"
os.makedirs(artifact_dir, exist_ok=True)

# Re-create the mock data
att_file = 'test_att_physics.csv'
edpl_folder = 'test_edpl_physics'
os.makedirs(edpl_folder, exist_ok=True)

E_kev = np.linspace(1, 100, 500) # finer grid for nicer plots
E_mev = E_kev / 1000.0

# Mo Attenuation
mu_mo = np.where(E_kev < 20.0, 10.0, 50.0)
tauK_mo = np.where(E_kev < 20.0, 0.0, 40.0)

# Cu Attenuation
mu_cu = np.where(E_kev < 9.0, 15.0, 60.0)
tauK_cu = np.where(E_kev < 9.0, 0.0, 45.0)

# W Attenuation
mu_w = np.where(E_kev < 69.0, 8.0, 40.0)
tauK_w = np.where(E_kev < 69.0, 0.0, 35.0)

df_list = []
df_list.append(pd.DataFrame({'Energy_MeV': E_mev, 'Energy_KeV': E_kev, 'Element': 42, 'Mu_over_rho': mu_mo, 'Mu_en_over_rho': mu_mo}))
df_list.append(pd.DataFrame({'Energy_MeV': E_mev, 'Energy_KeV': E_kev, 'Element': 29, 'Mu_over_rho': mu_cu, 'Mu_en_over_rho': mu_cu}))
df_list.append(pd.DataFrame({'Energy_MeV': E_mev, 'Energy_KeV': E_kev, 'Element': 74, 'Mu_over_rho': mu_w, 'Mu_en_over_rho': mu_w}))
att_df = pd.concat(df_list)
att_df.to_csv(att_file, index=False)

pd.DataFrame({'E_keV': E_kev, 'tauK_over_rho_cm2_per_g': tauK_mo}).to_csv(os.path.join(edpl_folder, 'test_ZA042.csv'), index=False)
pd.DataFrame({'E_keV': E_kev, 'tauK_over_rho_cm2_per_g': tauK_cu}).to_csv(os.path.join(edpl_folder, 'test_ZA029.csv'), index=False)
pd.DataFrame({'E_keV': E_kev, 'tauK_over_rho_cm2_per_g': tauK_w}).to_csv(os.path.join(edpl_folder, 'test_ZA074.csv'), index=False)

# PLOT 1: Thickness Sweep
geom = {'beam_angle_deg': 180.0}
phi_in = np.zeros_like(E_kev)
phi_in[np.abs(E_kev - 25.0).argmin()] = 1e6

thicknesses = np.logspace(-3, 2, 100) # 0.001 to 100 mm
fl_results = []
for t in thicknesses:
    _, fl_out, _, _ = process_single_layer(E_kev, phi_in, 'Mo', t, att_df, edpl_folder, geom)
    fl_results.append(np.sum(fl_out))

plt.figure(figsize=(10, 6))
plt.plot(thicknesses, fl_results, lw=3, color='dodgerblue')
plt.xscale('log')
plt.title('Test 3: Fluorescence vs Filter Thickness (Mo Filter @ 25 keV)', fontsize=14)
plt.xlabel('Thickness (mm) [Log Scale]', fontsize=12)
plt.ylabel('Total Escaping Fluorescence (Photons/s)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'thickness_sweep.png'))
plt.close()

# PLOT 2: Cascading Multi-Layer
phi_in_df = pd.DataFrame({'Energy_keV': E_kev, 'Relative_Intensity': 0.0})
# Broaden the input spike slightly so it plots visibly
mask = np.abs(E_kev - 75.0) < 0.5
phi_in_df.loc[mask, 'Relative_Intensity'] = 1e8 / np.sum(mask)
spec_file = 'test_spec.csv'
phi_in_df.to_csv(spec_file, index=False)

out_file = 'test_out.csv'
filters = [
    {'material': 'W', 'thickness': 0.05},
    {'material': 'Cu', 'thickness': 0.05}
]

run_stack(spec_file, filters, att_file, edpl_folder, geom, out_file, save_fluorescence=True)

res_df = pd.read_csv(out_file)
fluoro_df = pd.read_csv('test_out_fluoro.csv')

plt.figure(figsize=(12, 6))
# We will use bar plotting for the discrete fluorescence spikes so they're highly visible
plt.plot(res_df['Energy_KeV'], res_df['Photons_per_s'], label='Final Output Spectrum', color='navy', lw=2)

if not fluoro_df.empty:
    plt.bar(fluoro_df['Energy_KeV'], fluoro_df['Photons_per_s'], width=0.5, label='Pure Fluorescence Generated', color='crimson', alpha=0.7)

plt.annotate('Input Spike (75 keV)', xy=(75, 1e7), xytext=(80, 5e7), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
plt.annotate('W K-alpha\n(~59 keV)', xy=(59, fluoro_df['Photons_per_s'][np.abs(fluoro_df['Energy_KeV']-59).argmin()]), 
             xytext=(45, 1e6), arrowprops=dict(facecolor='crimson', shrink=0.05), fontsize=10)
plt.annotate('Cu K-alpha\n(~8 keV)\nTriggered by W', xy=(8, fluoro_df['Photons_per_s'][np.abs(fluoro_df['Energy_KeV']-8).argmin()]), 
             xytext=(15, 1e4), arrowprops=dict(facecolor='crimson', shrink=0.05), fontsize=10)

plt.title('Test 5: Cascading Fluorescence (75 keV -> W -> Cu)', fontsize=14)
plt.xlabel('Energy (keV)', fontsize=12)
plt.ylabel('Intensity (Photons/s)', fontsize=12)
plt.yscale('log')
plt.ylim(1e1, 1e8)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'cascading_fluorescence.png'))
plt.close()

# Cleanup
if os.path.exists(spec_file): os.remove(spec_file)
if os.path.exists(out_file): os.remove(out_file)
if os.path.exists('test_out_fluoro.csv'): os.remove('test_out_fluoro.csv')

print("Plots generated.")
