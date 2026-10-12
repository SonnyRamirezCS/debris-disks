"""
Split the clean stars into temperature groups (Skorpen's suggestion:
cooler and hotter stars behave differently, so report results by group).

Only keeps stars inside this term's K star range from VOCED_step4.py
(3930 K < Teff < 5380 K). Last semester's sample used 3500-5000 K, so on that
sample this drops the stars cooler than K (34 of 144) and is missing the
hottest K stars until the pipeline is rerun from the step 4 sample.

Groups use Gaia DR3 temperatures (Teff) and the usual dwarf star temperature
scale (Pecaut & Mamajek 2013):
    Early K  (K0-K3)    4700 - 5380 K
    Mid K    (K4-K5)    4400 - 4700 K
    Late K   (K6-K9)    3930 - 4400 K

For each star it also works out:
    Ks_minus_24  - 2MASS Ks minus Spitzer 24 um, in magnitudes (bigger = redder = more excess)
    can_see_star - True if Spitzer could detect the star's own light at 24 um
                   (calibrated prediction >= 3x the 24 um error). These are the
                   "comparable" stars. The rest only made the sample because
                   something extra made them bright at 24 um.
    near_edge    - Teff within 100 K of a group boundary (Gaia's Teff can be off by
                   about that much, so these could belong to the next group)
    aquila_patch - one of the clean stars just above the galactic plane cut in the
                   Aquila Rift area (l 23-29, b +5 to +8). Several have big excesses,
                   so treat them with suspicion.

Run from the debris-disks folder:  python temp_groups.py
Needs data/kdwarf_sfr.csv (made by sfr_cut.py)
Makes data/clean_temp_groups.csv and figures/temp_groups.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # just save the plot, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path('data')
FIGS = Path('figures')

# ---- Settings ----
TEFF_MIN, TEFF_MAX = 3930, 5380       # K star range from VOCED_step4.py
EDGES = [4400, 4700]                    # group boundaries in K
GROUPS = ['Late K', 'Mid K', 'Early K']  # coolest to hottest
SPECTRAL = {'Late K': 'K6-K9', 'Mid K': 'K4-K5', 'Early K': 'K0-K3'}
RJ_CORRECTION = 1.329   # measured / Rayleigh-Jeans at 24 um for plain K dwarfs (sed_fit_j2133.py)
SEE_MIN = 3             # calibrated star-only 24 um flux must be >= 3x the error
PLAIN_MAX = 0.75        # Ks-[24] below this counts as "plain" (no obvious excess)
EDGE_K = 100            # flag stars this close to a boundary
CANDIDATE = 'SSTSL2 J213307.73+830501.2'
KS_ZP, M24_ZP = 666.7, 7.17   # zero points in Jy (2MASS Ks, MIPS 24)

df = pd.read_csv(DATA / 'kdwarf_sfr.csv')
c = df[df['clean'].astype(bool)].copy()
n_clean = len(c)
c = c[(c['Teff'] > TEFF_MIN) & (c['Teff'] < TEFF_MAX)].copy()
print(f'Clean stars: {n_clean}, inside {TEFF_MIN}-{TEFF_MAX} K: {len(c)} '
      f'(dropped {n_clean - len(c)} outside the K star range)')
if c['Teff'].max() < TEFF_MAX - 200:
    print(f"NOTE: the hottest star here is {c['Teff'].max():.0f} K, so the sample is missing the "
          f"hottest K stars. Rerun the pipeline from the step 4 sample to add them.")

# ---- temperature group for each star ----
c['temp_group'] = pd.cut(c['Teff'], [TEFF_MIN] + EDGES + [TEFF_MAX], right=False, labels=GROUPS)
c['spectral_type'] = c['temp_group'].map(SPECTRAL)
c['near_edge'] = np.min(np.abs(c['Teff'].values[:, None] - np.array(EDGES)[None, :]), axis=1) < EDGE_K

# ---- excess measures ----
c['Ks_minus_24'] = (-2.5 * np.log10(c['k'] * 1e-6 / KS_ZP)) - (-2.5 * np.log10(c['m1_f_psf'] * 1e-6 / M24_ZP))
c['star_only_24'] = c['m1_pred'] * RJ_CORRECTION          # calibrated prediction, uJy
c['can_see_star'] = c['star_only_24'] / c['m1_df_psf'] >= SEE_MIN
c['ratio_24'] = c['m1_f_psf'] / c['star_only_24']          # measured / calibrated prediction
c['ratio_rj'] = c['m1_f_psf'] / c['m1_pred']               # measured / plain Rayleigh-Jeans
c['dered_bp_rp'] = c['BP-RP'] - c['E(BP-RP)']
c['aquila_patch'] = c['gal_l'].between(23, 29) & c['gal_b'].between(5, 8)

# ---- summary table per group ----
rows = []
for g in GROUPS:
    s = c[c['temp_group'] == g]
    see = s[s['can_see_star']]
    plain = see[see['Ks_minus_24'] < PLAIN_MAX]
    top = see.sort_values('Ks_minus_24', ascending=False).head(1)
    biggest = (f"{top['objid'].iloc[0].replace('SSTSL2 ', '')} ({top['ratio_24'].iloc[0]:.1f}x)"
               if len(top) else 'none')
    rows.append({
        'group': g,
        'type': SPECTRAL[g],
        'Teff range (K)': f"{s['Teff'].min():.0f}-{s['Teff'].max():.0f}" if len(s) else 'none',
        'stars': len(s),
        'Spitzer sees star': len(see),
        'near edge': int(s['near_edge'].sum()),
        'plain stars': len(plain),
        'plain median Ks-[24]': round(plain['Ks_minus_24'].median(), 2),
        'plain median meas/RJ': round(plain['ratio_rj'].median(), 2),
        'chi_24 >= 3 (of seen)': int((see['chi_24'] >= 3).sum()),
        '>= 2x calibrated (of seen)': int((see['ratio_24'] >= 2).sum()),
        'biggest excess (seen)': biggest,
    })
summary = pd.DataFrame(rows)
with pd.option_context('display.width', 200, 'display.max_columns', 20):
    print(summary.to_string(index=False))

faint = c[~c['can_see_star']]
print(f"\nStars Spitzer can't see on their own: {len(faint)}, "
      f"and {int((faint['chi_24'] >= 3).sum())} of them have chi_24 >= 3 (they were picked up because they're bright).")
print(f"Aquila Rift patch stars: {int(c['aquila_patch'].sum())} "
      f"({int((c['aquila_patch'] & c['can_see_star']).sum())} that Spitzer can see)")
print(f"Stars within {EDGE_K} K of a boundary: {int(c['near_edge'].sum())}")

keep = ['objid', 'ra', 'dec', 'gal_l', 'gal_b', 'Teff', 'temp_group', 'spectral_type', 'near_edge',
        'dered_bp_rp', 'Dist', 'Ks_minus_24', 'm1_f_psf', 'm1_df_psf', 'm1_pred', 'star_only_24',
        'ratio_rj', 'ratio_24', 'chi_24', 'chi_8', 'chi_45', 'can_see_star', 'aquila_patch']
c[keep].sort_values(['temp_group', 'Teff']).to_csv(DATA / 'clean_temp_groups.csv', index=False)
summary.to_csv(DATA / 'clean_temp_groups_summary.csv', index=False)
print('\nSaved data/clean_temp_groups.csv and data/clean_temp_groups_summary.csv')

# ---- figure: Ks-[24] vs temperature ----
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#a3a29d'
fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [2.1, 1]})

# group bands
lims = [TEFF_MIN, *EDGES, TEFF_MAX]
for i, g in enumerate(GROUPS):
    if i % 2 == 0:
        ax.axvspan(lims[i], lims[i + 1], color='#f3f2ef', zorder=0)
    ax.text((lims[i] + lims[i + 1]) / 2, 1.01, f'{g} ({SPECTRAL[g]})', ha='center', va='bottom',
            fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())

see, nosee = c[c['can_see_star']], c[~c['can_see_star']]
ax.scatter(nosee['Teff'], nosee['Ks_minus_24'], s=26, facecolors='none', edgecolors=GRAY, lw=1,
           label=f"Spitzer can't see the star alone ({len(nosee)})", zorder=2)
ax.scatter(see['Teff'], see['Ks_minus_24'], s=34, color=BLUE, edgecolors='white', lw=1,
           label=f'Spitzer can see the star alone ({len(see)})', zorder=3)
aq = c[c['aquila_patch'] & c['can_see_star']]
ax.scatter(aq['Teff'], aq['Ks_minus_24'], s=90, facecolors='none', edgecolors=INK2, lw=1, ls=':',
           label='Aquila Rift patch (suspect)', zorder=3)
cand = c[c['objid'] == CANDIDATE]
ax.scatter(cand['Teff'], cand['Ks_minus_24'], s=230, marker='*', color=ORANGE, edgecolors='white', lw=1.2,
           zorder=4, label='J2133+83')
ax.annotate('J2133+83', (cand['Teff'].iloc[0], cand['Ks_minus_24'].iloc[0]), xytext=(12, -4),
            textcoords='offset points', fontsize=9, color=INK)

ax.set_xlim(lims[-1] + 40, lims[0] - 40)   # hot on the left, like an HR diagram
ax.set_ylim(-0.3, c['Ks_minus_24'].max() + 0.2)
ax.set_xlabel('Gaia temperature (K)', color=INK)
ax.set_ylabel('Ks - [24] (mag), higher = more 24 micron light', color=INK)
ax.grid(axis='y', color=GRID, lw=0.8)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
ax.set_title(f'{len(c)} clean K stars by temperature group', loc='left', fontsize=11, color=INK, pad=22)

# right panel: plain-star color per group (comparable stars only)
plain = see[see['Ks_minus_24'] < PLAIN_MAX]
ORDER = GROUPS[::-1]   # hot to cool, same as the left panel
for i, g in enumerate(ORDER):
    v = plain.loc[plain['temp_group'] == g, 'Ks_minus_24']
    if v.empty:
        continue
    axb.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), v,
                s=18, color=BLUE, alpha=0.7, edgecolors='none')
    axb.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=INK, lw=2)
    axb.text(i + 0.31, v.median(), f'{v.median():.2f}', ha='left', va='center', fontsize=8.5, color=INK)
axb.set_xticks(range(len(ORDER)), ORDER, fontsize=8.5)
axb.set_xlim(-0.5, len(ORDER) - 0.2)
axb.set_ylabel('Ks - [24] (mag)', color=INK)
axb.set_title('Plain stars only: median color per group', loc='left', fontsize=10, color=INK, pad=22)
axb.grid(axis='y', color=GRID, lw=0.8)
axb.set_ylim(-0.3, 0.85)

for a in (ax, axb):
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        a.spines[side].set_color(GRAY)
    a.tick_params(colors=INK2)

fig.tight_layout()
fig.savefig(FIGS / 'temp_groups.png', dpi=150)
print('Saved figures/temp_groups.png')
