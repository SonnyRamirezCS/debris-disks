"""
Star forming region cut (version 3).

Cuts five kinds of stars that can fake a debris disk:
1. Young stars (SIMBAD tag or Em*)
2. Stars in crowded spots. Spitzer looked at most debris disk type stars one
   at a time, so real candidates are usually alone. A bunch of our K stars
   packed together means Spitzer was pointed at a cluster, and clusters are young.
3. Stars toward the Magellanic Clouds (a K dwarf there would be too faint for
   Spitzer, so these are probably giants in those galaxies)
4. Stars near the Milky Way's disk, where glowing dust can add fake 24 micron light
5. Known young regions (Taurus, Orion, Upper Sco, etc.) and stars in front of
   the Andromeda and Triangulum galaxies

Run from the debris-disks folder:  python sfr_cut.py
Needs data\\kdwarf_labeled.csv (made by young_star_check.py)
Makes data\\kdwarf_sfr.csv (with a 'clean' column) and figures\\sfr_cut.png
"""
import re
import warnings

import matplotlib
matplotlib.use('Agg')  # just save the plot, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

# ---- Settings (change these if Skorpen or Tyler suggest something different) ----
RADIUS_DEG = 0.5   # how far around each star to look (degrees on the sky)
MIN_YOUNG = 3      # this many tagged young neighbors or more = cut
MIN_CROWD = 3      # this many sample neighbors of ANY kind or more = cut
PLANE_B = 5        # cut stars within this many degrees of the Milky Way's disk
LMC_R, SMC_R = 5, 4  # cut stars within this many degrees of the Magellanic Clouds

# ---- Load the labeled K dwarfs ----
df = pd.read_csv(r'data\kdwarf_labeled.csv').copy()
df['young'] = df['young'].astype(bool) | (df['otype'] == 'Em*')
df['evolved'] = df['evolved'].astype(bool)

# ---- Get RA/Dec straight from the Spitzer name ----
# SSTSL2 J060601.34-061104.3  ->  RA 06h06m01.34s, Dec -06d11m04.3s
PAT = re.compile(r'J(\d{2})(\d{2})(\d{2}\.\d+)([+-])(\d{2})(\d{2})(\d{2}\.\d+)')


def name_to_coords(objid):
    h, m, s, sign, d, dm, ds = PAT.search(objid).groups()
    ra = 15 * (int(h) + int(m) / 60 + float(s) / 3600)
    dec = int(d) + int(dm) / 60 + float(ds) / 3600
    return ra, (-dec if sign == '-' else dec)


coords = np.array([name_to_coords(n) for n in df['objid']])
df['ra_deg'], df['dec_deg'] = coords[:, 0], coords[:, 1]

# ---- Galactic latitude b = how far above/below the Milky Way's disk (0 = right in it) ----
a, d = np.radians(df['ra_deg']), np.radians(df['dec_deg'])
xyz = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
TO_GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                   [0.4941094279, -0.4448296300, 0.7469822445],
                   [-0.8676661490, -0.1980763734, 0.4559837762]])
g = TO_GAL @ xyz
df['gal_l'] = np.degrees(np.arctan2(g[1], g[0])) % 360
df['gal_b'] = np.degrees(np.arcsin(g[2]))


def sky_dist(ra0, dec0):
    """Distance on the sky (degrees) from every star to one point."""
    r0, d0 = np.radians(ra0), np.radians(dec0)
    c = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(a - r0)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


df['in_mc'] = (sky_dist(80.9, -69.75) < LMC_R) | (sky_dist(13.2, -72.8) < SMC_R)

# ---- Known young regions and nearby galaxies (name, RA, Dec, radius in deg) ----
# These are too spread out for the crowding rule, so we cut them by name.
KNOWN_REGIONS = [
    ('Taurus', 67.0, 25.0, 7),
    ('Perseus', 54.0, 31.5, 3.5),
    ('Orion', 84.0, -3.0, 8),
    ('Lambda Orionis', 83.8, 9.9, 4),
    ('Upper Sco / Ophiuchus', 244.0, -23.0, 8),
    ('Lupus', 237.0, -37.0, 5),
    ('Chamaeleon', 167.0, -77.0, 4),
    ('Corona Australis', 285.5, -37.0, 2),
    ('Andromeda Galaxy (M31)', 10.68, 41.27, 2),
    ('Triangulum Galaxy (M33)', 23.46, 30.66, 1),
]
df['region'] = ''
for name, ra0, dec0, rad in KNOWN_REGIONS:
    hit = (sky_dist(ra0, dec0) < rad) & (df['region'] == '')
    df.loc[hit, 'region'] = name
df['in_known'] = df['region'] != ''

# ---- Count neighbors on the sky ----
pts = np.radians(df[['dec_deg', 'ra_deg']].to_numpy())  # BallTree wants [lat, lon]
is_young = df['young'].to_numpy()
young_tree = BallTree(pts[is_young], metric='haversine')
all_tree = BallTree(pts, metric='haversine')


def count_young(radius_deg):
    n = young_tree.query_radius(pts, r=np.radians(radius_deg), count_only=True)
    return n - is_young.astype(int)  # don't count a star as its own neighbor


def count_all(radius_deg):
    return all_tree.query_radius(pts, r=np.radians(radius_deg), count_only=True) - 1


df['n_young_near'] = count_young(RADIUS_DEG)
df['n_any_near'] = count_all(RADIUS_DEG)
df['in_sfr'] = (~df['young']) & ((df['n_young_near'] >= MIN_YOUNG) |
                                 (df['n_any_near'] >= MIN_CROWD))

df['in_plane'] = df['gal_b'].abs() < PLANE_B

old_clean = ~df['young'] & ~df['evolved']
new_clean = old_clean & ~df['in_sfr'] & ~df['in_mc'] & ~df['in_plane'] & ~df['in_known']
df['clean'] = new_clean  # use this column in the Isolation Forest redo

print(f"K dwarfs: {len(df)}")
print(f"Young (SIMBAD tag or Em*): {df['young'].sum()}")
print(f"Clean before (no young, no giants): {old_clean.sum()}")
print(f"  cut, crowded or young area: {(old_clean & df['in_sfr']).sum()}")
print(f"  cut, Magellanic Clouds: {(old_clean & df['in_mc']).sum()}")
print(f"  cut, Milky Way disk (|b| < {PLANE_B}): {(old_clean & df['in_plane']).sum()}")
print(f"  cut, known young region or galaxy: {(old_clean & df['in_known']).sum()}")
for name in df.loc[old_clean & df['in_known'], 'region'].value_counts().index:
    print(f"      {name}: {(old_clean & (df['region'] == name)).sum()}")
print("  (a star can hit more than one of these)")
print(f"Clean after all cuts: {new_clean.sum()}")
print(f"(radius {RADIUS_DEG} deg, {MIN_YOUNG}+ young neighbors OR {MIN_CROWD}+ neighbors of any kind)\n")

base = old_clean & ~df['in_sfr'] & ~df['in_mc'] & ~df['in_known']
print("Clean stars left for different Milky Way disk cuts:")
for bcut in [0, 2, 5, 10]:
    print(f"  |b| < {bcut:<2} cut: {(base & ~(df['gal_b'].abs() < bcut)).sum()} left")
print()

# ---- How much does the choice of settings matter? ----
print("Crowded/young area cut only, stars left for different settings:")
print("radius (deg) | 2+ nearby | 3+ nearby | 5+ nearby")
for r in [0.25, 0.5, 1.0]:
    ny, na = count_young(r), count_all(r)
    left = [(old_clean & ~((ny >= MIN_YOUNG) | (na >= k))).sum() for k in (2, 3, 5)]
    print(f"    {r:<8} |   {left[0]:<7} |   {left[1]:<7} |   {left[2]}")

# ---- Did it catch the top stars from last time? ----
cols = ['objid', 'Teff', 'chi_24', 'norm_chi24', 'n_any_near', 'gal_b', 'otype']
old_top = df[old_clean].sort_values('chi_24', ascending=False).head(15)
print(f"\nOld top 15 by 24 micron excess: {(~old_top['clean']).sum()} of 15 now cut")
print(old_top[cols + ['in_sfr', 'in_mc', 'in_plane', 'region']].round(2).to_string())

print("\n--- New top 15 by 24 micron excess (after all cuts) ---")
print(df[new_clean].sort_values('chi_24', ascending=False).head(15)[cols].round(2).to_string())

df.to_csv(r'data\kdwarf_sfr.csv', index=False)
print("\nSaved data\\kdwarf_sfr.csv")

# ---- Sky map ----
cut_sfr = old_clean & df['in_sfr']
cut_mc = old_clean & ~df['in_sfr'] & df['in_mc']
cut_plane = old_clean & ~df['in_sfr'] & ~df['in_mc'] & df['in_plane']
cut_known = old_clean & ~df['in_sfr'] & ~df['in_mc'] & ~df['in_plane'] & df['in_known']
fig, ax = plt.subplots(figsize=(10, 5))

# Milky Way disk line (b = 0), turned back into RA/Dec
lg = np.radians(np.linspace(0, 360, 721))
eq = TO_GAL.T @ np.stack([np.cos(lg), np.sin(lg), np.zeros_like(lg)])
ra_line = np.degrees(np.arctan2(eq[1], eq[0])) % 360
order = np.argsort(ra_line)
ax.plot(ra_line[order], np.degrees(np.arcsin(eq[2]))[order], c='gray', lw=1, ls='--',
        label='Milky Way disk (b = 0)')

ax.scatter(df.loc[df['young'], 'ra_deg'], df.loc[df['young'], 'dec_deg'], s=6, c='tab:red',
           alpha=0.4, label=f"Young ({df['young'].sum()})")
ax.scatter(df.loc[cut_sfr, 'ra_deg'], df.loc[cut_sfr, 'dec_deg'], s=14, c='tab:orange', marker='x',
           label=f"Cut: crowded or young area ({cut_sfr.sum()})")
ax.scatter(df.loc[cut_mc, 'ra_deg'], df.loc[cut_mc, 'dec_deg'], s=14, c='tab:purple', marker='x',
           label=f"Cut: Magellanic Clouds ({cut_mc.sum()})")
ax.scatter(df.loc[cut_plane, 'ra_deg'], df.loc[cut_plane, 'dec_deg'], s=14, c='black', marker='x',
           label=f"Cut: Milky Way disk ({cut_plane.sum()})")
ax.scatter(df.loc[cut_known, 'ra_deg'], df.loc[cut_known, 'dec_deg'], s=14, c='tab:green', marker='x',
           label=f"Cut: known young region or galaxy ({cut_known.sum()})")
ax.scatter(df.loc[new_clean, 'ra_deg'], df.loc[new_clean, 'dec_deg'], s=10, c='tab:blue',
           label=f'Kept ({new_clean.sum()})')
ax.set_xlim(360, 0)  # astronomers draw RA increasing to the left
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
ax.set_title('Cutting young, crowded, and Milky Way disk stars')
ax.legend(loc='lower left', fontsize=7)
fig.tight_layout()
fig.savefig(r'figures\sfr_cut.png', dpi=150)
print("Saved figures\\sfr_cut.png")