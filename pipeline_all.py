"""
pipeline_all.py: run our debris disk pipeline on the whole SEIP file (all star types),
not just the 1,180 K stars, and list stars that look like J2133+83.

Same steps as before, generalized:
  1. Main sequence only (Gaia): M_G within 1 mag of the dwarf sequence (Mamajek table),
     parallax S/N >= 5. This replaces the SIMBAD giant cut.
  2. Region cuts from sfr_cut.py: Milky Way disk (|b| < 5), Magellanic Clouds,
     known young regions / M31 / M33, and crowding (3+ sample stars within 0.5 deg).
     No SIMBAD young-star tags here (CDS was down); candidates get checked by hand.
  3. Predict each star at 4.5, 5.8, 8 and 24 um from 3.6 um (Rayleigh-Jeans), calibrated
     on plain stars in each temperature group (same plain-star rule as excess_shape_v2.py).
  4. Excess cut: 24 um at least 3 sigma above the star and at least 1.2x the star.
  5. Dust temperature from the 8/24 um shape when 8 um shows extra light at 3 sigma,
     and a rough dust brightness f = L_dust / L_star from the 24 um excess.
  6. "Like J2133+83" = real 24 um excess, warm dust seen at 8 um, f >= 0.1%.

Run:  python pipeline_all.py SEIP_data.csv pipeline_all_out
      (set CROWD=1000000 first to turn the crowding cut off and see what it removes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# Mamajek dwarf sequence (Bp-Rp, M_G), B9V to M6V
MAM = np.array([
    (-0.120, 0.515), (-0.087, 0.615), (-0.037, 1.00), (0.005, 1.16), (0.068, 1.345), (0.110, 1.69),
    (0.166, 1.92), (0.194, 1.98), (0.222, 2.09), (0.263, 2.19), (0.320, 2.27), (0.327, 2.37),
    (0.377, 2.51), (0.434, 2.69), (0.490, 2.89), (0.518, 2.99), (0.546, 3.10), (0.587, 3.26),
    (0.640, 3.56), (0.670, 3.66), (0.694, 3.90), (0.719, 4.105), (0.767, 4.195), (0.784, 4.325),
    (0.803, 4.462), (0.823, 4.635), (0.832, 4.703), (0.841, 4.757), (0.850, 4.801), (0.869, 4.914),
    (0.880, 5.006), (0.900, 5.098), (0.950, 5.34), (0.983, 5.553), (1.01, 5.65), (1.10, 5.83),
    (1.21, 6.20), (1.34, 6.53), (1.43, 6.83), (1.53, 7.02), (1.70, 7.57), (1.73, 7.74),
    (1.79, 8.03), (1.84, 8.16), (1.97, 8.44), (2.09, 8.82), (2.13, 8.98), (2.23, 9.29),
    (2.39, 9.67), (2.50, 10.05), (2.78, 10.87), (2.94, 11.21), (3.16, 12.04), (3.35, 12.45),
    (3.71, 13.35), (4.16, 14.26)])

GROUPS = [('M', 2500, 3930), ('Late K', 3930, 4400), ('Mid K', 4400, 4700), ('Early K', 4700, 5380),
          ('G', 5380, 6000), ('Late F', 6000, 6700), ('Early F', 6700, 7400), ('A', 7400, 10000),
          ('B', 10000, 60000)]
LAM = {'I1': 3.550, 'I2': 4.493, 'I3': 5.731, 'I4': 7.872, 'M1': 23.68}
FLUX = {'I2': ('i2_f_ap2', 'i2_df_ap2'), 'I3': ('i3_f_ap2', 'i3_df_ap2'),
        'I4': ('i4_f_ap2', 'i4_df_ap2'), 'M1': ('m1_f_psf', 'm1_df_psf')}
CAL_ERR = {'I2': 0.03, 'I3': 0.03, 'I4': 0.03, 'M1': 0.04}
SEE_MIN, CAL_SN, MIN_PLAIN = 3, 5, 5
EXCESS_SIG, EXCESS_RATIO = 3, 1.2
h, c, kB = 6.626e-34, 2.998e8, 1.381e-23
J2133 = 'SSTSL2 J213307.73+830501.2'


def bnu(lam_um, T):
    nu = c / (np.asarray(lam_um) * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (kB * np.asarray(T, float)))


df = pd.read_csv(SRC, low_memory=False)
n0 = len(df)

# ---- 1. main sequence (Gaia) ----
ag = df['AG'].fillna(0.0)
ebr = df['E(BP-RP)'].fillna(0.0)
df['MG'] = df['Gmag'] + 5 * np.log10(df['Plx'] / 100.0) - ag
df['bprp0'] = df['BP-RP'] - ebr
df['MG_ms'] = np.interp(df['bprp0'], MAM[:, 0], MAM[:, 1], left=np.nan, right=np.nan)
df['dMG'] = df['MG_ms'] - df['MG']               # > 0 means brighter than (above) the sequence
good_plx = (df['Plx'] / df['e_Plx']) >= 5
df['on_ms'] = good_plx & df['dMG'].between(-1.0, 1.0) & df['Teff'].notna()

# ---- 2. region cuts (sfr_cut.py rules) ----
ra, dec = np.radians(df['ra']), np.radians(df['dec'])
def sky_dist(ra0, dec0):
    r0, d0 = np.radians(ra0), np.radians(dec0)
    cc = np.sin(dec) * np.sin(d0) + np.cos(dec) * np.cos(d0) * np.cos(ra - r0)
    return np.degrees(np.arccos(np.clip(cc, -1, 1)))
df['in_mc'] = (sky_dist(80.9, -69.75) < 5) | (sky_dist(13.2, -72.8) < 4)
KNOWN = [('Taurus', 67.0, 25.0, 7), ('Perseus', 54.0, 31.5, 3.5), ('Orion', 84.0, -3.0, 8),
         ('Lambda Orionis', 83.8, 9.9, 4), ('Upper Sco / Ophiuchus', 244.0, -23.0, 8),
         ('Lupus', 237.0, -37.0, 5), ('Chamaeleon', 167.0, -77.0, 4), ('Chamaeleon II / III', 195.0, -78.0, 3),
         ('Mon R2', 91.9, -6.4, 3), ('Magellanic Bridge', 32.0, -74.0, 5), ('Corona Australis', 285.5, -37.0, 2),
         ('M31', 10.68, 41.27, 2), ('M33', 23.46, 30.66, 1)]
df['region'] = ''
for nm, r0, d0, rad in KNOWN:
    hit = (sky_dist(r0, d0) < rad) & (df['region'] == '')
    df.loc[hit, 'region'] = nm
df['in_plane'] = df['b'].abs() < 5
ms = df[df['on_ms']].copy()
pts = np.radians(ms[['dec', 'ra']].to_numpy())
tree = BallTree(pts, metric='haversine')
ms['n_near'] = tree.query_radius(pts, r=np.radians(0.5), count_only=True) - 1
df['n_near'] = ms['n_near']
CROWD = int(__import__('os').environ.get('CROWD', 3))
df['crowded_area'] = df['n_near'].fillna(0) >= CROWD
df['clean'] = df['on_ms'] & ~df['in_mc'] & ~df['in_plane'] & (df['region'] == '') & ~df['crowded_area']

# ---- 3. calibration per temperature group ----
d = df[df['clean']].copy()
d['group'] = ''
for g, lo, hi in GROUPS:
    d.loc[(d['Teff'] >= lo) & (d['Teff'] < hi), 'group'] = g
d = d[d['group'] != ''].copy()
for b, (f, _) in FLUX.items():
    d['rj_' + b] = d['i1_f_ap2'] * (LAM['I1'] / LAM[b])**2
    d['r_' + b] = d[f] / d['rj_' + b]


def plain_stars(s):
    c24 = 1.35
    plain = s.iloc[:0]
    for _ in range(30):
        seen = s[s['rj_M1'] * c24 / s['m1_df_psf'] >= CAL_SN]
        plain = seen
        for _ in range(30):
            med = plain['r_M1'].median()
            sig = 1.4826 * (plain['r_M1'] - med).abs().median()
            keep = seen[(seen['r_M1'] - med).abs() < 3 * sig]
            if len(keep) == len(plain):
                break
            plain = keep
        if len(plain) == 0 or abs(plain['r_M1'].median() - c24) < 1e-4:
            break
        c24 = plain['r_M1'].median()
    return plain


def calibration(p):
    out = {'n': len(p)}
    for b in FLUX:
        v = p['r_' + b].dropna()
        out[b] = v.median()
        out['sd_' + b] = 1.4826 * (v - v.median()).abs().median() / v.median()
    return out


cal, cal_rows = {}, []
for g, lo, hi in GROUPS:
    s = d[d['group'] == g]
    p = plain_stars(s) if len(s) else s
    cal[g] = calibration(p) if len(p) >= MIN_PLAIN else None
    cal_rows.append({'group': g, 'teff': f'{lo}-{hi}', 'clean_stars': len(s), 'plain_stars': len(p),
                     **({k: round(v, 3) for k, v in cal[g].items() if k != 'n'} if cal[g] else {})})
# groups with too few plain stars borrow the nearest group that has a calibration
names = [g for g, _, _ in GROUPS]
for i, g in enumerate(names):
    if cal[g] is None:
        for j in sorted(range(len(names)), key=lambda j: abs(j - i)):
            if cal[names[j]] is not None:
                cal[g] = dict(cal[names[j]], borrowed=names[j])
                break
pd.DataFrame(cal_rows).to_csv(OUT / 'calibration_by_group.csv', index=False)

for b, (f, e) in FLUX.items():
    corr = d['group'].map({g: cal[g][b] for g in names}).astype(float)
    frac = d['group'].map({g: max(cal[g]['sd_' + b], 0.02) for g in names}).astype(float)
    d['pred_' + b] = d['rj_' + b] * corr
    d['ratio_' + b] = d[f] / d['pred_' + b]
    d['ex_' + b] = d[f] - d['pred_' + b]
    d['eex_' + b] = np.sqrt(d[e]**2 + (CAL_ERR[b] * d[f])**2 + (frac * d['pred_' + b])**2)
    d['exsig_' + b] = d['ex_' + b] / d['eex_' + b]
d['comparable'] = d['pred_M1'] / d['m1_df_psf'] >= SEE_MIN
d['excess'] = (d['exsig_M1'] >= EXCESS_SIG) & (d['ratio_M1'] >= EXCESS_RATIO)

# ---- 5. dust temperature from the 8/24 shape, and rough f ----
Tg = np.arange(60, 1501, 1.0)
s8 = (bnu(LAM['I4'], Tg) / LAM['I4']) / (bnu(LAM['M1'], Tg) / LAM['M1'])
d['shape_I4'] = (d['ex_I4'] / LAM['I4']) / (d['ex_M1'] / LAM['M1'])
d['eshape_I4'] = (d['eex_I4'] / LAM['I4']) / (d['ex_M1'] / LAM['M1'])
d['warm_8um'] = d['excess'] & (d['ex_I4'] / d['eex_I4'] >= 3)
d['T_dust'] = np.where(d['warm_8um'], np.interp(d['shape_I4'].clip(s8.min(), s8.max()), s8, Tg), np.nan)
T_for_f = d['T_dust'].fillna(150.0)        # stars with no 8 um excess: assume 150 K dust for f
d['f_dust'] = np.where(d['excess'], (d['ex_M1'] / d['pred_M1']) * bnu(LAM['M1'], d['Teff']) /
                       bnu(LAM['M1'], T_for_f) * (T_for_f / d['Teff'])**4, np.nan)
# like J2133+83: Spitzer can see the star itself at 24 um (so the excess isn't just
# "something faint is there"), warm dust at 8 um, and f >= 0.1%
d['like_J2133'] = d['comparable'] & d['warm_8um'] & (d['f_dust'] >= 1e-3)

keep = ['objid', 'group', 'Teff', 'logg', 'Plx', 'Gmag', 'bprp0', 'MG', 'dMG', 'RUWE', 'b', 'n_near',
        'm1_f_psf', 'pred_M1', 'ratio_M1', 'exsig_M1', 'ratio_I4', 'exsig_I4', 'warm_8um', 'T_dust',
        'f_dust', 'comparable', 'like_J2133']
d.to_csv(OUT / 'all_clean_stars.csv', index=False)
d[d['excess']].sort_values('f_dust', ascending=False)[keep].to_csv(OUT / 'excess_stars.csv', index=False)
d[d['like_J2133']].sort_values('f_dust', ascending=False)[keep].to_csv(OUT / 'like_J2133.csv', index=False)

# ---- summary ----
print(f'SEIP sources: {n0}')
print(f'Main sequence with good parallax: {int(df["on_ms"].sum())}')
cuts = df['on_ms']
print(f'  cut Milky Way disk |b|<5: {int((cuts & df["in_plane"]).sum())}, Magellanic Clouds: {int((cuts & df["in_mc"]).sum())}, '
      f'known regions: {int((cuts & (df["region"] != "")).sum())}, crowded: {int((cuts & df["crowded_area"]).sum())}')
print(f'Clean main sequence stars with a temperature group: {len(d)}')
print(pd.DataFrame(cal_rows).to_string(index=False))
d['excess_comp'] = d['excess'] & d['comparable']
tab = d.groupby('group', sort=False).agg(clean=('objid', 'size'), comparable=('comparable', 'sum'),
                                         excess=('excess', 'sum'), excess_comparable=('excess_comp', 'sum'),
                                         warm=('warm_8um', 'sum'), like_J2133=('like_J2133', 'sum'))
print(tab.reindex(names).dropna().astype(int).to_string())
print(f'\nComparable (star seen alone at 24 um): {int(d["comparable"].sum())}; comparable with excess: {int(d["excess_comp"].sum())}')
print(f'Total with a real 24 um excess: {int(d["excess"].sum())}; warm (8 um too): {int(d["warm_8um"].sum())}; '
      f'like J2133+83 (warm and f >= 0.1%): {int(d["like_J2133"].sum())}')
j = d[d['objid'] == J2133]
if len(j):
    r = j.iloc[0]
    print(f'J2133+83 check: ratio_M1 {r.ratio_M1:.2f}, exsig {r.exsig_M1:.1f}, T_dust {r.T_dust:.0f} K, f {r.f_dust*100:.2f}%')
pd.set_option('display.width', 250)
print(d[d['like_J2133']].sort_values('f_dust', ascending=False)[keep].round(3).head(40).to_string(index=False))
