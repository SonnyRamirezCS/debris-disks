"""
Size and shape of the 24 micron excess (version 1), for the December ML talk.

Why: the Isolation Forest in isoforest_v2.py uses chi values. Chi says how
SIGNIFICANT an excess is, and that mostly tracks how bright the star is (bright
star = tiny error bars = huge chi). So it picks J2133+83 only about 1 time in 4.
This script uses how BIG the excess is (measured / predicted star-only flux)
and then what SHAPE it has across 4.5, 5.8, 8 and 24 microns.

Step 1  Predict each star's own light at 4.5, 5.8, 8, 24 um from its 3.6 um flux,
        calibrated on the plain stars of its own temperature group (Early, Mid,
        Late K). Plain = Spitzer sees the star at 24 um with S/N >= 5, then a
        3 sigma clip on the 24 um ratio (same rule as j2133_numbers.py).
Part A  SIZE. Stars where Spitzer can see the star alone at 24 um (the
        "comparable" stars). Features = log10(measured / predicted) at 4.5, 8,
        24 um, in units of the plain stars' scatter -> Isolation Forest, plus
        the same 50-seed check as check_rerun.py.
Part B  SHAPE. Every clean star with a real 24 um excess (>= 3 sigma and
        >= 1.2x, the usual excess cut). shape_b = extra light at band b divided
        by extra light at 24 um, both in nu*F_nu (energy). Dust at one
        temperature T gives a known shape, so shape at 8 um -> a dust temperature.
        KMeans groups the stars whose excess looks alike.

Run from the debris-disks folder:  python excess_shape.py
Needs data/kdwarf_sfr.csv (made by sfr_cut.py)
Makes data/excess_shape.csv and figures/excess_shape.png
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # quiets a harmless KMeans warning on Windows

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
DATA, FIGS = HERE / 'data', HERE / 'figures'

# ---- Settings ----
TEFF_MIN, TEFF_MAX, EDGES = 3930, 5380, [4400, 4700]
GROUPS = ['Late K', 'Mid K', 'Early K']
SEE_MIN = 3          # star-only 24 um flux >= 3x its error = Spitzer can see the star alone
CAL_SN = 5           # calibration stars need the star seen at >= 5x the error
MIN_PLAIN = 5        # a group with fewer plain stars than this uses the all-K calibration
EXCESS_SIG, EXCESS_RATIO = 3, 1.2
CONTAMINATION = 0.05
SEEDS = 50
K_RANGE = range(2, 5)
NAMED = {'SSTSL2 J213307.73+830501.2': 'J2133+83', 'SSTSL2 J084517.41-785937.1': 'J0845-78'}

LAM = {'I1': 3.550, 'I2': 4.493, 'I3': 5.731, 'I4': 7.872, 'M1': 23.68}
FLUX = {'I2': ('i2_f_ap2', 'i2_df_ap2'), 'I3': ('i3_f_ap2', 'i3_df_ap2'),
        'I4': ('i4_f_ap2', 'i4_df_ap2'), 'M1': ('m1_f_psf', 'm1_df_psf')}
BANDS = list(FLUX)
CAL_ERR = {'I2': 0.03, 'I3': 0.03, 'I4': 0.03, 'M1': 0.04}   # absolute calibration
h, c, k_B = 6.626e-34, 2.998e8, 1.381e-23

# ---- Load the clean K stars ----
df = pd.read_csv(DATA / 'kdwarf_sfr.csv').copy()
df = df[df['clean'].astype(str).str.lower().isin(['true', '1', '1.0'])]
df = df[(df['Teff'] > TEFF_MIN) & (df['Teff'] < TEFF_MAX)].copy()
df['group'] = pd.cut(df['Teff'], [TEFF_MIN] + EDGES + [TEFF_MAX], right=False, labels=GROUPS)
for b, (f, _) in FLUX.items():
    df['rj_' + b] = df['i1_f_ap2'] * (LAM['I1'] / LAM[b])**2       # Rayleigh-Jeans from 3.6 um
    df['r_' + b] = df[f] / df['rj_' + b]                           # measured / RJ
print(f'Clean K stars: {len(df)}')


# ---- Step 1: calibration per temperature group ----
def plain_stars(s):
    """Plain stars: star seen at 24 um with S/N >= CAL_SN, then 3-sigma clip on the 24 um ratio."""
    c24 = 1.35
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
        if abs(plain['r_M1'].median() - c24) < 1e-4:
            break
        c24 = plain['r_M1'].median()
    return plain


def calibration(plain):
    out = {}
    for b in BANDS:
        v = plain['r_' + b].dropna()
        out[b] = v.median()
        out['sd_' + b] = 1.4826 * (v - v.median()).abs().median() / v.median()   # fractional scatter
    out['n'] = len(plain)
    return out


all_plain = plain_stars(df)
cal_all = calibration(all_plain)
cal = {}
print('\n--- Calibration (measured / Rayleigh-Jeans from 3.6 um, plain stars) ---')
print(f"{'group':8s} {'plain':>5s} " + ' '.join(f'{LAM[b]:>6.1f}' for b in BANDS))
for g in GROUPS:
    p = all_plain[all_plain['group'] == g]
    cal[g] = calibration(p) if len(p) >= MIN_PLAIN else cal_all
    note = '' if len(p) >= MIN_PLAIN else f'  (only {len(p)} plain stars, using all K)'
    print(f"{g:8s} {len(p):5d} " + ' '.join(f'{cal[g][b]:6.3f}' for b in BANDS) + note)
print(f"{'all K':8s} {cal_all['n']:5d} " + ' '.join(f'{cal_all[b]:6.3f}' for b in BANDS))

for b, (f, e) in FLUX.items():
    corr = df['group'].map({g: cal[g][b] for g in GROUPS}).astype(float)
    frac = df['group'].map({g: max(cal[g]['sd_' + b], 0.02) for g in GROUPS}).astype(float)
    df['pred_' + b] = df['rj_' + b] * corr                        # star-only prediction (uJy)
    df['ratio_' + b] = df[f] / df['pred_' + b]
    df['ex_' + b] = df[f] - df['pred_' + b]                       # extra light (uJy)
    df['eex_' + b] = np.sqrt(df[e]**2 + (CAL_ERR[b] * df[f])**2 + (frac * df['pred_' + b])**2)
    df['exsig_' + b] = df['ex_' + b] / df['eex_' + b]
df['comparable'] = df['pred_M1'] / df['m1_df_psf'] >= SEE_MIN
df['name'] = df['objid'].map(NAMED).fillna(df['objid'].str.replace('SSTSL2 ', '', regex=False))

# ======================================================================
# Part A: SIZE of the excess, comparable stars only
# ======================================================================
A_FEAT = ['z_I2', 'z_I4', 'z_M1']
a = df[df['comparable']].copy()
for b in ['I2', 'I4', 'M1']:
    lr = np.log10(a['ratio_' + b])
    sd = np.log10(1 + max(cal_all['sd_' + b], 0.02))    # plain-star scatter in log units
    a['z_' + b] = lr / sd                                # 0 = normal star for its group
a = a.dropna(subset=A_FEAT)

iso = IsolationForest(contamination=CONTAMINATION, random_state=42).fit(a[A_FEAT])
a['weirdness'] = -iso.score_samples(a[A_FEAT])
a['flagged'] = iso.predict(a[A_FEAT]) == -1
a['picked'] = a['flagged'] & (a['exsig_M1'] >= EXCESS_SIG) & (a['ratio_M1'] >= EXCESS_RATIO)

print(f'\n========== Part A: size of the excess ({len(a)} comparable stars) ==========')
print(f"Isolation Forest flagged {int(a['flagged'].sum())}, "
      f"{int(a['picked'].sum())} with a real 24 um excess (>= {EXCESS_SIG} sigma and >= {EXCESS_RATIO}x):")
cols = ['name', 'group', 'Teff', 'ratio_I2', 'ratio_I4', 'ratio_M1', 'exsig_M1', 'chi_24', 'weirdness']
print(a[a['flagged']].sort_values('weirdness', ascending=False)[cols].round(2).to_string(index=False))

print(f'\nHow often each star gets flagged over {SEEDS} random seeds:')
OLD_FEAT = ['norm_chi24', 'norm_chi8', 'norm_chi45']
runs = [('chi features, all clean stars (isoforest_v2 Run B)', df, OLD_FEAT),
        ('chi features, comparable stars only', a, OLD_FEAT),
        ('size features, comparable stars only (this script)', a, A_FEAT)]
for label, sample, feats in runs:
    s = sample.dropna(subset=feats)
    print(f'  {label}, {len(s)} stars:')
    for objid, nm in NAMED.items():
        if objid not in set(s['objid']):
            print(f'    {nm}: not in this sample')
            continue
        hit, ranks = 0, []
        for seed in range(SEEDS):
            m = IsolationForest(contamination=CONTAMINATION, random_state=seed).fit(s[feats])
            w = -m.score_samples(s[feats])
            jw = w[(s['objid'] == objid).to_numpy()][0]
            ranks.append(int((w > jw).sum()) + 1)
            hit += int(m.predict(s.loc[s['objid'] == objid, feats])[0] == -1)
        print(f'    {nm}: flagged {hit}/{SEEDS}, median rank {int(np.median(ranks))}')


# ======================================================================
# Part B: SHAPE of the excess, clean stars with a real 24 um excess
# ======================================================================
def bnu(lam_um, T):
    nu = c / (lam_um * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k_B * T))


def bb_shape(b, T):
    """nu*F_nu at band b over nu*F_nu at 24 um for dust at temperature T."""
    return (bnu(LAM[b], T) / LAM[b]) / (bnu(LAM['M1'], T) / LAM['M1'])


bsel = (df['exsig_M1'] >= EXCESS_SIG) & (df['ratio_M1'] >= EXCESS_RATIO)
B = df[bsel].copy()
for b in ['I2', 'I3', 'I4']:
    B['shape_' + b] = (B['ex_' + b] / LAM[b]) / (B['ex_M1'] / LAM['M1'])
    B['eshape_' + b] = (B['eex_' + b] / LAM[b]) / (B['ex_M1'] / LAM['M1'])

# dust temperature from the 8/24 shape (a 2-sigma upper limit when 8 um shows nothing)
Tgrid = np.arange(60, 1500, 1.0)
s8grid = bb_shape('I4', Tgrid)


def temp_from_shape(s):
    return float(np.interp(s, s8grid, Tgrid)) if s > s8grid[0] else np.nan


B['warm_8um'] = B['shape_I4'] / B['eshape_I4'] >= 3
B['T_dust'] = [temp_from_shape(s) if w else np.nan for s, w in zip(B['shape_I4'], B['warm_8um'])]
B['T_dust_max'] = [np.nan if w else temp_from_shape(s + 2 * e)
                   for s, e, w in zip(B['shape_I4'], B['eshape_I4'], B['warm_8um'])]

B_FEAT = ['f_shape_I2', 'f_shape_I4', 'f_size']
B['f_shape_I2'] = B['shape_I2'].clip(-1, 3)
B['f_shape_I4'] = B['shape_I4'].clip(-1, 3)
B['f_size'] = np.log10(B['ratio_M1'])
X = (B[B_FEAT] - B[B_FEAT].median()) / (1.4826 * (B[B_FEAT] - B[B_FEAT].median()).abs().median())
scores = {}
for kk in K_RANGE:
    lab = KMeans(n_clusters=kk, n_init=20, random_state=0).fit_predict(X)
    scores[kk] = silhouette_score(X, lab)
best_k = max(scores, key=scores.get)
km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
# number the groups by median 24 um size, smallest first, so the labels are stable
order = B.assign(c=km.labels_).groupby('c')['ratio_M1'].median().sort_values().index
B['shape_group'] = pd.Series(km.labels_, index=B.index).map({o: i + 1 for i, o in enumerate(order)})

print(f'\n========== Part B: shape of the excess ({len(B)} stars with a real 24 um excess) ==========')
print(f"Spitzer can see the star alone for {int(B['comparable'].sum())} of them; "
      f"for the other {int((~B['comparable']).sum())} the 24 um light is almost all excess.")
print(f"8 um excess >= 3 sigma (warm material): {int(B['warm_8um'].sum())}")
print('Blackbody shape at 8 um (nu*F_nu vs 24 um): ' +
      ', '.join(f'{T} K = {bb_shape("I4", T):.2f}' for T in (150, 200, 250, 300, 400)))
print('KMeans silhouette by number of groups: ' + ', '.join(f'{kk}: {v:.2f}' for kk, v in scores.items()) +
      f'  -> {best_k} groups')

summ = B.groupby('shape_group').agg(
    stars=('objid', 'size'),
    comparable=('comparable', 'sum'),
    median_24um_ratio=('ratio_M1', 'median'),
    median_shape_8=('shape_I4', 'median'),
    median_shape_45=('shape_I2', 'median'),
    warm_8um=('warm_8um', 'sum'),
    median_Teff=('Teff', 'median'))
print(summ.round(2).to_string())

print('\nNamed stars:')
for objid, nm in NAMED.items():
    r = B[B['objid'] == objid]
    if r.empty:
        print(f'  {nm}: no real 24 um excess in this sample')
        continue
    r = r.iloc[0]
    T = (f"about {r['T_dust']:.0f} K" if np.isfinite(r['T_dust'])
         else f"below about {r['T_dust_max']:.0f} K" if np.isfinite(r['T_dust_max']) else 'cold, under ~150 K')
    print(f"  {nm}: shape group {r['shape_group']}, 24 um ratio {r['ratio_M1']:.1f}x, "
          f"shape at 8 um {r['shape_I4']:.2f} +- {r['eshape_I4']:.2f} -> dust {T}")

# ---- save ----
keep = ['objid', 'name', 'group', 'Teff', 'comparable'] + \
       [f'{p}_{b}' for b in BANDS for p in ('pred', 'ratio', 'ex', 'exsig')]
out = df[keep].merge(a[['objid', 'weirdness', 'flagged', 'picked']], on='objid', how='left').merge(
    B[['objid', 'shape_I2', 'shape_I3', 'shape_I4', 'eshape_I4', 'warm_8um', 'T_dust', 'T_dust_max',
       'shape_group']], on='objid', how='left')
out.to_csv(DATA / 'excess_shape.csv', index=False)
print('\nSaved data/excess_shape.csv')

# ======================================================================
# Figure
# ======================================================================
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e4e3df', '#a3a29d'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']      # fixed order, never cycled
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# A: size at 8 um vs size at 24 um, comparable stars
norm = a[~a['flagged']]
ax1.scatter(norm['ratio_M1'], norm['ratio_I4'], s=36, color=GRAY, edgecolors='white', lw=1,
            label=f'Normal ({len(norm)})', zorder=2)
fl = a[a['flagged']]
ax1.scatter(fl['ratio_M1'], fl['ratio_I4'], s=56, color=SERIES[0], edgecolors='white', lw=1,
            label=f'Flagged by Isolation Forest ({len(fl)})', zorder=3)
ax1.axvline(EXCESS_RATIO, color=INK2, lw=0.8, ls=':', zorder=1)
ax1.text(EXCESS_RATIO * 1.03, 0.995, f'{EXCESS_RATIO}x', color=INK2, fontsize=8, va='top',
         transform=ax1.get_xaxis_transform())
ax1.set_xscale('log')
ax1.set_xlabel('24 um: measured / predicted star', color=INK)
ax1.set_ylabel('8 um: measured / predicted star', color=INK)
ax1.set_title(f'A. Size of the excess ({len(a)} comparable stars)', loc='left', fontsize=11, color=INK)

# B: shape at 8 um vs size at 24 um, all stars with an excess
for i, g in enumerate(sorted(B['shape_group'].unique())):
    s = B[B['shape_group'] == g]
    ax2.scatter(s['ratio_M1'], s['shape_I4'].clip(-1, 3), s=30, color=SERIES[i], edgecolors='white',
                lw=0.8, label=f'Shape group {g} ({len(s)})', zorder=3)
comp = B[B['comparable']]
ax2.scatter(comp['ratio_M1'], comp['shape_I4'].clip(-1, 3), s=110, facecolors='none',
            edgecolors=INK, lw=1, label='Spitzer sees the star alone', zorder=4)
for T in (200, 250, 300):
    ax2.axhline(bb_shape('I4', T), color=GRID, lw=1, zorder=1)
    ax2.text(1.0, bb_shape('I4', T), f' {T} K dust', color=INK2, fontsize=8, va='center',
             transform=ax2.get_yaxis_transform())
ax2.axhline(0, color=GRAY, lw=0.8, zorder=1)
ax2.set_xscale('log')
ax2.set_xlabel('24 um: measured / predicted star', color=INK)
ax2.set_ylabel('Extra light at 8 um / extra light at 24 um (energy)', color=INK)
ax2.set_title(f'B. Shape of the excess ({len(B)} stars with a real 24 um excess)', loc='left',
              fontsize=11, color=INK)

for ax, tab in ((ax1, a), (ax2, B)):
    ycol = 'ratio_I4' if ax is ax1 else 'shape_I4'
    for objid, nm in NAMED.items():
        r = tab[tab['objid'] == objid]
        if len(r):
            y = r[ycol].iloc[0] if ax is ax1 else min(max(r[ycol].iloc[0], -1), 3)
            ax.annotate(nm, (r['ratio_M1'].iloc[0], y), xytext=(8, 6), textcoords='offset points',
                        fontsize=9, color=INK, fontweight='bold')
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3)
    ticks = [0.8, 1, 1.5, 2, 3, 5, 10] if ax is ax1 else [2, 5, 10, 20, 50, 100]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}x'))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=INK2)

fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / 'excess_shape.png', dpi=150)
print('Saved figures/excess_shape.png')
