"""
Size and shape of the 24 micron excess, version 2 (two fixes from the fake disk test).

excess_shape.py (version 1) is unchanged. This copy makes two changes, both marked
"v2:" below, because fake_disks.py showed two problems with version 1:

Fix 1, Part A: how the Isolation Forest decides a star is unusual.
  v1 flagged the weirdest 5% of stars, about 3 of 59. J2133+83 and J0845-78
  already take 2 of those spots, so another real disk had to beat the 3rd
  weirdest star to get picked. In the fake test the ML needed disks 3 to 10x
  brighter than the excess cut alone did.
  v2 flags a star when its weirdness score is above 0.5. That's the cutoff from
  the paper that introduced the Isolation Forest (Liu, Ting & Zhou 2008), and
  sklearn's contamination='auto'. Now the number of flagged stars isn't fixed.

Fix 2, Part B: drop the 4.5 um shape from the KMeans features.
  Dust cooler than about 500 K adds almost no light at 4.5 um. So in v1 a star
  landed in the "warm" group mostly by having a 4.5 um bump, which is usually
  noise. Copies of J2133+83's own disk landed in the warm group only 5 of 51
  times. v2 groups stars on the 8 um shape and the 24 um size only, so the
  groups split by dust temperature.

Everything else (calibration, features, excess cut, dust temperature) is the
same as version 1.

Run from the debris-disks folder:  python excess_shape_v2.py
Needs data/kdwarf_sfr.csv (made by sfr_cut.py)
Uses, if they're there: data/excess_shape.csv (to compare with v1) and
  vetting/results/batch_vet.csv + control_cutoff.csv (image check of the picks)
Makes data/excess_shape_v2.csv and figures/excess_shape_v2.png
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # quiets a harmless KMeans warning on Windows

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
DATA, FIGS = HERE / 'data', HERE / 'figures'
VET = HERE / 'vetting' / 'results'

# ---- Settings ----
TEFF_MIN, TEFF_MAX, EDGES = 3930, 5380, [4400, 4700]
GROUPS = ['Late K', 'Mid K', 'Early K']
SEE_MIN = 3          # star-only 24 um flux >= 3x its error = Spitzer can see the star alone
CAL_SN = 5           # calibration stars need the star seen at >= 5x the error
MIN_PLAIN = 5        # a group with fewer plain stars than this uses the all-K calibration
EXCESS_SIG, EXCESS_RATIO = 3, 1.2
WEIRD_CUT = 0.5      # v2: flag when weirdness > 0.5 (v1 flagged the weirdest 5%)
CONTAMINATION = 0.05 # v1's rule, only used here to compare the two
SEEDS = 50
K_RANGE = range(2, 5)
NAMED = {'SSTSL2 J213307.73+830501.2': 'J2133+83', 'SSTSL2 J084517.41-785937.1': 'J0845-78'}
J2133 = 'SSTSL2 J213307.73+830501.2'

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


# ---- Step 1: calibration per temperature group (same as v1) ----
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
for g in GROUPS:
    p = all_plain[all_plain['group'] == g]
    cal[g] = calibration(p) if len(p) >= MIN_PLAIN else cal_all

for b, (f, e) in FLUX.items():
    corr = df['group'].map({g: cal[g][b] for g in GROUPS}).astype(float)
    frac = df['group'].map({g: max(cal[g]['sd_' + b], 0.02) for g in GROUPS}).astype(float)
    df['pred_' + b] = df['rj_' + b] * corr                        # star-only prediction (uJy)
    df['ratio_' + b] = df[f] / df['pred_' + b]
    df['ex_' + b] = df[f] - df['pred_' + b]                       # extra light (uJy)
    df['eex_' + b] = np.sqrt(df[e]**2 + (CAL_ERR[b] * df[f])**2 + (frac * df['pred_' + b])**2)
    df['exsig_' + b] = df['ex_' + b] / df['eex_' + b]
df['comparable'] = df['pred_M1'] / df['m1_df_psf'] >= SEE_MIN
df['cut'] = (df['exsig_M1'] >= EXCESS_SIG) & (df['ratio_M1'] >= EXCESS_RATIO)
df['name'] = df['objid'].map(NAMED).fillna(df['objid'].str.replace('SSTSL2 ', '', regex=False))

# ---- image check from vetting/batch_vet.py, with the cutoff from vetting/control_cutoff.py ----
vet, vet_cut = None, None
if (VET / 'batch_vet.csv').exists() and (VET / 'control_cutoff.csv').exists():
    vet = pd.read_csv(VET / 'batch_vet.csv').set_index('objid')
    cc = pd.read_csv(VET / 'control_cutoff.csv')
    vet_cut = float(cc.loc[cc['cutoff'] == '1 in 10', 'cutoff_value'].iloc[0])


def image_check(objid):
    if vet is None or objid not in vet.index:
        return 'not in the image check'
    r = vet.loc[objid]
    if r['status'] != 'ok' or not np.isfinite(r['offset_x_noise']):
        return 'image check could not run'
    word = 'OFF-CENTER' if r['offset_x_noise'] > vet_cut else 'centered on the star'
    return f"{word} ({r['offset_x_noise']:.1f} vs cutoff {vet_cut:.1f})"


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

iso = IsolationForest(contamination='auto', random_state=42).fit(a[A_FEAT])
a['weirdness'] = -iso.score_samples(a[A_FEAT])
a['flagged'] = a['weirdness'] > WEIRD_CUT                          # v2: score cutoff, not the weirdest 5%
a['picked'] = a['flagged'] & a['cut']

print(f'\n========== Part A: size of the excess ({len(a)} comparable stars) ==========')
print(f"Isolation Forest flagged {int(a['flagged'].sum())} (weirdness > {WEIRD_CUT}), "
      f"{int(a['picked'].sum())} with a real 24 um excess (>= {EXCESS_SIG} sigma and >= {EXCESS_RATIO}x):")
cols = ['name', 'group', 'Teff', 'ratio_I2', 'ratio_I4', 'ratio_M1', 'exsig_M1', 'weirdness', 'cut']
print(a[a['flagged']].sort_values('weirdness', ascending=False)[cols].round(2).to_string(index=False))
print('(cut = passes the excess cut. Flagged stars without it have odd 4.5/8 um light but no 24 um excess,'
      ' so they are not picked)')

# v2: compare the two flag rules over 50 seeds, for every comparable star with a real excess
ex = a[a['cut']]
hit1, hit2 = np.zeros(len(ex)), np.zeros(len(ex))
plain_flags = []
for seed in range(SEEDS):
    m = IsolationForest(contamination=CONTAMINATION, random_state=seed).fit(a[A_FEAT])
    w = -m.score_samples(a[A_FEAT])
    f1 = m.predict(a[A_FEAT]) == -1          # v1 rule (same trees, only the cutoff differs)
    f2 = w > WEIRD_CUT                       # v2 rule
    hit1 += f1[a['cut'].to_numpy()]
    hit2 += f2[a['cut'].to_numpy()]
    plain_flags.append(int((f2 & ~a['cut'].to_numpy()).sum()))
ex = ex.assign(v1_pct=100 * hit1 / SEEDS, v2_pct=100 * hit2 / SEEDS).sort_values('ratio_M1', ascending=False)
print(f'\nThe {len(ex)} comparable stars with a real 24 um excess: % of {SEEDS} seeds where each rule picks them')
print(f"{'star':22s} {'24 um':>6s} {'sigma':>6s} {'v1 pct':>7s} {'v2 pct':>7s}   image check")
for _, r in ex.iterrows():
    print(f"{r['name']:22s} {r['ratio_M1']:5.1f}x {r['exsig_M1']:6.1f} {r['v1_pct']:6.0f}% {r['v2_pct']:6.0f}%   "
          f"{image_check(r['objid'])}")
print(f'v1 (weirdest 5%) picks {hit1.sum() / SEEDS:.1f} of them per seed on average, '
      f'v2 (weirdness > {WEIRD_CUT}) picks {hit2.sum() / SEEDS:.1f}.')
print(f'v2 also flags about {np.median(plain_flags):.0f} stars with no 24 um excess per seed; '
      'the excess cut drops those.')


# ======================================================================
# Part B: SHAPE of the excess, clean stars with a real 24 um excess
# ======================================================================
def bnu(lam_um, T):
    nu = c / (lam_um * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k_B * T))


def bb_shape(b, T):
    """nu*F_nu at band b over nu*F_nu at 24 um for dust at temperature T."""
    return (bnu(LAM[b], T) / LAM[b]) / (bnu(LAM['M1'], T) / LAM['M1'])


B = df[df['cut']].copy()
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

B_FEAT = ['f_shape_I4', 'f_size']                                  # v2: no 4.5 um shape
B['f_shape_I4'] = B['shape_I4'].clip(-1, 3)
B['f_size'] = np.log10(B['ratio_M1'])
medB = B[B_FEAT].median()
madB = 1.4826 * (B[B_FEAT] - medB).abs().median()
X = ((B[B_FEAT] - medB) / madB).to_numpy(float)
scores = {}
for kk in K_RANGE:
    lab = KMeans(n_clusters=kk, n_init=20, random_state=0).fit_predict(X)
    scores[kk] = silhouette_score(X, lab)
best_k = max(scores, key=scores.get)
km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
# number the groups by median 24 um size, smallest first (same rule as v1)
order = B.assign(c=km.labels_).groupby('c')['ratio_M1'].median().sort_values().index
B['shape_group'] = pd.Series(km.labels_, index=B.index).map({o: i + 1 for i, o in enumerate(order)})
warm_label = int(B.assign(c=km.labels_).groupby('c')['shape_I4'].median().idxmax())
B['shape_name'] = np.where(km.labels_ == warm_label, 'warm', 'cool/flat')


def warm_starts_at(size_ratio):
    """For a star of this 24 um size, the 8 um shape where the warm group begins."""
    s = np.linspace(-1, 3, 4001)
    pts = (np.column_stack([s, np.full_like(s, np.log10(size_ratio))]) - medB.to_numpy()) / madB.to_numpy()
    w = np.where(km.predict(pts) == warm_label)[0]
    return s[w.min()] if len(w) else np.nan


print(f'\n========== Part B: shape of the excess ({len(B)} stars with a real 24 um excess) ==========')
print(f"Spitzer can see the star alone for {int(B['comparable'].sum())} of them; "
      f"for the other {int((~B['comparable']).sum())} the 24 um light is almost all excess.")
print(f"8 um excess >= 3 sigma (warm material): {int(B['warm_8um'].sum())}")
print('Blackbody shape at 8 um (nu*F_nu vs 24 um): ' +
      ', '.join(f'{T} K = {bb_shape("I4", T):.2f}' for T in (150, 200, 250, 300, 400)))
print('KMeans silhouette by number of groups: ' + ', '.join(f'{kk}: {v:.2f}' for kk, v in scores.items()) +
      f'  -> {best_k} groups')

summ = B.groupby('shape_group').agg(
    name=('shape_name', 'first'),
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
    line = (f"  {nm}: {r['shape_name']} group ({r['shape_group']}), 24 um ratio {r['ratio_M1']:.1f}x, "
            f"shape at 8 um {r['shape_I4']:.2f} +- {r['eshape_I4']:.2f} -> dust {T}")
    print(line)
    s0 = warm_starts_at(r['ratio_M1'])
    if np.isfinite(s0):
        print(f"     at this 24 um size the warm group starts at shape {s0:.2f} (about {temp_from_shape(s0):.0f} K dust), "
              f"so this star is {(r['shape_I4'] - s0) / r['eshape_I4']:+.1f} sigma from the line")

# ---- compare with version 1 (if excess_shape.py has been run) ----
v1_file = DATA / 'excess_shape.csv'
if v1_file.exists():
    v1 = pd.read_csv(v1_file)
    v1B = v1[v1['shape_group'].notna()]
    v1_warm_g = v1B.groupby('shape_group')['shape_I4'].median().idxmax()
    v1_warm = set(v1B.loc[v1B['shape_group'] == v1_warm_g, 'objid'])
    v2_warm = set(B.loc[B['shape_name'] == 'warm', 'objid'])
    v1_picks = set(v1.loc[v1['picked'] == True, 'objid'])
    v2_picks = set(a.loc[a['picked'], 'objid'])
    nm = lambda ids: ', '.join(sorted(df.set_index('objid').loc[list(ids), 'name'])) or 'none'
    print('\n========== Version 1 vs version 2 ==========')
    print(f'Part A picks (seed 42): v1 {len(v1_picks)}, v2 {len(v2_picks)}')
    print(f'  new in v2: {nm(v2_picks - v1_picks)}')
    print(f'  dropped in v2: {nm(v1_picks - v2_picks)}')
    print(f'Warm group: v1 {len(v1_warm)} stars, v2 {len(v2_warm)} stars, {len(v1_warm & v2_warm)} in both')
    print(f'  moved to warm in v2: {len(v2_warm - v1_warm)}, moved to cool/flat in v2: {len(v1_warm - v2_warm)}')
    moved = B[B['objid'].isin(v1_warm ^ v2_warm)]
    print(f"  stars that moved: median 8 um shape {moved['shape_I4'].median():.2f}, "
          f"median 4.5 um shape {moved['shape_I2'].median():.2f}")

# ---- save ----
keep = ['objid', 'name', 'group', 'Teff', 'comparable'] + \
       [f'{p}_{b}' for b in BANDS for p in ('pred', 'ratio', 'ex', 'exsig')]
out = df[keep].merge(a[['objid', 'weirdness', 'flagged', 'picked']], on='objid', how='left').merge(
    B[['objid', 'shape_I2', 'shape_I3', 'shape_I4', 'eshape_I4', 'warm_8um', 'T_dust', 'T_dust_max',
       'shape_group', 'shape_name']], on='objid', how='left')
out.to_csv(DATA / 'excess_shape_v2.csv', index=False)
print('\nSaved data/excess_shape_v2.csv')

# ======================================================================
# Figure (same layout as v1; colors: warm = orange, cool/flat = blue, like cluster_stability.py)
# ======================================================================
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e4e3df', '#a3a29d'
COLOR = {'warm': '#eb6834', 'cool/flat': '#2a78d6'}
FLAG = '#2a78d6'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# A: size at 8 um vs size at 24 um, comparable stars
norm = a[~a['flagged']]
ax1.scatter(norm['ratio_M1'], norm['ratio_I4'], s=36, color=GRAY, edgecolors='white', lw=1,
            label=f'Normal ({len(norm)})', zorder=2)
fl = a[a['flagged'] & ~a['cut']]
ax1.scatter(fl['ratio_M1'], fl['ratio_I4'], s=56, facecolors='white', edgecolors=FLAG, lw=1.5,
            label=f'Flagged, no 24 um excess ({len(fl)})', zorder=3)
pk = a[a['picked']]
ax1.scatter(pk['ratio_M1'], pk['ratio_I4'], s=56, color=FLAG, edgecolors='white', lw=1,
            label=f'Picked: flagged + real excess ({len(pk)})', zorder=3)
ax1.axvline(EXCESS_RATIO, color=INK2, lw=0.8, ls=':', zorder=1)
ax1.text(EXCESS_RATIO * 1.03, 0.995, f'{EXCESS_RATIO}x', color=INK2, fontsize=8, va='top',
         transform=ax1.get_xaxis_transform())
ax1.set_xscale('log')
ax1.set_xlabel('24 um: measured / predicted star', color=INK)
ax1.set_ylabel('8 um: measured / predicted star', color=INK)
ax1.set_title(f'A. Size of the excess ({len(a)} comparable stars), flag = weirdness > {WEIRD_CUT}',
              loc='left', fontsize=11, color=INK)

# B: shape at 8 um vs size at 24 um, all stars with an excess, plus the KMeans boundary
for nm_g in ['warm', 'cool/flat']:
    s = B[B['shape_name'] == nm_g]
    ax2.scatter(s['ratio_M1'], s['shape_I4'].clip(-1, 3), s=30, color=COLOR[nm_g], edgecolors='white',
                lw=0.8, label=f'{nm_g.capitalize()} group ({len(s)})', zorder=3)
comp = B[B['comparable']]
ax2.scatter(comp['ratio_M1'], comp['shape_I4'].clip(-1, 3), s=110, facecolors='none',
            edgecolors=INK, lw=1, label='Spitzer sees the star alone', zorder=4)
xs = np.logspace(np.log10(B['ratio_M1'].min() * 0.9), np.log10(B['ratio_M1'].max() * 1.1), 200)
bound = [warm_starts_at(x) for x in xs]
ax2.plot(xs, bound, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=2, label='Group boundary')
for T in (200, 250, 300):
    ax2.axhline(bb_shape('I4', T), color=GRID, lw=1, zorder=1)
    ax2.text(1.0, bb_shape('I4', T), f' {T} K dust', color=INK2, fontsize=8, va='center',
             transform=ax2.get_yaxis_transform())
ax2.axhline(0, color=GRAY, lw=0.8, zorder=1)
ax2.set_xscale('log')
ax2.set_xlabel('24 um: measured / predicted star', color=INK)
ax2.set_ylabel('Extra light at 8 um / extra light at 24 um (energy)', color=INK)
ax2.set_title(f'B. Shape of the excess ({len(B)} stars), grouped on 8 um shape + 24 um size',
              loc='left', fontsize=11, color=INK)

for ax, tab in ((ax1, a), (ax2, B)):
    ycol = 'ratio_I4' if ax is ax1 else 'shape_I4'
    for objid, nm_s in NAMED.items():
        r = tab[tab['objid'] == objid]
        if len(r):
            y = r[ycol].iloc[0] if ax is ax1 else min(max(r[ycol].iloc[0], -1), 3)
            ax.annotate(nm_s, (r['ratio_M1'].iloc[0], y), xytext=(8, 6), textcoords='offset points',
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
fig.savefig(FIGS / 'excess_shape_v2.png', dpi=150)
print('Saved figures/excess_shape_v2.png')
