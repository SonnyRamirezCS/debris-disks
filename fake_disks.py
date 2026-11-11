"""
Fake disk test (injection and recovery), version 2: tests both versions of the ML pipeline.

The question someone will ask: "How do you know your ML would find a real disk?"
This script answers it. It takes normal stars, adds the light of a fake dust disk,
reruns the pipeline, and counts how often the fake is found. That gives the
faintest disk the ML can find, and checks which shape group the fakes land in.

Two versions of the pipeline are tested on the exact same fakes:
  version 1 = excess_shape.py     Isolation Forest flags the weirdest 5% of stars;
                                  KMeans groups on the 4.5 um shape, 8 um shape, 24 um size
  version 2 = excess_shape_v2.py  Isolation Forest flags weirdness > 0.5;
                                  KMeans groups on the 8 um shape and 24 um size only
The first run of this test (Nov 9) found two problems with version 1. Version 2
is the fix for both, and this run checks whether the fixes work.

1. Hosts = comparable stars (Spitzer sees the star alone at 24 um) that look
   normal: no real 24 um excess and not flagged by version 1. The same 51 hosts
   are used for both versions.
2. Fake disk = a blackbody at dust temperature T with fractional luminosity
   f = L_dust / L_star (the dust gives off f times the star's total light).
   Its light is added to every band the pipeline uses (3.6, 4.5, 5.8, 8, 24 um),
   so hot dust also brightens 3.6 um, the band the star predictions come from,
   the same way a real disk would.
3. One fake at a time replaces its host, then the pipeline runs again:
   Part A  Isolation Forest refit on the comparable stars with the fake in.
           The two versions use the same forest, only the cutoff differs.
           FOUND = flagged AND passes the excess cut (>= 3 sigma and >= 1.2x),
           the same rule the pipeline uses for "picked".
   Part B  If the fake passes the excess cut: which shape group it lands in
           (warm or cool/flat) in each version, and what dust temperature its
           8/24 um shape gives (the same in both versions).
4. Every host gets every fake on a grid of T and f, plus a copy of J2133+83's own
   disk (its T and f as measured by this same pipeline).

Choices to know about:
- Calibration, the KMeans groups and their scaling come from the real data and
  stay fixed. One fake star barely moves them.
- Each fake gets its own random seed, so the result also averages over the
  forest's randomness. Version 1's numbers are the same as the Nov 9 run.
- L_star comes from Gaia G with A_G and BC_G = -0.16, same as j2133_numbers.py.
- The host's measured fluxes already carry real noise. The dust light goes on top.

Run from the debris-disks folder:  python fake_disks.py
Quick test run (5 hosts):          python fake_disks.py --quick
Needs data/kdwarf_sfr.csv (made by sfr_cut.py)
Makes data/fake_disks.csv, data/fake_disks_summary.csv,
  figures/fake_disks.png (version 1), figures/fake_disks_v2.png (version 2)
  and figures/fake_disks_compare.png (the two side by side)
Takes about a minute on 12+ cores. Uses all CPU cores; --jobs 1 runs on one core.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # quiets a harmless KMeans warning on Windows

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
DATA, FIGS = HERE / 'data', HERE / 'figures'

# ---- Pipeline settings (same as excess_shape.py and excess_shape_v2.py, keep them in sync) ----
TEFF_MIN, TEFF_MAX, EDGES = 3930, 5380, [4400, 4700]
GROUPS = ['Late K', 'Mid K', 'Early K']
SEE_MIN = 3          # star-only 24 um flux >= 3x its error = Spitzer can see the star alone
CAL_SN = 5           # calibration stars need the star seen at >= 5x the error
MIN_PLAIN = 5        # a group with fewer plain stars than this uses the all-K calibration
EXCESS_SIG, EXCESS_RATIO = 3, 1.2
CONTAMINATION = 0.05 # version 1: flag the weirdest 5%
WEIRD_CUT = 0.5      # version 2: flag weirdness > 0.5
K_RANGE = range(2, 5)
VERSIONS = {1: {'rule': 'weirdest 5%', 'B_FEAT': ['f_shape_I2', 'f_shape_I4', 'f_size'],
                'saved': 'excess_shape.csv'},
            2: {'rule': f'weirdness > {WEIRD_CUT}', 'B_FEAT': ['f_shape_I4', 'f_size'],
                'saved': 'excess_shape_v2.csv'}}
NAMED = {'SSTSL2 J213307.73+830501.2': 'J2133+83', 'SSTSL2 J084517.41-785937.1': 'J0845-78'}
J2133 = 'SSTSL2 J213307.73+830501.2'

LAM = {'I1': 3.550, 'I2': 4.493, 'I3': 5.731, 'I4': 7.872, 'M1': 23.68}
FLUX = {'I2': ('i2_f_ap2', 'i2_df_ap2'), 'I3': ('i3_f_ap2', 'i3_df_ap2'),
        'I4': ('i4_f_ap2', 'i4_df_ap2'), 'M1': ('m1_f_psf', 'm1_df_psf')}
BANDS = list(FLUX)
RAW = {'I1': 'i1_f_ap2', **{b: f for b, (f, _) in FLUX.items()}}   # every band the dust light goes into
CAL_ERR = {'I2': 0.03, 'I3': 0.03, 'I4': 0.03, 'M1': 0.04}       # absolute calibration
A_FEAT = ['z_I2', 'z_I4', 'z_M1']
h, c, k_B = 6.626e-34, 2.998e8, 1.381e-23
SIGMA_SB = 5.670e-8
BC_G = -0.16

# ---- Fake disk grid ----
T_GRID = [100, 150, 200, 250, 300, 400, 500]      # dust temperature (K)
F_GRID = 10 ** np.arange(-5.0, -1.99, 0.25)       # L_dust / L_star, from 0.001% to 1% of the star's light
LEVEL = 90                                        # "reliably found" = found in >= 90% of the normal stars
MIN_FOR_PCT = 5                                   # need at least this many fakes before quoting a %
BRIGHT_F = 1e-3                                   # "warm group" %s use fakes at f >= 0.1%


# ======================================================================
# The pipeline, as functions (same math as excess_shape.py)
# ======================================================================
def bnu(lam_um, T):
    nu = c / (lam_um * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k_B * T))


def bb_shape(b, T):
    """nu*F_nu at band b over nu*F_nu at 24 um for dust at temperature T."""
    return (bnu(LAM[b], T) / LAM[b]) / (bnu(LAM['M1'], T) / LAM['M1'])


T_FINE = np.arange(60, 1500, 1.0)
S8_FINE = bb_shape('I4', T_FINE)


def temp_from_shape(s):
    return float(np.interp(s, S8_FINE, T_FINE)) if np.isfinite(s) and s > S8_FINE[0] else np.nan


def add_ratios(t):
    for b, (f, _) in FLUX.items():
        t['rj_' + b] = t['i1_f_ap2'] * (LAM['I1'] / LAM[b])**2       # Rayleigh-Jeans from 3.6 um
        t['r_' + b] = t[f] / t['rj_' + b]                           # measured / RJ
    return t


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


def add_features(t, cal, cal_all):
    """Everything excess_shape.py measures for a star, for any table of stars (real or fake)."""
    for b, (f, e) in FLUX.items():
        corr = t['group'].map({g: cal[g][b] for g in GROUPS}).astype(float)
        frac = t['group'].map({g: max(cal[g]['sd_' + b], 0.02) for g in GROUPS}).astype(float)
        t['pred_' + b] = t['rj_' + b] * corr                        # star-only prediction (uJy)
        t['ratio_' + b] = t[f] / t['pred_' + b]
        t['ex_' + b] = t[f] - t['pred_' + b]                       # extra light (uJy)
        t['eex_' + b] = np.sqrt(t[e]**2 + (CAL_ERR[b] * t[f])**2 + (frac * t['pred_' + b])**2)
        t['exsig_' + b] = t['ex_' + b] / t['eex_' + b]
    t['comparable'] = t['pred_M1'] / t['m1_df_psf'] >= SEE_MIN
    t['cut'] = (t['exsig_M1'] >= EXCESS_SIG) & (t['ratio_M1'] >= EXCESS_RATIO)
    with np.errstate(invalid='ignore', divide='ignore'):
        for b in ['I2', 'I4', 'M1']:                                 # Part A: size features
            sd = np.log10(1 + max(cal_all['sd_' + b], 0.02))
            t['z_' + b] = np.log10(t['ratio_' + b]) / sd
        for b in ['I2', 'I3', 'I4']:                                 # Part B: shape features
            t['shape_' + b] = (t['ex_' + b] / LAM[b]) / (t['ex_M1'] / LAM['M1'])
            t['eshape_' + b] = (t['eex_' + b] / LAM[b]) / (t['ex_M1'] / LAM['M1'])
        t['f_shape_I2'] = t['shape_I2'].clip(-1, 3)
        t['f_shape_I4'] = t['shape_I4'].clip(-1, 3)
        t['f_size'] = np.log10(t['ratio_M1'])
        t['warm_8um'] = t['shape_I4'] / t['eshape_I4'] >= 3
    return t


def fit_groups(B, feats):
    """KMeans shape groups on the real stars, the same way excess_shape(_v2).py makes them."""
    med = B[feats].median()
    mad = 1.4826 * (B[feats] - med).abs().median()
    X = ((B[feats] - med) / mad).to_numpy(float)    # plain numbers, so predicting the fakes gives no warning
    scores = {kk: silhouette_score(X, KMeans(n_clusters=kk, n_init=20, random_state=0).fit_predict(X))
              for kk in K_RANGE}
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
    warm_label = int(pd.Series(B['shape_I4'].to_numpy()).groupby(km.labels_).median().idxmax())
    return {'feats': feats, 'med': med, 'mad': mad, 'km': km, 'warm_label': warm_label, 'k': best_k,
            'warm': km.labels_ == warm_label}


def star_light(t):
    """Star's total light reaching us (W/m^2) from Gaia G, A_G and BC_G (same as j2133_numbers.py)."""
    return 2.518e-8 * 10**(-0.4 * (t['Gmag'] - np.nan_to_num(t['AG'], nan=0.0) + BC_G))


def dust_uJy(lam_um, T, f, F_star):
    """Flux (uJy) at lam_um from dust at temperature T that gives off f times the star's light."""
    return f * F_star * np.pi * bnu(lam_um, T) / (SIGMA_SB * T**4) * 1e32


# ======================================================================
# One host star: all its fakes
# ======================================================================
def run_host(h_i, pos, host, F_star, grid, Xa, ctx):
    """Swap the host for each fake in turn, refit the Isolation Forest, record what happens."""
    fk = host.loc[host.index.repeat(len(grid))].reset_index(drop=True)
    fk['kind'] = [g[0] for g in grid]
    fk['iT'] = [g[1] for g in grid]
    fk['iF'] = [g[2] for g in grid]
    fk['T_in'] = [g[3] for g in grid]
    fk['f_in'] = [g[4] for g in grid]
    T_in, f_in = fk['T_in'].to_numpy(float), fk['f_in'].to_numpy(float)
    fk['dust_24'] = dust_uJy(LAM['M1'], T_in, f_in, F_star)
    for b, col in RAW.items():
        fk[col] = fk[col] + dust_uJy(LAM[b], T_in, f_in, F_star)
    fk = add_features(add_ratios(fk), ctx['cal'], ctx['cal_all'])

    # Part A: one Isolation Forest per fake, with the fake in place of its host.
    # Version 1 and 2 use the same forest; only the cutoff differs.
    flag1, flag2 = np.zeros(len(fk), bool), np.zeros(len(fk), bool)
    weird, rank = np.full(len(fk), np.nan), np.full(len(fk), np.nan)
    Z = fk[A_FEAT].to_numpy(float)
    for i in range(len(fk)):
        if not np.isfinite(Z[i]).all():
            continue
        X = Xa.copy()
        X[pos] = Z[i]
        iso = IsolationForest(contamination=CONTAMINATION, random_state=1000 * h_i + i).fit(X)
        w = -iso.score_samples(X)
        flag1[i] = iso.predict(X[pos:pos + 1])[0] == -1     # version 1: weirdest 5%
        flag2[i] = w[pos] > WEIRD_CUT                        # version 2: weirdness > 0.5
        weird[i] = w[pos]
        rank[i] = int((w > w[pos]).sum()) + 1
    fk['weirdness'], fk['rank'] = weird, rank
    fk['flagged_v1'], fk['flagged_v2'] = flag1, flag2
    fk['found_v1'] = fk['flagged_v1'] & fk['cut']
    fk['found_v2'] = fk['flagged_v2'] & fk['cut']

    # Part B: shape group in each version, and dust temperature, for fakes that pass the excess cut
    inB = fk['cut'].to_numpy()
    for v, G in ctx['groups'].items():
        fk[f'group_v{v}'] = ''
        if inB.any():
            Xb = ((fk.loc[inB, G['feats']] - G['med']) / G['mad']).to_numpy(float)
            lab = G['km'].predict(Xb)
            fk.loc[inB, f'group_v{v}'] = np.where(lab == G['warm_label'], 'warm', 'cool/flat')
    fk['T_rec'] = [temp_from_shape(s) if (cut and w8) else np.nan
                   for s, cut, w8 in zip(fk['shape_I4'], fk['cut'], fk['warm_8um'])]
    fk['T_rec_max'] = [temp_from_shape(s + 2 * e) if (cut and not w8) else np.nan
                       for s, e, cut, w8 in zip(fk['shape_I4'], fk['eshape_I4'], fk['cut'], fk['warm_8um'])]
    fk['host'] = fk['name']
    keep = ['kind', 'iT', 'iF', 'T_in', 'f_in', 'host', 'group', 'Teff', 'dust_24', 'ratio_M1', 'exsig_M1',
            'ratio_I4', 'exsig_I4', 'cut', 'weirdness', 'rank', 'flagged_v1', 'found_v1', 'flagged_v2',
            'found_v2', 'shape_I2', 'shape_I4', 'warm_8um', 'group_v1', 'group_v2', 'T_rec', 'T_rec_max']
    return fk[keep]


def faintest(pcts, level):
    """Index of the faintest f where the % reaches `level` and stays there for every brighter f."""
    bad = np.where(~(np.nan_to_num(pcts, nan=-1) >= level))[0]
    if len(bad) == 0:
        return 0
    return bad[-1] + 1 if bad[-1] + 1 < len(pcts) else None


def pct_light(f):
    return f'{100 * f:.2g}%'


def warm_pct_bright(res, v, T):
    """% of fakes in the warm group, for fakes at dust temperature T, f >= 0.1%, that pass the cut."""
    s = res[(res['kind'] == 'grid') & (res['T_in'] == T) & res['cut'] & (res['f_in'] >= BRIGHT_F)]
    return 100 * (s[f'group_v{v}'] == 'warm').mean() if len(s) >= MIN_FOR_PCT else np.nan


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='only 5 host stars, to check it runs')
    ap.add_argument('--jobs', type=int, default=-1, help='CPU cores to use (-1 = all)')
    args = ap.parse_args()
    t0 = time.time()

    # ---- the real pipeline, exactly as excess_shape.py runs it ----
    df = pd.read_csv(DATA / 'kdwarf_sfr.csv').copy()
    df = df[df['clean'].astype(str).str.lower().isin(['true', '1', '1.0'])]
    df = df[(df['Teff'] > TEFF_MIN) & (df['Teff'] < TEFF_MAX)].copy()
    df['group'] = pd.cut(df['Teff'], [TEFF_MIN] + EDGES + [TEFF_MAX], right=False, labels=GROUPS)
    df['name'] = df['objid'].map(NAMED).fillna(df['objid'].str.replace('SSTSL2 ', '', regex=False))
    df = add_ratios(df)
    all_plain = plain_stars(df)
    cal_all = calibration(all_plain)
    cal = {}
    for g in GROUPS:
        p = all_plain[all_plain['group'] == g]
        cal[g] = calibration(p) if len(p) >= MIN_PLAIN else cal_all
    df = add_features(df, cal, cal_all)

    a = df[df['comparable']].dropna(subset=A_FEAT).copy()
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42).fit(a[A_FEAT])
    a['flagged_v1'] = iso.predict(a[A_FEAT]) == -1
    a['flagged_v2'] = -iso.score_samples(a[A_FEAT]) > WEIRD_CUT   # same trees as contamination='auto'
    B = df[df['cut']].copy()
    groups = {v: fit_groups(B, V['B_FEAT']) for v, V in VERSIONS.items()}

    print(f'Clean K stars: {len(df)}, comparable: {len(a)}, with a real 24 um excess: {len(B)}')
    for v, V in VERSIONS.items():
        picks = a.loc[a[f'flagged_v{v}'] & a['cut'], 'name']
        G = groups[v]
        print(f"Version {v} ({V['rule']}): picks {', '.join(picks)}; "
              f"{G['k']} shape groups, warm group = {int(G['warm'].sum())} stars")

    # same answer as excess_shape.py / excess_shape_v2.py? (only if their output files are there)
    for v, V in VERSIONS.items():
        saved = DATA / V['saved']
        if not saved.exists():
            print(f"(no data/{V['saved']} yet, so no check against it)")
            continue
        s = pd.read_csv(saved)
        same_picks = set(s.loc[s['picked'] == True, 'objid']) == set(a.loc[a[f'flagged_v{v}'] & a['cut'], 'objid'])
        sB = s[s['shape_group'].notna()]
        warm_saved = sB.groupby('shape_group')['shape_I4'].median().idxmax()
        same_groups = set(sB.loc[sB['shape_group'] == warm_saved, 'objid']) == set(B.loc[groups[v]['warm'], 'objid'])
        print(f"Check against data/{V['saved']}: " + ('same picks and same warm group, good'
              if same_picks and same_groups else
              f"DIFFERENT, rerun {V['saved'].replace('.csv', '.py')} first or check the settings"))

    # ---- J2133+83's own disk, measured by this pipeline ----
    j = df[df['objid'] == J2133].iloc[0]
    j_T = temp_from_shape(j['shape_I4'])
    j_f = float(j['ex_M1'] * 1e-32 / bnu(LAM['M1'], j_T) * SIGMA_SB * j_T**4 / np.pi / star_light(j))
    print(f"J2133+83: 24 um {j['ratio_M1']:.1f}x the star, dust about {j_T:.0f} K, f = {j_f:.1e} "
          f"({pct_light(j_f)} of the star's light)")

    # ---- hosts and the fake grid ----
    hosts = a[~a['cut'] & ~a['flagged_v1']]
    if args.quick:
        hosts = hosts.iloc[:5]
    grid = [('grid', iT, iF, T, f) for iT, T in enumerate(T_GRID) for iF, f in enumerate(F_GRID)]
    grid.append(('J2133 copy', -1, -1, j_T, j_f))
    Xa = a[A_FEAT].to_numpy(float)
    pos_of = {idx: k for k, idx in enumerate(a.index)}
    ctx = dict(cal=cal, cal_all=cal_all, groups=groups)
    n_fakes = len(hosts) * len(grid)
    print(f'\nFake disks: {len(T_GRID)} temperatures x {len(F_GRID)} brightnesses + a J2133+83 copy, '
          f'in {len(hosts)} normal stars = {n_fakes} fakes, one Isolation Forest each')

    jobs = (delayed(run_host)(k, pos_of[idx], df.loc[[idx]], float(star_light(df.loc[idx])), grid, Xa, ctx)
            for k, idx in enumerate(hosts.index))
    parts = []
    try:
        out = Parallel(n_jobs=args.jobs, return_as='generator')(jobs)
    except TypeError:                       # older joblib: no progress, same result
        out = Parallel(n_jobs=args.jobs)(jobs)
    for k, part in enumerate(out, 1):
        parts.append(part)
        if k % 10 == 0 or k == len(hosts):
            print(f'  {k}/{len(hosts)} stars done ({time.time() - t0:.0f} s)')
    res = pd.concat(parts, ignore_index=True)

    # ---- summary per grid cell ----
    g = res[res['kind'] == 'grid']
    keys = ['iT', 'iF', 'T_in', 'f_in']
    summ = g.groupby(keys).agg(stars=('host', 'size'), pct_cut=('cut', 'mean'),
                               pct_found_v1=('found_v1', 'mean'), pct_found_v2=('found_v2', 'mean'),
                               median_rank=('rank', 'median'), median_ratio_24=('ratio_M1', 'median'),
                               n_cut=('cut', 'sum')).reset_index()
    summ[['pct_cut', 'pct_found_v1', 'pct_found_v2']] *= 100
    for v in VERSIONS:
        warm = (g[g['cut']].assign(w=lambda t: t[f'group_v{v}'] == 'warm')
                .groupby(keys)['w'].mean().mul(100).rename(f'pct_warm_v{v}').reset_index())
        summ = summ.merge(warm, on=keys, how='left')
        summ.loc[summ['n_cut'] < MIN_FOR_PCT, f'pct_warm_v{v}'] = np.nan
    summ['pct_of_star_light'] = 100 * summ['f_in']

    # ---- print the results ----
    print(f'\n========== How faint a disk can we find? ({len(hosts)} normal stars) ==========')
    print(f'Faintest disk found in >= {LEVEL}% of the stars (24 um size = median measured / predicted star)')
    print(f"{'dust T':>7s}   {'ML version 1':20s} {'ML version 2':20s} {'excess cut alone':20s} {'warm group* v1':>14s} {'v2':>5s}")
    for iT, T in enumerate(T_GRID):
        col = summ[summ['iT'] == iT].sort_values('iF')
        cells = []
        for what in ('pct_found_v1', 'pct_found_v2', 'pct_cut'):
            i = faintest(col[what].to_numpy(), LEVEL)
            cells.append('not even at 1%' if i is None else
                         f"f >= {pct_light(F_GRID[i])} ({col['median_ratio_24'].iloc[i]:.1f}x)")
        wtxt = [f'{p:.0f}%' if np.isfinite(p) else '-' for p in (warm_pct_bright(res, v, T) for v in VERSIONS)]
        print(f'{T:6d} K   {cells[0]:20s} {cells[1]:20s} {cells[2]:20s} {wtxt[0]:>14s} {wtxt[1]:>5s}')
    print(f'* % in the warm shape group, for fakes at f >= {pct_light(BRIGHT_F)} that pass the excess cut')

    jc = res[res['kind'] == 'J2133 copy']
    jcut = jc[jc['cut']]
    n = len(jc)
    print(f"\nCopies of J2133+83's disk ({j_T:.0f} K, {pct_light(j_f)} of the star's light) in {n} normal stars:")
    print(f"{'':28s}{'version 1':>11s}{'version 2':>11s}")
    print(f"{'  found by the ML':28s}{int(jc['found_v1'].sum()):>6d}/{n:<4d}{int(jc['found_v2'].sum()):>6d}/{n:<4d}")
    print(f"{'  lands in the warm group':28s}{int((jcut['group_v1'] == 'warm').sum()):>6d}/{len(jcut):<4d}"
          f"{int((jcut['group_v2'] == 'warm').sum()):>6d}/{len(jcut):<4d}")
    print(f"  median weirdness rank {jc['rank'].median():.0f} of {len(a)}, "
          f"median weirdness {jc['weirdness'].median():.2f} (version 2 flags above {WEIRD_CUT})")
    tr = jcut['T_rec'].dropna()
    if len(tr):
        print(f'  dust temperature back from the 8/24 um shape (same in both versions): median {tr.median():.0f} K '
              f'(middle 68%: {tr.quantile(0.16):.0f} to {tr.quantile(0.84):.0f} K), from {len(tr)} with an 8 um signal')
    for v in VERSIONS:
        missed = jc[~jc[f'found_v{v}']]
        if len(missed):
            print(f"  version {v} missed it in: {', '.join(missed['host'])}")

    # ---- false alarms: hosts with no fake disk at all (the faintest grid row is basically nothing) ----
    nothing = g[g['iF'] == 0]
    print(f"\nWith almost no dust (f = {pct_light(F_GRID[0])}), stars found by mistake: "
          f"version 1 {int(nothing['found_v1'].sum())}/{len(nothing)}, "
          f"version 2 {int(nothing['found_v2'].sum())}/{len(nothing)}")

    # ---- save ----
    tag = '_quick' if args.quick else ''
    res.to_csv(DATA / f'fake_disks{tag}.csv', index=False)
    summ.round(3).to_csv(DATA / f'fake_disks_summary{tag}.csv', index=False)
    print(f'\nSaved data/fake_disks{tag}.csv and data/fake_disks_summary{tag}.csv')
    for v in VERSIONS:
        name = f"fake_disks{'' if v == 1 else '_v2'}{tag}.png"
        make_figure(summ, res, j_T, j_f, FIGS / name, len(hosts), v)
        print(f'Saved figures/{name}')
    make_compare_figure(summ, res, j_T, j_f, FIGS / f'fake_disks_compare{tag}.png', len(hosts))
    print(f'Saved figures/fake_disks_compare{tag}.png ({time.time() - t0:.0f} s total)')


# ======================================================================
# Figures
# ======================================================================
INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(length=0)


def make_figure(summ, res, j_T, j_f, path, n_hosts, v):
    """Three panels for one version: found %, warm group %, dust temperature."""
    NONE = '#f0efec'
    BLUES = LinearSegmentedColormap.from_list(
        'found', ['#f3f7fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    ORANGES = LinearSegmentedColormap.from_list(
        'warm', ['#fdf3ee', '#f9d5c4', '#f2a27f', '#eb6834', '#b9471b', '#7c2a0b'])
    ORANGES.set_bad(NONE)
    nT, nF = len(T_GRID), len(F_GRID)

    def grid_of(col):
        m = np.full((nF, nT), np.nan)
        for _, r in summ.iterrows():
            m[int(r.iF), int(r.iT)] = r[col]
        return m

    def step_line(m):
        """x, y of the line above which the % stays >= LEVEL, one step per temperature column."""
        xs, ys = [], []
        for jT in range(nT):
            i = faintest(m[:, jT], LEVEL)
            y = nF - 0.5 if i is None else i - 0.5
            xs += [jT - 0.5, jT + 0.5]
            ys += [y, y]
        return xs, ys

    jx = np.interp(j_T, T_GRID, np.arange(nT))
    jy = np.interp(np.log10(j_f), np.log10(F_GRID), np.arange(nF))
    yt = [i for i, f in enumerate(F_GRID) if abs(np.log10(f) - round(np.log10(f))) < 1e-6]

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={'width_ratios': [1, 1, 0.95]})
    fig.patch.set_facecolor(BG)

    # A: found by the ML
    im1 = a1.imshow(grid_of(f'pct_found_v{v}'), origin='lower', aspect='auto', cmap=BLUES, vmin=0, vmax=100)
    a1.plot(*step_line(grid_of(f'pct_found_v{v}')), color=INK, lw=2, solid_capstyle='butt')
    a1.plot(*step_line(grid_of('pct_cut')), color=INK2, lw=1.5, ls=(0, (4, 2)))
    cb1 = fig.colorbar(im1, ax=a1, fraction=0.05, pad=0.02)
    cb1.set_label('% of normal stars where the fake is found', color=INK2)
    a1.set_title('A. Does the ML find the fake disk?', loc='left', fontsize=11.5, color=INK)
    leg1 = [Line2D([], [], color=INK, lw=2, label=f'Found by the ML in >= {LEVEL}% (above the line)'),
            Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2)), label=f'Passes the excess cut alone in >= {LEVEL}%'),
            Line2D([], [], marker='*', ls='', ms=13, mfc='white', mec=INK, label="J2133+83's disk")]
    a1.legend(handles=leg1, fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

    # B: lands in the warm group
    im2 = a2.imshow(np.ma.masked_invalid(grid_of(f'pct_warm_v{v}')), origin='lower', aspect='auto',
                    cmap=ORANGES, vmin=0, vmax=100)
    cb2 = fig.colorbar(im2, ax=a2, fraction=0.05, pad=0.02)
    cb2.set_label('% landing in the warm group (of fakes that pass the cut)', color=INK2)
    a2.set_title('B. Does it land in the warm group?', loc='left', fontsize=11.5, color=INK)
    leg2 = [Patch(facecolor=NONE, edgecolor=AXIS, label=f'Too faint: fewer than {MIN_FOR_PCT} pass the excess cut'),
            Line2D([], [], marker='*', ls='', ms=13, mfc='white', mec=INK, label="J2133+83's disk")]
    a2.legend(handles=leg2, fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

    for ax in (a1, a2):
        ax.plot(jx, jy, marker='*', ms=17, mfc='white', mec=INK, mew=1.2, zorder=5)
        ax.annotate('J2133+83', (jx, jy), xytext=(-10, 0), textcoords='offset points', ha='right', va='center',
                    fontsize=9, fontweight='bold', color=INK,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9))
        ax.set_xticks(range(nT), [f'{T}' for T in T_GRID])
        ax.set_yticks(yt, [pct_light(F_GRID[i]) for i in yt])
        ax.set_xlabel('Fake dust temperature (K)')
        ax.set_ylabel("Fake dust brightness (% of the star's light)")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)

    # C: dust temperature measured from the 8/24 um shape (same for both versions)
    g = res[(res['kind'] == 'grid') & res['cut'] & res['warm_8um']]
    pts = []
    for T in T_GRID:
        vals = g.loc[g['T_in'] == T, 'T_rec'].dropna()
        if len(vals) >= MIN_FOR_PCT:
            pts.append((T, vals.median(), vals.quantile(0.16), vals.quantile(0.84), len(vals)))
    lim = (0, max(T_GRID) * 1.15)
    a3.plot(lim, lim, color=AXIS, lw=1, ls=(0, (4, 2)), zorder=1)
    a3.text(lim[1] * 0.97, lim[1] * 0.93, 'measured = fake', color=MUTED, fontsize=8.5, ha='right', rotation=0)
    if pts:
        P = np.array(pts)
        a3.vlines(P[:, 0], P[:, 2], P[:, 3], color='#86b6ef', lw=3, zorder=2)
        a3.plot(P[:, 0], P[:, 1], 'o', ms=8, color='#2a78d6', mec='white', mew=1.5, zorder=3,
                label='Grid fakes with an 8 um signal (median, middle 68%)')
    jc = res[(res['kind'] == 'J2133 copy') & res['cut']]['T_rec'].dropna()
    if len(jc) >= MIN_FOR_PCT:
        a3.vlines(j_T, jc.quantile(0.16), jc.quantile(0.84), color='#f2a27f', lw=3, zorder=2)
        a3.plot(j_T, jc.median(), '*', ms=15, color='#eb6834', mec='white', mew=1.2, zorder=4,
                label="Copies of J2133+83's disk")
    no8 = [T for T in T_GRID if T not in [p[0] for p in pts]]
    if no8:
        a3.text(0.03, 0.04, 'No 8 um signal (cold dust): ' + ', '.join(f'{T} K' for T in no8),
                transform=a3.transAxes, fontsize=8.5, color=MUTED)
    a3.set_xlim(lim)
    a3.set_ylim(lim)
    a3.set_xlabel('Fake dust temperature (K)')
    a3.set_ylabel('Temperature the pipeline measures (K)')
    a3.set_title('C. Is the dust temperature right?', loc='left', fontsize=11.5, color=INK)
    a3.grid(color=GRID, lw=0.8, zorder=0)
    a3.spines[['top', 'right']].set_visible(False)
    a3.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1)
    a3.set_facecolor(BG)

    fig.suptitle(f"Fake disk test, version {v} ({VERSIONS[v]['rule']}): fake dust added to {n_hosts} normal "
                 'stars, one at a time', x=0.01, ha='left', fontsize=12.5, color=INK)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)


def make_compare_figure(summ, res, j_T, j_f, path, n_hosts):
    """Version 1 vs version 2 on the same fakes: how faint, and which group."""
    V1, V2, CUT = '#a3a29d', '#2a78d6', INK          # old = gray, new = blue, the simple cut = ink dashed
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.patch.set_facecolor(BG)
    T = np.array(T_GRID, float)

    # A: faintest disk found in >= 90% of stars, per temperature (lower = finds fainter disks)
    def faint_curve(col):
        ys = []
        for iT in range(len(T_GRID)):
            c_ = summ[summ['iT'] == iT].sort_values('iF')
            i = faintest(c_[col].to_numpy(), LEVEL)
            ys.append(np.nan if i is None else 100 * F_GRID[i])
        return np.array(ys)

    lines = [('pct_found_v1', 'ML version 1 (weirdest 5%)', V1, '-', 'o', 2.5),
             ('pct_found_v2', f'ML version 2 (weirdness > {WEIRD_CUT})', V2, '-', 'o', 2.5),
             ('pct_cut', 'Excess cut alone', CUT, (0, (4, 2)), None, 1.4)]
    for col, lab, colr, ls, mk, lw in lines:
        y = faint_curve(col)
        a1.plot(T, y, color=colr, ls=ls, lw=lw, marker=mk, ms=8, mec='white', mew=1.5, label=lab,
                zorder=3 if col != 'pct_cut' else 4)
    a1.plot(j_T, 100 * j_f, marker='*', ms=17, mfc='white', mec=INK, mew=1.2, ls='', zorder=5,
            label="J2133+83's disk")
    a1.annotate('J2133+83', (j_T, 100 * j_f), xytext=(10, 4), textcoords='offset points', fontsize=9,
                fontweight='bold', color=INK)
    a1.set_yscale('log')
    a1.set_ylim(100 * F_GRID[6] * 0.8, 100 * F_GRID[-1] * 1.6)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}%'))
    a1.set_ylabel("Faintest disk found in >= 90% of stars\n(% of the star's light, lower = better)")
    a1.set_title('A. How faint a disk does each version find?', loc='left', fontsize=11.5, color=INK)

    # B: % of fakes landing in the warm group, per temperature (f >= 0.1%, passing the cut)
    jc = res[(res['kind'] == 'J2133 copy') & res['cut']]
    for v, colr, lab in ((1, V1, 'Version 1 (4.5 + 8 um shape + size)'), (2, V2, 'Version 2 (8 um shape + size)')):
        y = [warm_pct_bright(res, v, t) for t in T_GRID]
        a2.plot(T, y, color=colr, lw=2.5, marker='o', ms=8, mec='white', mew=1.5, label=lab, zorder=3)
        jw = 100 * (jc[f'group_v{v}'] == 'warm').mean()
        a2.plot(j_T, jw, marker='*', ms=17, color=colr, mec=INK, mew=1, ls='', zorder=5)
        a2.annotate(f'{jw:.0f}%', (j_T, jw), xytext=(10, -3), textcoords='offset points', fontsize=9,
                    color=INK, fontweight='bold')
    a2.plot([], [], marker='*', ms=13, mfc='white', mec=INK, ls='', label="Copies of J2133+83's disk")
    a2.set_ylim(-5, 105)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}%'))
    a2.set_ylabel(f'Fakes landing in the warm group\n(f >= {pct_light(BRIGHT_F)}, passing the cut)')
    a2.set_title('B. Does the warm group mean warm dust?', loc='left', fontsize=11.5, color=INK)

    for ax in (a1, a2):
        style(ax)
        ax.set_xticks(T_GRID)
        ax.set_xlabel('Fake dust temperature (K)')
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

    fig.suptitle(f'Fake disk test, version 1 vs version 2: the same fakes in {n_hosts} normal stars',
                 x=0.01, ha='left', fontsize=12.5, color=INK)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    main()
