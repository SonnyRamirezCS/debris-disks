"""
Fake disk test (injection and recovery), version 1, for the Dec 18 ML talk.

The question someone will ask: "How do you know your ML would find a real disk?"
This script answers it. It takes normal stars, adds the light of a fake dust disk,
reruns the same pipeline as excess_shape.py, and counts how often the fake is found.
That gives the faintest disk the ML can find, and checks that warm fakes land in
the warm shape group.

1. Hosts = comparable stars (Spitzer sees the star alone at 24 um) that look
   normal: no real 24 um excess and not flagged by the Isolation Forest.
2. Fake disk = a blackbody at dust temperature T with fractional luminosity
   f = L_dust / L_star (the dust gives off f times the star's total light).
   Its light is added to every band the pipeline uses (3.6, 4.5, 5.8, 8, 24 um),
   so hot dust also brightens 3.6 um, the band the star predictions come from,
   the same way a real disk would.
3. One fake at a time replaces its host, then the pipeline runs again:
   Part A  Isolation Forest refit on the comparable stars with the fake in.
           FOUND = flagged AND passes the excess cut (>= 3 sigma and >= 1.2x),
           the same rule excess_shape.py uses for "picked".
   Part B  If the fake passes the excess cut: which shape group it lands in
           (warm or cool/flat), and what dust temperature its 8/24 um shape gives.
4. Every host gets every fake on a grid of T and f, plus a copy of J2133+83's own
   disk (its T and f as measured by this same pipeline).

Choices to know about:
- Calibration, the KMeans groups and their scaling come from the real data and
  stay fixed. One fake star barely moves them.
- The Isolation Forest flags the weirdest 5%, about 3 of 59 stars, and J2133+83
  and J0845-78 already hold 2 of those spots. So a fake has to beat the 3rd
  weirdest real star. That's how the real pipeline behaves, so the test keeps it.
- Each fake gets its own random seed, so the result also averages over the
  forest's randomness.
- L_star comes from Gaia G with A_G and BC_G = -0.16, same as j2133_numbers.py.
- The host's measured fluxes already carry real noise. The dust light goes on top.

Run from the debris-disks folder:  python fake_disks.py
Quick test run (5 hosts):          python fake_disks.py --quick
Needs data/kdwarf_sfr.csv (made by sfr_cut.py)
Makes data/fake_disks.csv, data/fake_disks_summary.csv and figures/fake_disks.png
Takes a few minutes. Uses all CPU cores; --jobs 1 runs on one core.
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

# ---- Pipeline settings (same as excess_shape.py, keep them in sync) ----
TEFF_MIN, TEFF_MAX, EDGES = 3930, 5380, [4400, 4700]
GROUPS = ['Late K', 'Mid K', 'Early K']
SEE_MIN = 3          # star-only 24 um flux >= 3x its error = Spitzer can see the star alone
CAL_SN = 5           # calibration stars need the star seen at >= 5x the error
MIN_PLAIN = 5        # a group with fewer plain stars than this uses the all-K calibration
EXCESS_SIG, EXCESS_RATIO = 3, 1.2
CONTAMINATION = 0.05
K_RANGE = range(2, 5)
NAMED = {'SSTSL2 J213307.73+830501.2': 'J2133+83', 'SSTSL2 J084517.41-785937.1': 'J0845-78'}
J2133 = 'SSTSL2 J213307.73+830501.2'

LAM = {'I1': 3.550, 'I2': 4.493, 'I3': 5.731, 'I4': 7.872, 'M1': 23.68}
FLUX = {'I2': ('i2_f_ap2', 'i2_df_ap2'), 'I3': ('i3_f_ap2', 'i3_df_ap2'),
        'I4': ('i4_f_ap2', 'i4_df_ap2'), 'M1': ('m1_f_psf', 'm1_df_psf')}
BANDS = list(FLUX)
RAW = {'I1': 'i1_f_ap2', **{b: f for b, (f, _) in FLUX.items()}}   # every band the dust light goes into
CAL_ERR = {'I2': 0.03, 'I3': 0.03, 'I4': 0.03, 'M1': 0.04}       # absolute calibration
A_FEAT = ['z_I2', 'z_I4', 'z_M1']
B_FEAT = ['f_shape_I2', 'f_shape_I4', 'f_size']
h, c, k_B = 6.626e-34, 2.998e8, 1.381e-23
SIGMA_SB = 5.670e-8
BC_G = -0.16

# ---- Fake disk grid ----
T_GRID = [100, 150, 200, 250, 300, 400, 500]      # dust temperature (K)
F_GRID = 10 ** np.arange(-5.0, -1.99, 0.25)       # L_dust / L_star, from 0.001% to 1% of the star's light
LEVEL = 90                                        # "reliably found" = found in >= 90% of the normal stars
MIN_FOR_PCT = 5                                   # need at least this many fakes before quoting a %


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

    # Part A: one Isolation Forest per fake, with the fake in place of its host
    flagged, rank = np.zeros(len(fk), bool), np.full(len(fk), np.nan)
    Z = fk[A_FEAT].to_numpy(float)
    for i in range(len(fk)):
        if not np.isfinite(Z[i]).all():
            continue
        X = Xa.copy()
        X[pos] = Z[i]
        iso = IsolationForest(contamination=CONTAMINATION, random_state=1000 * h_i + i).fit(X)
        w = -iso.score_samples(X)
        flagged[i] = iso.predict(X[pos:pos + 1])[0] == -1
        rank[i] = int((w > w[pos]).sum()) + 1
    fk['flagged'], fk['rank'] = flagged, rank
    fk['found'] = fk['flagged'] & fk['cut']

    # Part B: shape group and dust temperature, for fakes that pass the excess cut
    fk['shape_group'] = ''
    inB = fk['cut'].to_numpy()
    if inB.any():
        Xb = ((fk.loc[inB, B_FEAT] - ctx['medB']) / ctx['madB']).to_numpy(float)
        lab = ctx['km'].predict(Xb)
        fk.loc[inB, 'shape_group'] = np.where(lab == ctx['warm_label'], 'warm', 'cool/flat')
    fk['T_rec'] = [temp_from_shape(s) if (cut and w8) else np.nan
                   for s, cut, w8 in zip(fk['shape_I4'], fk['cut'], fk['warm_8um'])]
    fk['T_rec_max'] = [temp_from_shape(s + 2 * e) if (cut and not w8) else np.nan
                       for s, e, cut, w8 in zip(fk['shape_I4'], fk['eshape_I4'], fk['cut'], fk['warm_8um'])]
    fk['host'] = fk['name']
    keep = ['kind', 'iT', 'iF', 'T_in', 'f_in', 'host', 'group', 'Teff', 'dust_24', 'ratio_M1', 'exsig_M1',
            'ratio_I4', 'exsig_I4', 'cut', 'flagged', 'rank', 'found', 'shape_I4', 'warm_8um', 'shape_group',
            'T_rec', 'T_rec_max']
    return fk[keep]


def faintest(pcts, level):
    """Index of the faintest f where the % reaches `level` and stays there for every brighter f."""
    bad = np.where(~(np.nan_to_num(pcts, nan=-1) >= level))[0]
    if len(bad) == 0:
        return 0
    return bad[-1] + 1 if bad[-1] + 1 < len(pcts) else None


def pct_light(f):
    return f'{100 * f:.2g}%'


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
    a['flagged'] = iso.predict(a[A_FEAT]) == -1
    a['picked'] = a['flagged'] & a['cut']

    B = df[df['cut']].copy()
    medB = B[B_FEAT].median()
    madB = 1.4826 * (B[B_FEAT] - medB).abs().median()
    XB = (B[B_FEAT] - medB) / madB
    XB = XB.to_numpy(float)       # plain numbers, so predicting the fakes later gives no warning
    scores = {kk: silhouette_score(XB, KMeans(n_clusters=kk, n_init=20, random_state=0).fit_predict(XB))
              for kk in K_RANGE}
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(XB)
    B['label'] = km.labels_
    warm_label = int(B.groupby('label')['shape_I4'].median().idxmax())   # the group with the most 8 um light
    B['shape_group'] = np.where(B['label'] == warm_label, 'warm', 'cool/flat')

    print(f'Clean K stars: {len(df)}, comparable: {len(a)}, with a real 24 um excess: {len(B)} '
          f'({best_k} shape groups, warm group = {int((B["shape_group"] == "warm").sum())} stars)')
    print(f"Real picks (Isolation Forest seed 42 + excess cut): {', '.join(a.loc[a['picked'], 'name'])}")

    # same answer as excess_shape.py? (only if its output file is there)
    saved = DATA / 'excess_shape.csv'
    if saved.exists():
        s = pd.read_csv(saved)
        same_picks = set(s.loc[s['picked'] == True, 'objid']) == set(a.loc[a['picked'], 'objid'])
        sB = s[s['shape_group'].notna()]
        warm_saved = sB.groupby('shape_group')['shape_I4'].median().idxmax()
        same_groups = set(sB.loc[sB['shape_group'] == warm_saved, 'objid']) == \
            set(B.loc[B['shape_group'] == 'warm', 'objid'])
        ok = same_picks and same_groups
        print('Check against data/excess_shape.csv: ' + ('same picks and same warm group, good' if ok else
              'DIFFERENT from excess_shape.py, rerun excess_shape.py first or check the settings'))

    # ---- J2133+83's own disk, measured by this pipeline ----
    j = df[df['objid'] == J2133].iloc[0]
    jB = B[B['objid'] == J2133].iloc[0]
    j_T = temp_from_shape(jB['shape_I4'])
    j_f = float(j['ex_M1'] * 1e-32 / bnu(LAM['M1'], j_T) * SIGMA_SB * j_T**4 / np.pi / star_light(j))
    print(f"J2133+83: 24 um {j['ratio_M1']:.1f}x the star, dust about {j_T:.0f} K, f = {j_f:.1e} "
          f"({pct_light(j_f)} of the star's light)")

    # ---- hosts and the fake grid ----
    hosts = a[~a['cut'] & ~a['flagged']]
    if args.quick:
        hosts = hosts.iloc[:5]
    grid = [('grid', iT, iF, T, f) for iT, T in enumerate(T_GRID) for iF, f in enumerate(F_GRID)]
    grid.append(('J2133 copy', -1, -1, j_T, j_f))
    Xa = a[A_FEAT].to_numpy(float)
    pos_of = {idx: k for k, idx in enumerate(a.index)}
    ctx = dict(cal=cal, cal_all=cal_all, km=km, medB=medB, madB=madB, warm_label=warm_label)
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
    res['warm'] = res['shape_group'] == 'warm'

    # ---- summary per grid cell ----
    g = res[res['kind'] == 'grid']
    keys = ['iT', 'iF', 'T_in', 'f_in']
    summ = g.groupby(keys).agg(stars=('host', 'size'), pct_cut=('cut', 'mean'), pct_found=('found', 'mean'),
                               median_rank=('rank', 'median'), median_ratio_24=('ratio_M1', 'median'),
                               n_cut=('cut', 'sum')).reset_index()
    summ[['pct_cut', 'pct_found']] *= 100
    warm = g[g['cut']].groupby(keys)['warm'].mean().mul(100).rename('pct_warm').reset_index()
    summ = summ.merge(warm, on=keys, how='left')
    summ.loc[summ['n_cut'] < MIN_FOR_PCT, 'pct_warm'] = np.nan
    summ['pct_of_star_light'] = 100 * summ['f_in']

    # ---- print the results ----
    print(f'\n========== How faint a disk can we find? ({len(hosts)} normal stars) ==========')
    print(f'Faintest disk found in >= {LEVEL}% of the stars (24 um size = median measured / predicted star)')
    print(f"{'dust T':>7s}   {'found by the ML':28s} {'passes the excess cut alone':28s} {'warm group*':>11s}")
    for iT, T in enumerate(T_GRID):
        col = summ[summ['iT'] == iT].sort_values('iF')
        cells = []
        for what in ('pct_found', 'pct_cut'):
            i = faintest(col[what].to_numpy(), LEVEL)
            cells.append('not even at 1%' if i is None else
                         f"f >= {pct_light(F_GRID[i])} (24 um {col['median_ratio_24'].iloc[i]:.1f}x)")
        bright = g[(g['iT'] == iT) & g['cut'] & (g['f_in'] >= 1e-3)]
        wtxt = f"{100 * bright['warm'].mean():.0f}%" if len(bright) >= MIN_FOR_PCT else '-'
        print(f'{T:6d} K   {cells[0]:28s} {cells[1]:28s} {wtxt:>11s}')
    print('* % in the warm shape group, for fakes at f >= 0.1% that pass the excess cut')

    jc = res[res['kind'] == 'J2133 copy']
    jcut = jc[jc['cut']]
    print(f'\nCopies of J2133+83\'s disk ({j_T:.0f} K, {pct_light(j_f)} of the star\'s light) in '
          f'{len(jc)} normal stars:')
    print(f"  found by the ML: {int(jc['found'].sum())}/{len(jc)}, "
          f"median weirdness rank {jc['rank'].median():.0f} of {len(a)}")
    print(f"  passes the excess cut: {len(jcut)}/{len(jc)}, lands in the warm group: "
          f"{int(jcut['warm'].sum())}/{len(jcut)}")
    tr = jcut['T_rec'].dropna()
    if len(tr):
        print(f'  dust temperature back from the 8/24 um shape: median {tr.median():.0f} K '
              f'(middle 68%: {tr.quantile(0.16):.0f} to {tr.quantile(0.84):.0f} K), from {len(tr)} with an 8 um signal')

    missed = jc[~jc['found']]
    if len(missed):
        print(f"  missed in: {', '.join(missed['host'])} (weirdness rank "
              f"{', '.join(str(int(r)) for r in missed['rank'])})")

    # ---- save ----
    tag = '_quick' if args.quick else ''
    res.to_csv(DATA / f'fake_disks{tag}.csv', index=False)
    summ.round(3).to_csv(DATA / f'fake_disks_summary{tag}.csv', index=False)
    print(f'\nSaved data/fake_disks{tag}.csv and data/fake_disks_summary{tag}.csv')
    make_figure(summ, res, j_T, j_f, FIGS / f'fake_disks{tag}.png', len(hosts))
    print(f'Saved figures/fake_disks{tag}.png ({time.time() - t0:.0f} s total)')


# ======================================================================
# Figure
# ======================================================================
def make_figure(summ, res, j_T, j_f, path, n_hosts):
    INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
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
    im1 = a1.imshow(grid_of('pct_found'), origin='lower', aspect='auto', cmap=BLUES, vmin=0, vmax=100)
    a1.plot(*step_line(grid_of('pct_found')), color=INK, lw=2, solid_capstyle='butt')
    a1.plot(*step_line(grid_of('pct_cut')), color=INK2, lw=1.5, ls=(0, (4, 2)))
    cb1 = fig.colorbar(im1, ax=a1, fraction=0.05, pad=0.02)
    cb1.set_label('% of normal stars where the fake is found', color=INK2)
    a1.set_title('A. Does the ML find the fake disk?', loc='left', fontsize=11.5, color=INK)
    leg1 = [Line2D([], [], color=INK, lw=2, label=f'Found by the ML in >= {LEVEL}% (above the line)'),
            Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2)), label=f'Passes the excess cut alone in >= {LEVEL}%'),
            Line2D([], [], marker='*', ls='', ms=13, mfc='white', mec=INK, label="J2133+83's disk")]
    a1.legend(handles=leg1, fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

    # B: lands in the warm group
    im2 = a2.imshow(np.ma.masked_invalid(grid_of('pct_warm')), origin='lower', aspect='auto', cmap=ORANGES,
                    vmin=0, vmax=100)
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

    # C: dust temperature measured from the 8/24 um shape
    g = res[(res['kind'] == 'grid') & res['cut'] & res['warm_8um']]
    pts = []
    for T in T_GRID:
        v = g.loc[g['T_in'] == T, 'T_rec'].dropna()
        if len(v) >= MIN_FOR_PCT:
            pts.append((T, v.median(), v.quantile(0.16), v.quantile(0.84), len(v)))
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

    fig.suptitle(f'Fake disk test: fake dust added to {n_hosts} normal stars, one at a time, '
                 'then the same pipeline as excess_shape.py', x=0.01, ha='left', fontsize=12.5, color=INK)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)


if __name__ == '__main__':
    main()
