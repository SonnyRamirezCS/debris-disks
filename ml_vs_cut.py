"""
What does the ML add over a simple cut? (version 1, for the Dec 18 "why ML" slide)

Someone will ask: "Why do you need machine learning? Couldn't a simple rule find
these stars?" This script puts the methods side by side on the same stars.

The rules compared:
  Skorpen's rule   24 um excess >= 3 sigma AND at least 10% above the star
                   (the "3 sigma + 10%" rule from her paper, as we understand it,
                   using our star predictions and error bars)
  our cut          >= 3 sigma AND >= 1.2x (what excess_shape.py calls a real excess)
  ML version 1     Isolation Forest flags the weirdest 5% (excess_shape.py)
  ML version 2     Isolation Forest flags weirdness > 0.5 (excess_shape_v2.py)
  (both ML versions also need our cut to count a star as picked)

Part 1  Finding: which of the 59 comparable stars each method picks, with the
        ML as a % of 50 random seeds, plus the image check from vetting/.
        If data/fake_disks.csv is there, also the faintest fake disk each rule
        finds in >= 90% of the normal stars.
Part 2  What else the ML flags: stars it calls unusual that have no 24 um excess.
Part 3  Sorting: does the warm vs cool/flat split line up with the image check?
        Off-center rates (cutoff from vetting/control_cutoff.py, 1 in 10) for
        plain stars and each group, version 1 vs version 2 groups.

Run from the debris-disks folder:  python ml_vs_cut.py
Needs data/kdwarf_sfr.csv, data/excess_shape.csv, data/excess_shape_v2.csv,
  vetting/results/batch_vet.csv
Uses data/fake_disks.csv if it's there (made by fake_disks.py)
Makes data/ml_vs_cut.csv and figures/ml_vs_cut.png
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import IsolationForest

# the same pipeline functions the fake test uses, so the features match excess_shape(_v2).py exactly
from fake_disks import (A_FEAT, CONTAMINATION, EDGES, EXCESS_SIG, F_GRID, GROUPS, MIN_PLAIN, NAMED,
                        T_GRID, TEFF_MAX, TEFF_MIN, WEIRD_CUT, add_features, add_ratios, calibration,
                        faintest, pct_light, plain_stars)

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
DATA, FIGS, VET = HERE / 'data', HERE / 'figures', HERE / 'vetting' / 'results'

SKORPEN_RATIO = 1.1     # "at least 10% above the star"
OUR_RATIO = 1.2
SEEDS = 50
CUT_LEVEL = 0.90        # off-center cutoff: only 1 in 10 plain stars get past it (as in control_cutoff.py)
LEVEL = 90              # fake test: "found" = found in >= 90% of the normal stars

# ---- real stars, same features as excess_shape(_v2).py ----
df = pd.read_csv(DATA / 'kdwarf_sfr.csv').copy()
df = df[df['clean'].astype(str).str.lower().isin(['true', '1', '1.0'])]
df = df[(df['Teff'] > TEFF_MIN) & (df['Teff'] < TEFF_MAX)].copy()
df['group'] = pd.cut(df['Teff'], [TEFF_MIN] + EDGES + [TEFF_MAX], right=False, labels=GROUPS)
df['name'] = df['objid'].map(NAMED).fillna(df['objid'].str.replace('SSTSL2 ', '', regex=False))
df = add_ratios(df)
all_plain = plain_stars(df)
cal_all = calibration(all_plain)
cal = {g: (calibration(all_plain[all_plain['group'] == g]) if (all_plain['group'] == g).sum() >= MIN_PLAIN
           else cal_all) for g in GROUPS}
df = add_features(df, cal, cal_all)
df['skorpen'] = (df['exsig_M1'] >= EXCESS_SIG) & (df['ratio_M1'] >= SKORPEN_RATIO)
df['ours'] = df['cut']

a = df[df['comparable']].dropna(subset=A_FEAT).copy()
X = a[A_FEAT].to_numpy(float)
hit1, hit2, wsum = np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a))
for seed in range(SEEDS):
    m = IsolationForest(contamination=CONTAMINATION, random_state=seed).fit(X)
    w = -m.score_samples(X)
    hit1 += m.predict(X) == -1
    hit2 += w > WEIRD_CUT
    wsum += w
a['ml_v1_pct'] = 100 * hit1 / SEEDS * a['ours']
a['ml_v2_pct'] = 100 * hit2 / SEEDS * a['ours']
a['weirdness'] = wsum / SEEDS
a['flag_v2_pct'] = 100 * hit2 / SEEDS

# image check (vetting/batch_vet.py), cutoff from the control stars
vet = pd.read_csv(VET / 'batch_vet.csv')
ok = vet[(vet['status'] == 'ok') & vet['offset_x_noise'].notna()].copy()
off_cut = float(ok.loc[ok['group_name'] == 'control', 'offset_x_noise'].quantile(CUT_LEVEL))
vet_i = vet.set_index('objid')


def image_word(objid):
    if objid not in vet_i.index:
        return 'not checked'
    r = vet_i.loc[objid]
    if r['status'] != 'ok' or not np.isfinite(r['offset_x_noise']):
        return 'could not run'
    return 'OFF-CENTER' if r['offset_x_noise'] > off_cut else 'centered'


# ======================================================================
# Part 1: finding
# ======================================================================
print(f'Clean K stars: {len(df)}. Pass Skorpen\'s rule: {int(df["skorpen"].sum())}, '
      f'pass our cut: {int(df["ours"].sum())}, '
      f'both: {int((df["skorpen"] & df["ours"]).sum())}')
print(f'\n========== Part 1: finding, the {len(a)} comparable stars ==========')
ex = a[a['skorpen'] | a['ours']].sort_values('ratio_M1', ascending=False)
print(f"{'star':22s} {'24 um':>6s} {'sigma':>6s}  {'Skorpen':>7s} {'ours':>5s} {'ML v1':>6s} {'ML v2':>6s}  "
      f"{'weirdness':>9s}  image check")
for _, r in ex.iterrows():
    print(f"{r['name']:22s} {r['ratio_M1']:5.1f}x {r['exsig_M1']:6.1f}  {'yes' if r['skorpen'] else '-':>7s} "
          f"{'yes' if r['ours'] else '-':>5s} {r['ml_v1_pct']:5.0f}% {r['ml_v2_pct']:5.0f}%  "
          f"{r['weirdness']:9.2f}  {image_word(r['objid'])}")
print(f'(ML columns = % of {SEEDS} seeds that pick the star. Weirdness = average Isolation Forest score; '
      f'version 2 flags above {WEIRD_CUT})')
rank = a.sort_values('weirdness', ascending=False)['name'].tolist()
print(f'Weirdest 3 of all {len(a)} comparable stars: {", ".join(rank[:3])}')

fk_file = DATA / 'fake_disks.csv'
faint_rows = []
if fk_file.exists():
    fk = pd.read_csv(fk_file)
    fk = fk[fk['kind'] == 'grid'].copy()
    fk['skorpen'] = (fk['exsig_M1'] >= EXCESS_SIG) & (fk['ratio_M1'] >= SKORPEN_RATIO)
    rules = {'Skorpen\'s rule': 'skorpen', 'our cut': 'cut', 'ML version 1': 'found_v1', 'ML version 2': 'found_v2'}
    print(f'\nFake disks (data/fake_disks.csv): faintest disk found in >= {LEVEL}% of the normal stars')
    print(f"{'dust T':>7s}  " + ' '.join(f'{k:>15s}' for k in rules))
    for T in T_GRID:
        cells = []
        for k, col in rules.items():
            pct = fk[fk['T_in'] == T].groupby('iF')[col].mean().reindex(range(len(F_GRID))).to_numpy() * 100
            i = faintest(pct, LEVEL)
            cells.append('not at 1%' if i is None else f'{pct_light(F_GRID[i])}')
            faint_rows.append(dict(T=T, rule=k, faintest_pct_of_star_light=np.nan if i is None else 100 * F_GRID[i]))
        print(f'{T:6d} K  ' + ' '.join(f'{c:>15s}' for c in cells))
else:
    print('\n(no data/fake_disks.csv, run fake_disks.py to add the fake disk comparison)')

# ======================================================================
# Part 2: what else the ML flags
# ======================================================================
odd = a[(a['flag_v2_pct'] >= 50) & ~a['ours']].sort_values('weirdness', ascending=False)
print(f'\n========== Part 2: stars the ML calls unusual that have no 24 um excess ({len(odd)}) ==========')
print(f"{'star':22s} {'4.5 um':>7s} {'5.8 um':>7s} {'8 um':>6s} {'24 um':>6s}  (measured / predicted star)")
for _, r in odd.iterrows():
    print(f"{r['name']:22s} {r['ratio_I2']:6.2f}x {r['ratio_I3']:6.2f}x {r['ratio_I4']:5.2f}x {r['ratio_M1']:5.2f}x")
print('If 4.5, 5.8 and 8 um are all high by about the same amount, the 3.6 um flux (which every prediction')
print('is scaled from) is probably low. That is a photometry issue, not dust, and the excess cut drops these.')

# ======================================================================
# Part 3: sorting, do the shape groups line up with the image check?
# ======================================================================
v1 = pd.read_csv(DATA / 'excess_shape.csv')
v1B = v1[v1['shape_group'].notna()]
v1_warm = v1B.groupby('shape_group')['shape_I4'].median().idxmax()
v1_name = dict(zip(v1B['objid'], np.where(v1B['shape_group'] == v1_warm, 'warm', 'cool/flat')))
v2 = pd.read_csv(DATA / 'excess_shape_v2.csv')
v2_name = dict(zip(v2.loc[v2['shape_name'].notna(), 'objid'], v2.loc[v2['shape_name'].notna(), 'shape_name']))

print(f'\n========== Part 3: sorting, off-center rates by group (cutoff {off_cut:.2f}, only 1 in 10 plain stars '
      'get past it) ==========')
rows = []
for ver, names in ((1, v1_name), (2, v2_name)):
    g = np.where(ok['group_name'] == 'control', 'control', ok['objid'].map(names))
    counts = {}
    for grp in ['control', 'warm', 'cool/flat']:
        s = ok.loc[g == grp, 'offset_x_noise']
        k, n = int((s > off_cut).sum()), len(s)
        counts[grp] = (k, n)
        p = k / n
        z = 1.0                                   # 1 sigma Wilson range, same as control_cutoff.py
        mid = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        rows.append(dict(version=ver, group=grp, off_center=k, stars=n, percent=100 * p,
                         low=100 * (mid - half), high=100 * (mid + half)))
    txt = ', '.join(f'{grp} {k}/{n} = {100 * k / n:.0f}%' for grp, (k, n) in counts.items())
    print(f'  version {ver} groups: {txt}')
    for x, y in [('cool/flat', 'control'), ('warm', 'control'), ('cool/flat', 'warm')]:
        (kx, nx), (ky, ny) = counts[x], counts[y]
        pval = fisher_exact([[kx, nx - kx], [ky, ny - ky]])[1]
        print(f'     {x} vs {y}: p = {pval:.3f} -> ' + ('probably real' if pval < 0.05 else 'could be luck'))
R = pd.DataFrame(rows)

print('\nIn short: on these stars the ML does not find anything the excess cut misses (a pick has to pass')
print('the cut anyway). What it adds is a ranking (how unusual each excess is) and the shape groups.')
print('The cool/flat group has more off-center (likely background galaxy) stars than plain stars do;')
print('the warm group is too small to tell apart from either.')

# ---- save ----
keep = ['objid', 'name', 'ratio_M1', 'exsig_M1', 'skorpen', 'ours', 'ml_v1_pct', 'ml_v2_pct', 'weirdness']
out = a[keep].copy()
out['image_check'] = [image_word(o) for o in out['objid']]
out.sort_values('ratio_M1', ascending=False).round(3).to_csv(DATA / 'ml_vs_cut.csv', index=False)
print('\nSaved data/ml_vs_cut.csv')

# ======================================================================
# Figure: left = who picks which star, right = off-center rate by group, v1 vs v2
# ======================================================================
INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE, GRAYC = '#2a78d6', '#eb6834', '#898781'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
fig.patch.set_facecolor(BG)

# left: dot table
cols = [('Skorpen\'s\nrule', 'skorpen'), ('Our\ncut', 'ours'), ('ML\nversion 1', 'ml_v1_pct'),
        ('ML\nversion 2', 'ml_v2_pct')]
stars = ex.reset_index(drop=True)
ny = len(stars)
for j, (lab, col) in enumerate(cols):
    for i, r in stars.iterrows():
        y = ny - 1 - i
        v = 100.0 * r[col] if col in ('skorpen', 'ours') else float(r[col])
        if v > 0:
            a1.scatter(j, y, s=40 + 3.2 * v, color=BLUE, alpha=0.25 + 0.75 * v / 100, edgecolors='white', lw=1,
                       zorder=3)
        else:
            a1.scatter(j, y, s=40, facecolors='none', edgecolors=AXIS, lw=1.2, zorder=3)
        if col.startswith('ml'):
            a1.text(j + 0.22, y, f'{v:.0f}%', va='center', fontsize=8.5, color=INK2)
XI = len(cols) + 0.2                     # the image check column
for i, r in stars.iterrows():
    y = ny - 1 - i
    word = image_word(r['objid'])
    a1.text(XI, y, word, va='center', ha='center', fontsize=8.5,
            color=ORANGE if word == 'OFF-CENTER' else INK2)
a1.set_xticks(list(range(len(cols))) + [XI], [c[0] for c in cols] + ['Image\ncheck'])
a1.xaxis.tick_top()
a1.set_yticks(range(ny), [f"{r['name']}  {r['ratio_M1']:.1f}x" for _, r in stars.iloc[::-1].iterrows()])
a1.set_xlim(-0.5, XI + 0.6)
a1.set_ylim(-0.7, ny - 0.3)
a1.set_title(f'Who picks which star? ({len(a)} comparable stars, the {ny} with a 24 um excess shown)',
             loc='left', fontsize=11.5, color=INK, pad=34)
a1.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, color=BLUE, label='picked (ML: bigger = more seeds)'),
                   Line2D([], [], marker='o', ls='', ms=7, mfc='none', mec=AXIS, label='not picked')],
          frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.45, -0.02), ncol=2)
for side in a1.spines.values():
    side.set_visible(False)
a1.tick_params(length=0)

# right: off-center % by group, version 1 vs version 2 groups
grp_order = ['control', 'warm', 'cool/flat']
gcol = {'control': GRAYC, 'warm': ORANGE, 'cool/flat': BLUE}
xbase = np.arange(len(grp_order))
for k, ver in enumerate((1, 2)):
    sub = R[R['version'] == ver].set_index('group').loc[grp_order]
    xs = xbase + (k - 0.5) * 0.36
    a2.bar(xs, sub['percent'], width=0.32, color=[gcol[g] for g in grp_order],
           alpha=0.45 if ver == 1 else 1.0, edgecolor=BG, lw=2, zorder=2)
    a2.errorbar(xs, sub['percent'], yerr=[sub['percent'] - sub['low'], sub['high'] - sub['percent']],
                fmt='none', ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for x, (g, r) in zip(xs, sub.iterrows()):
        a2.text(x, r['high'] + 1.2, f"{r['percent']:.0f}%", ha='center', va='bottom', fontsize=8.5, color=INK)
        a2.text(x, 0.8, f'v{ver}', ha='center', va='bottom', fontsize=8.5, color=INK)
a2.set_xticks(xbase, ['Plain stars\n(control)', 'Warm\ngroup', 'Cool/flat\ngroup'])
a2.set_ylim(0, 50)
a2.set_ylabel('% off-center (24 um light not on the star)')
a2.set_title('Do the groups line up with the image check?', loc='left', fontsize=11.5, color=INK, pad=34)
a2.text(0, 1.03, f'Pale = version 1 groups, solid = version 2 groups. Error bars: 1 sigma. Cutoff {off_cut:.1f}.',
        transform=a2.transAxes, fontsize=8.5, color=INK2)
a2.spines[['top', 'right']].set_visible(False)
a2.grid(axis='y', color=GRID, lw=0.8, zorder=0)
a2.tick_params(length=0)

for ax in (a1, a2):
    ax.set_facecolor(BG)
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / 'ml_vs_cut.png', dpi=200, facecolor=BG)
print('Saved figures/ml_vs_cut.png')
