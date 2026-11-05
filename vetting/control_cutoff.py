"""
Set the off-center cutoff from the control stars (version 1).

Why: batch_vet.py calls a star "off-center" when its 24 um light is at least
2.5x the field scatter and 1" away from the star. But about half of the plain
control stars get flagged that way too, even though their 24 um light has to
come from the star itself. So that cutoff is too loose.

Fix: let the control stars set the cutoff. A plain star should only pass it
1 time in 10 (or 1 time in 20 for the strict version). Then count how often
each shape group from excess_shape.py goes past it.

Offset used: offset_x_noise from batch_vet.py = how far the 24 um center is
from the star, divided by how much it should wobble (the field scatter plus
the star's own centering noise). Only stars whose pointing check worked
(status "ok") are used.

Fisher's exact test gives p = how often luck alone would make two groups look
at least this different. Small p (under 0.05) = probably a real difference.

Run from the debris-disks folder:  python vetting/control_cutoff.py
Needs vetting/results/batch_vet.csv (made by vetting/batch_vet.py)
Makes vetting/results/control_cutoff.csv and figures/control_cutoff.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
FIGS = HERE.parent / 'figures'

# ---- Settings ----
METRIC = 'offset_x_noise'
LEVELS = {'1 in 10': 0.90, '1 in 20': 0.95}   # how often a plain star is allowed past the cutoff
MAIN = '1 in 10'                                # the cutoff shown in the figure
GROUPS = ['control', 'warm', 'cool/flat']
NAMED = ['J2133+83', 'J0845-78']

# plot colors: plain stars neutral gray, warm group orange, cool/flat group blue
# (cluster_stability.py uses the same orange and blue)
COLOR = {'control': '#898781', 'warm': '#eb6834', 'cool/flat': '#2a78d6'}
LABEL = {'control': 'Plain stars (control)', 'warm': 'Warm group', 'cool/flat': 'Cool/flat group'}
INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def wilson(k, n, z=1.0):
    """Range for a fraction k out of n. z=1 gives a 1 sigma (68%) error bar."""
    p = k / n
    mid = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return mid - half, mid + half


# ---- load ----
df = pd.read_csv(RESULTS / 'batch_vet.csv')
ok = df[(df['status'] == 'ok') & df[METRIC].notna()].copy()
print('Stars with a working pointing check: ' +
      ', '.join(f"{LABEL[g]} {(ok['group_name'] == g).sum()}" for g in GROUPS))
ctrl = ok.loc[ok['group_name'] == 'control', METRIC]

# ---- count off-center stars at each cutoff ----
rows, cuts = [], {}
for level, q in LEVELS.items():
    cut = float(ctrl.quantile(q))
    cuts[level] = cut
    print(f'\n========== Cutoff where only {level} plain stars pass: {METRIC} > {cut:.2f} ==========')
    counts = {}
    for g in GROUPS:
        s = ok.loc[ok['group_name'] == g, METRIC]
        k, n = int((s > cut).sum()), len(s)
        lo, hi = wilson(k, n)
        counts[g] = (k, n)
        rows.append(dict(cutoff=level, cutoff_value=round(cut, 3), group=g, off_center=k, stars=n,
                         percent=round(100 * k / n, 1),
                         low_1sigma=round(100 * lo, 1), high_1sigma=round(100 * hi, 1)))
        print(f'  {LABEL[g]:22s} {k:3d} of {n:3d} off-center = {100 * k / n:3.0f}%'
              f'   (1 sigma range {100 * lo:.0f} to {100 * hi:.0f}%)')

    print('  Fisher exact test:')
    for a, b in [('cool/flat', 'control'), ('warm', 'control'), ('cool/flat', 'warm')]:
        (ka, na), (kb, nb) = counts[a], counts[b]
        p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        verdict = 'probably a real difference' if p < 0.05 else 'could just be luck'
        print(f'    {a} vs {b}: p = {p:.3f} -> {verdict}')

    for nm in NAMED:
        v = ok.loc[ok['name'] == nm, METRIC].iloc[0]
        print(f'  {nm}: {v:.2f} -> {"OFF-CENTER" if v > cut else "centered"}')

out = pd.DataFrame(rows)
out.to_csv(RESULTS / 'control_cutoff.csv', index=False)
print(f'\nSaved {(RESULTS / "control_cutoff.csv").relative_to(HERE.parent)}')

# ---- figure ----
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
fig.patch.set_facecolor(BG)
cut = cuts[MAIN]

# left: what fraction of each group is at least this far off-center
for g in GROUPS:
    v = np.sort(ok.loc[ok['group_name'] == g, METRIC].to_numpy())
    frac = 1 - np.arange(len(v)) / len(v)          # fraction of stars at or past each value
    a1.step(v, 100 * frac, where='post', color=COLOR[g], lw=2, label=f'{LABEL[g]} ({len(v)})')
a1.axvline(cut, color=INK2, lw=1, ls=(0, (4, 3)))
a1.text(cut * 1.06, 97, f'cutoff = {cut:.1f}\n(only {MAIN} plain\nstars get past it)',
        va='top', fontsize=8.5, color=INK2)
for nm, g in [('J2133+83', 'warm'), ('J0845-78', 'cool/flat')]:
    x = ok.loc[ok['name'] == nm, METRIC].iloc[0]
    s = ok.loc[ok['group_name'] == g, METRIC]
    y = 100 * (s >= x).mean()
    a1.plot(x, y, 'o', ms=8, color=COLOR[g], mec=BG, mew=2, zorder=5)
    left = nm == 'J2133+83'     # J2133+83 label to the left, J0845-78 to the right, so they don't overlap
    a1.annotate(nm, (x, y), xytext=(-8, -14) if left else (8, 8), textcoords='offset points',
                ha='right' if left else 'left', fontsize=9, color=INK)
a1.set_xscale('log')
a1.set_xlim(0.1, 50)
a1.set_xticks([0.1, 0.3, 1, 3, 10, 30], ['0.1', '0.3', '1', '3', '10', '30'])
a1.xaxis.set_minor_locator(plt.NullLocator())
a1.set_ylim(0, 102)
a1.set_xlabel('24 µm offset from the star, in units of its expected wobble')
a1.set_ylabel('% of stars at least this far off-center')
a1.set_title('How far off-center is the 24 µm light?', loc='left', fontsize=11.5, color=INK)
a1.legend(frameon=False, fontsize=9, loc='lower left')

# right: % of each group past the cutoff
m = out[out['cutoff'] == MAIN].set_index('group').loc[GROUPS]
x = np.arange(len(GROUPS))
a2.bar(x, m['percent'], width=0.55, color=[COLOR[g] for g in GROUPS], zorder=2)
a2.errorbar(x, m['percent'], yerr=[m['percent'] - m['low_1sigma'], m['high_1sigma'] - m['percent']],
            fmt='none', ecolor=INK2, elinewidth=1.2, capsize=4, zorder=3)
for i, g in enumerate(GROUPS):
    a2.text(i, m.loc[g, 'high_1sigma'] + 1.5, f"{m.loc[g, 'percent']:.0f}%\n"
            f"{m.loc[g, 'off_center']} of {m.loc[g, 'stars']}", ha='center', va='bottom', fontsize=9, color=INK)
a2.set_xticks(x, ['Plain stars\n(control)', 'Warm\ngroup', 'Cool/flat\ngroup'])
a2.set_ylim(0, 55)
a2.set_ylabel('% off-center (past the cutoff)')
a2.set_title('Off-center past the cutoff (error bars: 1 sigma)', loc='left', fontsize=11.5, color=INK)

for ax in (a1, a2):
    ax.set_facecolor(BG)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(FIGS / 'control_cutoff.png', dpi=200, facecolor=BG)
print('Saved figures/control_cutoff.png')
