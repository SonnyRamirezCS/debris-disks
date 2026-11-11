"""
Do the warm and cool/flat groups hold up? (version 1)

Why: excess_shape.py Part B splits the 116 stars with a real 24 um excess into
2 groups with KMeans: warm (also bright at 8 um, like J2133+83) and cool/flat
(extra light mostly at 24 um, like J0845-78). Before showing that on Dec 18 we
need to know the split isn't a fluke of one random seed or one method.

Tests (each one compared to the groups saved by excess_shape.py):
  1. KMeans again with 50 different random seeds, 1 start each
  2. Bootstrap: pick 116 stars at random (repeats allowed), refit, 200 times
  3. Hierarchical clustering (Ward), a different method with no randomness
  4. Gaussian Mixture, a different method that allows stretched groups, 20 seeds
  5. KMeans with 3 groups instead of 2
  6. KMeans on the shape only (drops the 24 um size)

Agreement = adjusted Rand index (ARI). 1 = exactly the same groups,
0 = no better than random.
A group counts as "warm" when its typical (median) 8 um shape is closer to the
original warm group's than to the original cool/flat group's.

Run from the debris-disks folder:  python cluster_stability.py
Needs data/excess_shape.csv (made by excess_shape.py)
Makes data/cluster_stability.csv and figures/cluster_stability.png

Version 2 groups (excess_shape_v2.py, no 4.5 um shape):  python cluster_stability.py --version 2
Needs data/excess_shape_v2.csv, makes data/cluster_stability_v2.csv and figures/cluster_stability_v2.png.
Same tests; test 6 ("shape only") then uses the 8 um shape alone.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # quiets a harmless KMeans warning on Windows

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
DATA, FIGS = HERE / 'data', HERE / 'figures'

# ---- Settings ----
SEEDS = 50
BOOTSTRAPS = 200
GMM_SEEDS = 20
ap = argparse.ArgumentParser()
ap.add_argument('--version', type=int, choices=[1, 2], default=1,
                help='1 = groups from excess_shape.py, 2 = groups from excess_shape_v2.py')
VERSION = ap.parse_args().version
TAG = '' if VERSION == 1 else '_v2'
if VERSION == 1:
    FEAT = ['f_shape_I2', 'f_shape_I4', 'f_size']   # same features as excess_shape.py Part B
    SHAPE_ONLY = ['f_shape_I2', 'f_shape_I4']
else:
    FEAT = ['f_shape_I4', 'f_size']                 # same features as excess_shape_v2.py Part B
    SHAPE_ONLY = ['f_shape_I4']
NAMED = {'SSTSL2 J213307.73+830501.2': 'J2133+83', 'SSTSL2 J084517.41-785937.1': 'J0845-78'}

# plot colors: J2133+83 in the warm group's orange, J0845-78 in the cool/flat group's blue
# (same as vetting/control_cutoff.py)
COLOR = {'J2133+83': '#eb6834', 'J0845-78': '#2a78d6'}
INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'

# ---- load and rebuild the features exactly like excess_shape.py ----
print(f'Version {VERSION} groups (features: {", ".join(FEAT)})')
d = pd.read_csv(DATA / f'excess_shape{TAG}.csv')
B = d[d['shape_group'].notna()].reset_index(drop=True)
B['f_shape_I2'] = B['shape_I2'].clip(-1, 3)
B['f_shape_I4'] = B['shape_I4'].clip(-1, 3)
B['f_size'] = np.log10(B['ratio_M1'])
med = B[FEAT].median()
X = (B[FEAT] - med) / (1.4826 * (B[FEAT] - med).abs().median())
base = B['shape_group'].astype(int).to_numpy()
star = {nm: B.index[B['objid'] == objid][0] for objid, nm in NAMED.items()}

shape8 = B['shape_I4'].to_numpy()
typical = pd.Series(shape8).groupby(base).median()
WARM_LINE = typical.mean()     # halfway between the two original groups' typical 8 um shape
print(f'{len(B)} stars with a real 24 um excess. Original groups from excess_shape.py: ' +
      ', '.join(f'group {g}: {(base == g).sum()} stars, typical 8 um shape {typical[g]:.2f}' for g in typical.index))
print(f'Silhouette score of the original split: {silhouette_score(X, base):.2f} '
      '(1 = perfectly separated, 0 = groups overlap)')
print(f'A group counts as warm if its typical 8 um shape is above {WARM_LINE:.2f}')


def warm_mask(labels):
    """True for every star that lands in a warm-looking group."""
    t = pd.Series(shape8).groupby(labels).median()
    return np.isin(labels, t[t > WARM_LINE].index)


def kmeans(k, X_fit, seed, n_init):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X_fit)


# ---- run every test, one row per run ----
runs = []


def record(test, labels):
    w = warm_mask(labels)
    runs.append(dict(test=test, ari=adjusted_rand_score(base, labels), warm_size=int(w.sum()),
                     **{nm: bool(w[i]) for nm, i in star.items()}))


TESTS = [f'KMeans, {SEEDS} random seeds', f'Bootstrap, {BOOTSTRAPS} resamples', 'Hierarchical (Ward)',
         f'Gaussian Mixture, {GMM_SEEDS} seeds', 'KMeans with 3 groups', 'KMeans, shape only']

for s in range(SEEDS):
    record(TESTS[0], kmeans(2, X, s, n_init=1).labels_)

rng = np.random.default_rng(0)
for b in range(BOOTSTRAPS):
    pick = rng.choice(len(X), len(X), replace=True)
    record(TESTS[1], kmeans(2, X.iloc[pick], b, n_init=10).predict(X))

record(TESTS[2], AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(X))

for s in range(GMM_SEEDS):
    record(TESTS[3], GaussianMixture(n_components=2, n_init=3, random_state=s).fit(X).predict(X))

record(TESTS[4], kmeans(3, X, 0, n_init=20).labels_)
record(TESTS[5], kmeans(2, X[SHAPE_ONLY], 0, n_init=20).labels_)

R = pd.DataFrame(runs)
summ = R.groupby('test', sort=False).agg(
    runs=('ari', 'size'),
    agreement=('ari', 'median'),
    agreement_low=('ari', lambda a: a.quantile(0.1)),
    agreement_high=('ari', lambda a: a.quantile(0.9)),
    warm_size=('warm_size', 'median'),
    **{f'{nm}_warm_pct': (nm, lambda a: 100 * a.mean()) for nm in NAMED.values()}).reindex(TESTS)

print('\n========== How well each test agrees with the original groups ==========')
print('agreement = ARI (median, and the range of the middle 80% of runs)')
for t, r in summ.iterrows():
    rng_txt = f' ({r.agreement_low:.2f} to {r.agreement_high:.2f})' if r.runs > 1 else ''
    print(f'  {t:28s} agreement {r.agreement:.2f}{rng_txt:16s} warm group ~{r.warm_size:3.0f} stars | '
          + ' | '.join(f'{nm} warm in {r[f"{nm}_warm_pct"]:3.0f}% of runs' for nm in NAMED.values()))

summ.round(3).to_csv(DATA / f'cluster_stability{TAG}.csv')
print(f'\nSaved data/cluster_stability{TAG}.csv')

# ---- figure ----
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': INK2, 'text.color': INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
fig.patch.set_facecolor(BG)
y = np.arange(len(TESTS))[::-1]

# left: agreement with the original groups
for yi, (t, r) in zip(y, summ.iterrows()):
    if r.runs > 1:
        a1.plot([r.agreement_low, r.agreement_high], [yi, yi], color=MUTED, lw=2, solid_capstyle='round')
    a1.plot(r.agreement, yi, 'o', ms=8, color=INK2, zorder=3)
    a1.text(r.agreement + 0.03, yi + 0.2, f'{r.agreement:.2f}', fontsize=9, color=INK)
a1.set_xlim(0, 1.08)
a1.set_xlabel('Agreement with the original groups (ARI: 1 = identical)')
a1.set_title(f'Do the same groups come back? (version {VERSION})', loc='left', fontsize=11.5, color=INK)
a1.set_yticks(y, TESTS)

# right: how often each star lands in the warm group
for nm, dy in [('J2133+83', 0.14), ('J0845-78', -0.14)]:
    a2.plot(summ[f'{nm}_warm_pct'], y + dy, 'o', ms=8, color=COLOR[nm], label=nm, zorder=3)
a2.set_xlim(-5, 105)
a2.set_xlabel('% of runs where the star lands in the warm group')
a2.set_title('Which group do our two stars land in?', loc='left', fontsize=11.5, color=INK)
a2.legend(frameon=False, fontsize=9, loc='center')   # the middle of this panel is empty

for ax in (a1, a2):
    ax.set_facecolor(BG)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylim(-0.6, len(TESTS) - 0.4)
fig.tight_layout()
fig.savefig(FIGS / f'cluster_stability{TAG}.png', dpi=200, facecolor=BG)
print(f'Saved figures/cluster_stability{TAG}.png')
