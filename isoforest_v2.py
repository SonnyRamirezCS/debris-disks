"""
Isolation Forest redo (version 2).

Changes from isoforest_demo.py, based on what Tyler said:
1. Signal to noise filter: skip stars whose 24 micron measurement is too noisy
2. Only keep flagged stars with a REAL 24 micron excess (chi_24 >= 3, Skorpen's
   3 sigma). Stars flagged for weird 8 micron values with no 24 excess aren't disks.
3. Run it twice: once with young stars left in, once on the clean stars only
   (from sfr_cut.py), so we can compare
4. Compare the Isolation Forest picks to the chi method picks (Tyler's method)

Run from the debris-disks folder:  python isoforest_v2.py
Needs data\\kdwarf_sfr.csv (made by sfr_cut.py)
Makes figures\\isoforest_v2.png and data\\isoforest_v2_picks.csv
"""
import warnings

import matplotlib
matplotlib.use('Agg')  # just save the plot, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

# ---- Settings (ask Skorpen about SNR_MIN) ----
FEATURES = ['norm_chi24', 'norm_chi8', 'norm_chi45']
SNR_MIN = 5          # 24 micron flux must be at least this many times its error
BRIGHT_MIN = 3       # chi_24 must be at least 3 (Skorpen's 3 sigma) to count as too bright
CONTAMINATION = 0.05  # Isolation Forest flags about this fraction as weird
FLUX, ERR = 'm1_f_psf', 'm1_df_psf'  # 24 micron flux and its error

df = pd.read_csv(r'data\kdwarf_sfr.csv')
df['young'] = df['young'].astype(bool)
df['evolved'] = df['evolved'].astype(bool)
df['clean'] = df['clean'].astype(bool)

if FLUX in df.columns and ERR in df.columns:
    df['snr24'] = df[FLUX] / df[ERR]
else:
    print(f"WARNING: couldn't find {FLUX} / {ERR}, so no signal to noise filter.")
    print("Columns with 'm1' in the name:", [c for c in df.columns if 'm1' in c.lower()])
    df['snr24'] = np.inf


def run(sample, label):
    s = sample.dropna(subset=FEATURES)
    n_start = len(s)
    s = s[s['snr24'] >= SNR_MIN].copy()
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42).fit(s[FEATURES])
    s['weirdness'] = -iso.score_samples(s[FEATURES])  # higher = weirder
    s['flagged'] = iso.predict(s[FEATURES]) == -1
    s['bright_pick'] = s['flagged'] & (s['chi_24'] >= BRIGHT_MIN)

    picks = s[s['bright_pick']].sort_values('weirdness', ascending=False)
    chi_top = s.sort_values('chi_24', ascending=False).head(max(len(picks), 1))
    both = set(picks['objid']) & set(chi_top['objid'])

    print(f"\n========== {label} ==========")
    print(f"Stars with all features: {n_start}")
    print(f"After signal to noise >= {SNR_MIN} at 24 microns: {len(s)}")
    print(f"Isolation Forest flagged: {s['flagged'].sum()}")
    print(f"  real 24 micron excess, chi_24 >= {BRIGHT_MIN} (what we want): {len(picks)}")
    print(f"  no real 24 micron excess (thrown out): {(s['flagged'] & ~s['bright_pick']).sum()}")
    print(f"  young stars among the picks: {picks['young'].sum()} of {len(picks)}")
    print(f"  picks that survive all the cuts in sfr_cut.py: {picks['clean'].sum()} of {len(picks)}")
    print(f"Same stars as the top {len(chi_top)} by chi method: {len(both)} of {len(picks)}")
    cols = ['objid', 'Teff', 'chi_8', 'chi_24', 'snr24', 'weirdness', 'young', 'clean', 'otype']
    cols = [c for c in cols if c in s.columns]
    print(picks[cols].round(2).to_string())
    return s, picks


with_young, picks_a = run(df[~df['evolved']], "RUN A: young stars left in")
clean, picks_b = run(df[df['clean']], "RUN B: clean stars only")

picks_b.assign(run='clean').to_csv(r'data\isoforest_v2_picks.csv', index=False)
print("\nSaved data\\isoforest_v2_picks.csv (Run B picks)")

# ---- Plot: 8 micron excess vs 24 micron excess, both runs side by side ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, s, title in [(axes[0], with_young, 'Young stars left in'),
                     (axes[1], clean, 'Clean stars only')]:
    normal = ~s['flagged']
    faint = s['flagged'] & ~s['bright_pick']
    ax.scatter(s.loc[normal, 'norm_chi8'], s.loc[normal, 'norm_chi24'], s=8, c='tab:blue',
               alpha=0.5, label=f'Normal ({normal.sum()})')
    ax.scatter(s.loc[faint, 'norm_chi8'], s.loc[faint, 'norm_chi24'], s=20, c='gray',
               marker='x', label=f'Flagged, no real 24 micron excess ({faint.sum()})')
    ax.scatter(s.loc[s['bright_pick'], 'norm_chi8'], s.loc[s['bright_pick'], 'norm_chi24'],
               s=40, c='tab:red', marker='*',
               label=f"Flagged, chi_24 >= {BRIGHT_MIN} ({s['bright_pick'].sum()})")
    ax.set_title(title)
    ax.set_xlabel('Normalized chi_8 (8 micron excess)')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Normalized chi_24 (24 micron excess)')
fig.suptitle(f'Isolation Forest, signal to noise >= {SNR_MIN} at 24 microns')
fig.tight_layout()
fig.savefig(r'figures\isoforest_v2.png', dpi=150)
print("Saved figures\\isoforest_v2.png")