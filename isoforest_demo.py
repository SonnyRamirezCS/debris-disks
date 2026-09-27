"""
Isolation Forest demo on Tyler's infrared excess features.
Run from the debris-disks folder:  python isoforest_demo.py
"""
import contextlib
import io
import warnings

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---- Settings ----
# Teff is left out on purpose: temperature has nothing to do with dust,
# so stars at the edge of the temperature range would get flagged for the wrong reason.
FEATURES = ['norm_chi24', 'norm_chi8', 'norm_chi45']
CONTAMINATION = 0.02   # fraction of stars the model flags as anomalies (2%)
TYLER_STARS = ['SSTSL2 J205152.89+441557.4', 'SSTSL2 J054210.35-095838.2']

# ---- Load Tyler's K dwarf table (hides his printed tables and log warnings) ----
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    from tyler_kstar_features import df_kdwarf

df = df_kdwarf.dropna(subset=FEATURES).copy()
print(f"K dwarfs in Tyler's sample: {len(df_kdwarf)}")
print(f"Used for the model (no missing features): {len(df)}")
print(f"Dropped (NaN from negative chi in the log step): {len(df_kdwarf) - len(df)}")

# ---- Fit Isolation Forest ----
model = IsolationForest(contamination=CONTAMINATION, random_state=42)
df['flag'] = model.fit_predict(df[FEATURES])       # -1 = anomaly, 1 = normal
df['score'] = -model.score_samples(df[FEATURES])   # higher = more anomalous
df['rank'] = df['score'].rank(ascending=False).astype(int)

flagged = df[df['flag'] == -1].sort_values('score', ascending=False)
print(f"\nFlagged as anomalies: {len(flagged)}")
cols = ['objid', 'Teff', 'chi_8', 'chi_24', 'norm_chi8', 'norm_chi24', 'score', 'rank']
print(flagged[cols].to_string())

# ---- Excess or deficit? Isolation Forest flags anything weird, in any direction ----
n_high = (flagged['norm_chi24'] > 0).sum()
print(f"\nFlagged stars ABOVE the median at 24 microns: {n_high}")
print(f"Flagged stars BELOW the median at 24 microns: {len(flagged) - n_high}")

# ---- Did it find Tyler's 2 stars on its own? ----
print("\n--- Tyler's candidates ---")
for name in TYLER_STARS:
    row = df[df['objid'] == name]
    if row.empty:
        print(f"{name}: not in the model sample")
    else:
        r = row.iloc[0]
        status = "FLAGGED" if r['flag'] == -1 else "not flagged"
        print(f"{name}: {status}, rank {int(r['rank'])} of {len(df)}")

# ---- Overlap with Tyler's cut (excess at both 8 and 24) ----
tyler_cut = df[(df['norm_chi24'] > 1.0) & (df['norm_chi8'] > 1.0)]
overlap = tyler_cut['flag'].eq(-1).sum()
print(f"\nTyler's cut picked {len(tyler_cut)} stars. Isolation Forest also flagged {overlap} of them.")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 6))
normal = df[df['flag'] == 1]
ax.scatter(normal['norm_chi24'], normal['norm_chi8'], s=8, c='lightgray', label='Normal')
ax.scatter(flagged['norm_chi24'], flagged['norm_chi8'], s=25, c='red',
           label=f'Flagged by Isolation Forest ({len(flagged)})')
tyler = df[df['objid'].isin(TYLER_STARS)]
ax.scatter(tyler['norm_chi24'], tyler['norm_chi8'], s=300, marker='*',
           facecolors='none', edgecolors='blue', linewidths=1.5, label="Tyler's candidates")
ax.set_xlabel('Normalized chi_24 (24 micron excess)')
ax.set_ylabel('Normalized chi_8 (8 micron excess)')
ax.set_title('Isolation Forest on K dwarf infrared excess')
ax.legend()
plt.tight_layout()
plt.savefig(r'figures\isoforest_demo.png', dpi=200)
print("\nSaved figures\\isoforest_demo.png")
plt.show()
