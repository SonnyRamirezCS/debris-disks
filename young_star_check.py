"""
Label young stars using the SIMBAD cross match, then:
1. See if young stars explain the two humps in norm_chi24
2. Rerun Isolation Forest on the stars that are NOT young or giants
Run from the debris-disks folder:  python young_star_check.py
Needs data\\kdwarf_simbad.csv (the CDS XMatch download)
"""
import contextlib
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# SIMBAD object type codes
YOUNG_CODES = ['Y*O', 'Y*?', 'Or*', 'TT*', 'TT?', 'Ae*', 'Ae?', 'pr*', 'pr?']  # young stars
EVOLVED_CODES = ['LP*', 'LP?', 's?r', 'RG*', 'AB*', 'C*', 'Mi*']              # old giants
FEATURES = ['norm_chi24', 'norm_chi8', 'norm_chi45']

# ---- Load Tyler's K dwarf table (quietly) ----
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    from tyler_kstar_features import df_kdwarf

# ---- SIMBAD cross match: keep the closest match for each star ----
sim = pd.read_csv(r'data\kdwarf_simbad.csv')
sim = sim.sort_values('angDist').drop_duplicates('objid')


def has_code(row, codes):
    types = [str(row['otype'])] + str(row['other_types']).split('|')
    return any(t in codes for t in types)


sim['young'] = sim.apply(has_code, axis=1, codes=YOUNG_CODES)
sim['evolved'] = sim.apply(has_code, axis=1, codes=EVOLVED_CODES)

df = df_kdwarf.merge(sim[['objid', 'otype', 'main_id', 'young', 'evolved']],
                     on='objid', how='left')
df['in_simbad'] = df['otype'].notna()
df['young'] = df['young'].eq(True)
df['evolved'] = df['evolved'].eq(True)

print(f"K dwarfs in sample: {len(df)}")
print(f"Found in SIMBAD: {df['in_simbad'].sum()}")
print(f"Tagged young (confirmed or candidate): {df['young'].sum()} ({df['young'].mean():.0%} of sample)")
print(f"Tagged as evolved giants: {df['evolved'].sum()}")

df.to_csv(r'data\kdwarf_labeled.csv', index=False)
print(r"Saved data\kdwarf_labeled.csv")

# ---- Do young stars explain the two humps? ----
h = df.dropna(subset=['norm_chi24'])
bins = np.linspace(h['norm_chi24'].min(), h['norm_chi24'].max(), 31)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.hist(h.loc[~h['young'], 'norm_chi24'], bins=bins, alpha=0.6,
         label=f"Not tagged young ({(~h['young']).sum()})")
ax1.hist(h.loc[h['young'], 'norm_chi24'], bins=bins, alpha=0.6,
         label=f"Young stars in SIMBAD ({h['young'].sum()})")
ax1.set_xlabel('Normalized chi_24 (24 micron excess)')
ax1.set_ylabel('Number of stars')
ax1.set_title('Where do the young stars fall?')
ax1.legend()
fig1.tight_layout()
fig1.savefig(r'figures\young_vs_chi24.png', dpi=200)
print(r"Saved figures\young_vs_chi24.png")

# ---- Rerun Isolation Forest without young stars and giants ----
clean = df[~df['young'] & ~df['evolved']].dropna(subset=FEATURES).copy()
model = IsolationForest(contamination=0.02, random_state=42)
clean['flag'] = model.fit_predict(clean[FEATURES])
clean['score'] = -model.score_samples(clean[FEATURES])

flagged = clean[clean['flag'] == -1].sort_values('score', ascending=False)
high = flagged[flagged['norm_chi24'] > 0]
cols = ['objid', 'Teff', 'chi_8', 'chi_24', 'norm_chi8', 'norm_chi24', 'otype']

print(f"\nClean sample (young stars and giants removed): {len(clean)}")
print(f"Isolation Forest flagged {len(flagged)}, {len(high)} of them with extra 24 micron light:")
print(high[cols].to_string())

print("\n--- Top 15 by 24 micron excess after cleaning ---")
print(clean.sort_values('norm_chi24', ascending=False).head(15)[cols].to_string())

plt.show()