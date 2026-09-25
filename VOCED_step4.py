import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'data\SEIP_data.csv')
print('Starting stars:', len(df))

# quick check of what the Galactic/extragalactic column looks like
print(df['mips_obstype'].value_counts(dropna=False))

# filter 1: Galactic sources only (0 = Galactic, 1 = Extragalactic)
f1 = df[df['mips_obstype'] == 0]
print('After mips_obstype = 0:', len(f1))

# filter 2: K-type temperature range
f2 = f1[(f1['Teff'] > 3930) & (f1['Teff'] < 5380)]
print('After 3930 K < Teff < 5380 K:', len(f2))

# filter 3: main sequence only (high surface gravity, no giants)
kstars = f2[f2['logg'] > 3.8].copy()
print('After logg > 3.8:', len(kstars))

# save the reduced sample for later steps
kstars.to_csv(r'data\kstars_reduced.csv', index=False)

# plot: left = stars left after each filter, right = Teff of the final sample
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['All stars', 'Galactic', 'K-type Teff', 'Main sequence']
counts = [len(df), len(f1), len(f2), len(kstars)]
bars = ax1.bar(labels, counts, color=['gray', 'steelblue', 'orange', 'darkorange'])
ax1.bar_label(bars)
ax1.set_ylabel('Number of stars')
ax1.set_title('Stars left after each filter')

ax2.hist(kstars['Teff'], bins=30, color='orange', edgecolor='black')
ax2.set_xlabel('Teff (K)')
ax2.set_ylabel('Number of stars')
ax2.set_title('Temperatures of main-sequence K stars')
ax2.text(0.95, 0.95, f'N = {len(kstars)}', transform=ax2.transAxes,
         ha='right', va='top', fontsize=12, bbox=dict(facecolor='white'))

plt.tight_layout()
plt.savefig(r'figures\step4_reduction.png', dpi=200)
plt.show()