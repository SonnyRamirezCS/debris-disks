import numpy as np, pandas as pd
ks = pd.read_csv(r'data\kstars_reduced.csv')
targets = {'J205152.89+441557.4': (312.9704, 44.2659), 'J054210.35-095838.2': (85.5431, -9.9773)}
for name, (ra, dec) in targets.items():
    sep = np.hypot((ks['ra'] - ra) * np.cos(np.radians(dec)), ks['dec'] - dec) * 3600
    print(name, 'IN sample' if sep.min() < 2 else 'NOT in sample', round(sep.min(), 1), 'arcsec')