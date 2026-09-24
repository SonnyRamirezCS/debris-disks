import pandas as pd
import matplotlib.pyplot as plt

csv_filepath = r'data\SEIP_data.csv'
df = pd.read_csv(csv_filepath).copy()
# spectral types from Mamajek temperature ranges (Kelvin)
bins = [2350, 3930, 5380, 5990, 7400, 10400, 31900, float('inf')]
types = ['M', 'K', 'G', 'F', 'A', 'B', 'O']
df['spectype'] = pd.cut(df['Teff'], bins=bins, labels=types, right=False)

# how many stars in each type
print(df['spectype'].value_counts().reindex(types[::-1]))
print('No Teff or out of range:', df['spectype'].isna().sum())

# hot = blue, cool = red
colors = {'O': 'blue', 'B': 'deepskyblue', 'A': 'cyan', 'F': 'green',
          'G': 'gold', 'K': 'orange', 'M': 'red'}

plt.figure(figsize=(10, 5))
for t in types[::-1]:
    sub = df[df['spectype'] == t]
    plt.scatter(sub['ra'], sub['dec'], s=2, color=colors[t], label=t)

plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.title('Dec vs RA of SEIP stars by spectral type')
plt.legend(title='Spectral type', markerscale=5)
plt.savefig(r'figures\dec_vs_ra.png', dpi=200)
plt.show()