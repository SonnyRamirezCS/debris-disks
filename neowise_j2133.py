"""NEOWISE W1/W2 light curve of J2133+83, 2014-2024 visits (median of each ~1-day visit),
corrected by the median change of 20 nearby control stars of the same brightness.
Data: IRSA neowiser_p1bs_psd (qual_frame>0, qi_fact>0, saa_sep>0, moon_masked=00, nb<=1, na=0, cc_flags=0000)."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
# mjd, n, w1, e1, w2, e2, common-mode w1, common-mode w2   (from the in-browser binning)
V = np.array([[56717.2,26,12.924,0.008,12.9575,0.0134,-0.0098,-0.0102],[56908.4,32,12.925,0.009,12.9455,0.0165,-0.0152,-0.0285],
[57076.6,38,12.922,0.0065,12.94,0.0122,-0.0098,-0.0068],[57272.7,51,12.93,0.0044,12.974,0.0135,-0.01,-0.0105],
[57438.2,28,12.9335,0.0089,12.945,0.0138,-0.0057,-0.0095],[57637.4,31,12.929,0.0053,12.948,0.0186,-0.0048,-0.0117],
[57802.5,34,12.9315,0.0084,12.9875,0.0227,-0.0042,-0.0062],[58001.6,35,12.946,0.0063,12.954,0.0141,0.001,-0.0063],
[58159.7,33,12.918,0.0061,12.951,0.011,-0.0035,-0.002],[58368.6,32,12.936,0.0072,12.969,0.0215,0.001,-0.004],
[58524,35,12.946,0.0047,12.977,0.0179,0.0072,0.0098],[58732.7,31,12.944,0.0126,12.984,0.01,0.003,0.0078],
[58890.9,28,12.9355,0.0081,13.003,0.0165,0,0.0053],[59097,33,12.945,0.0074,12.993,0.0148,0.0023,0.0058],
[59255.1,24,12.9405,0.0136,12.9735,0.031,0.0055,0.0032],[59464,30,12.9335,0.0083,12.9445,0.0135,0.008,0.0077],
[59621.1,57,12.956,0.0064,12.971,0.0125,0.0028,0.0005],[59828.3,31,12.936,0.0087,12.965,0.012,0.0035,0.0015],
[59986.4,27,12.952,0.0086,12.968,0.0235,0.008,0.0068],[60193.5,56,12.948,0.0045,12.972,0.0146,0.0073,0.0122],
[60350.5,32,12.9335,0.0064,12.964,0.0167,0.0055,0.0055]])
year = 2000 + (V[:,0] - 51544.5) / 365.25
w1 = V[:,2] - V[:,6]; w2 = V[:,4] - V[:,7]
for name, y, e in (('W1', w1, V[:,3]), ('W2', w2, V[:,5])):
    w = 1/np.maximum(e, 0.003)**2; m = np.sum(w*y)/w.sum()
    A = np.vstack([np.ones_like(year), year-2019]).T; cov = np.linalg.inv(A.T @ (A*w[:,None])); p = cov @ (A.T @ (w*y))
    print(f"{name}: mean {m:.3f}, rms {np.std(y)*1000:.1f} mmag, chi2/dof {np.sum(w*(y-m)**2)/(len(y)-1):.2f}, "
          f"trend {p[1]*1000:+.2f} +- {np.sqrt(cov[1,1])*1000:.2f} mmag/yr, 3-sigma limit over 10 yr {3*np.sqrt(cov[1,1])*10*1000/1086*100:.1f}% ")
fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
for a, y, e, lab, col in ((ax[0], w1, V[:,3], 'W1 (3.4 µm)', '#2459B8'), (ax[1], w2, V[:,5], 'W2 (4.6 µm)', '#C2410C')):
    m = np.median(y)
    a.axhspan(m-0.01, m+0.01, color='0.9', zorder=0, label='±1% band')
    a.errorbar(year, y, e, fmt='o', color=col, ms=5, capsize=2, label=lab)
    a.set_ylim(m+0.06, m-0.06); a.set_ylabel('mag'); a.legend(loc='lower right', fontsize=9, frameon=False)
ax[1].set_xlabel('Year')
ax[0].set_title('J2133+83 in NEOWISE, 2014 to 2024 (one point per visit)', fontsize=12)
fig.tight_layout(); fig.savefig('neowise_j2133.png', dpi=150)
