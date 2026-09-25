import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv(r'data\SEIP_data.csv')

# ---------- 1. find the star and plot the data ----------
# SSTSL2 J012412.12-741735.2 -> RA 01h24m12.12s, Dec -74d17m35.2s
ra_star = (1 + 24/60 + 12.12/3600) * 15
dec_star = -(74 + 17/60 + 35.2/3600)

# pick the closest star in the dataset to those coordinates
sep = np.sqrt(((df['ra'] - ra_star) * np.cos(np.radians(dec_star)))**2 + (df['dec'] - dec_star)**2) * 3600
star = df.loc[sep.idxmin()]
print('Matched star:', star['objid'], ' separation:', round(sep.min(), 2), 'arcsec')
print('Teff from dataset:', star['Teff'], 'K')

cols = ['j', 'h', 'k', 'i1_f_ap2', 'i2_f_ap2', 'i3_f_ap2', 'i4_f_ap2', 'm1_f_psf']
x = np.array([1.235, 1.662, 2.159, 3.6, 4.5, 5.8, 8.0, 24.0])   # wavelength (microns)
y = star[cols].to_numpy(dtype=float) * 1e-6                        # flux density (Jy)
print('Flux (Jy):', y)

plt.figure()
plt.scatter(x, y, color='black', label='2MASS + Spitzer data')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux density (Jy)')
plt.title('Photometry of SSTSL2 J012412.12-741735.2')
plt.legend()
plt.savefig(r'figures\step5_data.png', dpi=200)

xfit = np.linspace(1, 25, 500)   # smooth x values for drawing the fits

# ---------- 2. line ----------
def line(x, m, b):
    return m * x + b

(m, b), _ = curve_fit(line, x, y)
print(f'Line: m = {m:.4g}, b = {b:.4g}')
print('Line residuals:', y - line(x, m, b))

plt.figure()
plt.scatter(x, y, color='black', label='Data')
plt.plot(xfit, line(xfit, m, b), label=f'Line: y = {m:.3g}x + {b:.3g}')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux density (Jy)')
plt.title('Line fit')
plt.legend()
plt.savefig(r'figures\step5_line.png', dpi=200)

# ---------- 3. parabola ----------
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

(a, b2, c), _ = curve_fit(parabola, x, y)
print(f'Parabola: a = {a:.4g}, b = {b2:.4g}, c = {c:.4g}')

plt.figure()
plt.scatter(x, y, color='black', label='Data')
plt.plot(xfit, parabola(xfit, a, b2, c), label=f'Parabola: y = {a:.3g}x² + {b2:.3g}x + {c:.3g}')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux density (Jy)')
plt.title('Parabola fit')
plt.legend()
plt.savefig(r'figures\step5_parabola.png', dpi=200)

# ---------- 4. blackbody ----------
h = 6.626e-34    # Planck's constant (J s)
c_light = 2.998e8  # speed of light (m/s)
kB = 1.381e-23   # Boltzmann's constant (J/K)

def planck_jy(x_um, T):
    lam = x_um * 1e-6   # microns -> meters
    # per-frequency Planck function (W/m^2/Hz/sr), times 1e26 to get Jy
    return 2 * h * c_light / lam**3 / (np.exp(h * c_light / (lam * kB * T)) - 1) * 1e26

def blackbody(x, T, logN):
    # N is tiny, so fit log10(N) instead to keep curve_fit happy
    return 10**logN * planck_jy(x, T)

logN0 = np.log10(y[0] / planck_jy(x[0], 5000))   # rough starting guess
(T, logN), _ = curve_fit(blackbody, x, y, p0=[5000, logN0])
N = 10**logN
print(f'Blackbody: T = {T:.0f} K, N = {N:.3e}')

plt.figure()
plt.scatter(x, y, color='black', label='Data')
plt.plot(xfit, blackbody(xfit, T, logN), color='red', label=f'Blackbody: T = {T:.0f} K, N = {N:.2e}')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux density (Jy)')
plt.title('Blackbody fit')
plt.legend()
plt.savefig(r'figures\step5_blackbody.png', dpi=200)

# ---------- final: everything on log-log ----------
plt.figure()
plt.scatter(x, y, color='black', label='Data', zorder=3)
plt.plot(xfit, line(xfit, m, b), label='Line fit')
plt.plot(xfit, parabola(xfit, a, b2, c), label='Parabola fit')
plt.plot(xfit, blackbody(xfit, T, logN), color='red', label=f'Blackbody fit (T = {T:.0f} K)')
plt.loglog()
plt.ylim(y.min() / 5, y.max() * 5)
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux density (Jy)')
plt.title('SED of SSTSL2 J012412.12-741735.2 with all fits')
plt.legend()
plt.savefig(r'figures\step5_all_fits.png', dpi=200)
plt.show()