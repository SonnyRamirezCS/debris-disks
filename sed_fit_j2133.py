"""
SED check for J2133+83 (SSTSL2 J213307.73+830501.2).

1. Predict the star's own (photospheric) brightness at 4.5-24 um three ways:
   a) Rayleigh-Jeans from IRAC 3.6, corrected with the median measured/RJ ratio
      of the 28 'plain' K dwarfs in our own sample (Ks-[24] < 0.75)
   b) Same, anchored on WISE W1 instead of IRAC 3.6
   c) Empirical Ks-[24] color of those same plain K dwarfs
2. Excess = measured - predicted, at 8, 12 (WISE W3) and 24 um.
3. Fit a single-temperature blackbody to the excess -> dust temperature,
   fractional luminosity f = L_dust/L_star, and blackbody dust distance.
All fluxes in microjanskys.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

h, c, k = 6.626e-34, 2.998e8, 1.381e-23

def bnu(lam_um, T):
    nu = c / (lam_um * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k * T))

# ---- measured photometry (uJy) ----
# SEIP (Spitzer) and 2MASS values from data/isoforest_v2_picks.csv in the repo
J, eJ = 4742.0, 122.3
H, eH = 5420.0, 184.7
K, eK = 3901.0, 129.4
I1, eI1 = 1692.0, 3.246
I2, eI2 = 1220.0, 3.391
I3, eI3 = 793.8, 6.796
I4, eI4 = 546.5, 10.92
M1, eM1 = 509.0, 9.289
# AllWISE (row at 0.3" in the CDS portal): W1 12.936+-0.023, W2 12.935+-0.024,
# W3 11.83+-0.225, W4 9.4 (no error -> upper limit). Vega zero points in Jy.
def wise(m, zp): return zp * 10**(-m / 2.5) * 1e6
W1, W2, W3 = wise(12.936, 309.54), wise(12.935, 171.787), wise(11.83, 31.674)
eW1, eW2 = W1 * 0.023 / 1.0857, W2 * 0.024 / 1.0857
W3_hi, W3_lo = wise(11.83 - 0.225, 31.674), wise(11.83 + 0.225, 31.674)
eW3 = (W3_hi - W3_lo) / 2
W4_ul = wise(9.4, 8.363)

lam = {'J': 1.235, 'H': 1.662, 'K': 2.159, 'W1': 3.353, 'I1': 3.550, 'I2': 4.493,
       'W2': 4.603, 'I3': 5.731, 'I4': 7.872, 'W3': 11.56, 'W4': 22.08, 'M1': 23.68}

# ---- empirical correction factors (measured / RJ-from-3.6) from 28 plain K dwarfs ----
corr_lam = np.array([3.550, 4.493, 5.731, 7.872, 23.68])
corr_val = np.array([1.000, 1.034, 1.112, 1.208, 1.329])
corr_sd = {'I4': 0.072, 'M1': 0.225}

def corr(l):
    return np.interp(np.log(l), np.log(corr_lam), corr_val)

def phot_from(anchor_flux, anchor_lam, l):
    return anchor_flux * (anchor_lam / l)**2 * corr(l) / corr(anchor_lam)

# three predictions at 24 um
pred24_a = phot_from(I1, lam['I1'], lam['M1'])
pred24_b = phot_from(W1, lam['I1'], lam['M1']) * (lam['W1'] / lam['I1'])**2  # move W1 to 3.55 by RJ first
Ks_mag = -2.5 * np.log10(K * 1e-6 / 666.7)
pred24_c = 7.17 * 10**(-(Ks_mag - 0.263) / 2.5) * 1e6
preds = np.array([pred24_a, pred24_b, pred24_c])
pred24 = preds.mean()

print('--- predicted star-only flux at 24 um ---')
for name, p in zip(['RJ from IRAC 3.6 (calibrated)', 'RJ from WISE W1 (calibrated)',
                    'empirical Ks-[24] = 0.26'], preds):
    print(f'  {name:32s} {p:6.1f} uJy  -> excess ratio {M1 / p:4.1f}x')
print(f'  adopted {pred24:.1f} uJy, ratio {M1 / pred24:.1f}x (range {M1 / preds.max():.1f}-{M1 / preds.min():.1f}x)')

# uncertainty on the 24 um excess: photometry + 4% MIPS calibration + spread of plain stars
e_ex24 = np.sqrt(eM1**2 + (0.04 * M1)**2 + (corr_sd['M1'] / corr_val[-1] * pred24)**2)
ex24 = M1 - pred24
print(f'  24 um excess = {ex24:.0f} +- {e_ex24:.0f} uJy  ({ex24 / e_ex24:.0f} sigma)')

# shorter wavelengths, scaled with the same anchor average
scale = pred24 / pred24_a
pred = {b: phot_from(I1, lam['I1'], lam[b]) * scale for b in ['I2', 'W2', 'I3', 'I4', 'W3', 'W4']}
meas = {'I2': (I2, eI2), 'W2': (W2, eW2), 'I3': (I3, eI3), 'I4': (I4, eI4), 'W3': (W3, eW3)}
print('--- other bands (measured / predicted) ---')
ex = {}
for b, (f, e) in meas.items():
    sd = 0.072 if b in ('I4',) else 0.05
    cal = 0.03 if b.startswith('I') else 0.024
    err = np.sqrt(e**2 + (cal * f)**2 + (sd * pred[b])**2)
    ex[b] = (f - pred[b], err)
    print(f'  {b}: {f:7.1f} vs {pred[b]:7.1f}  ratio {f / pred[b]:.2f}  excess {f - pred[b]:6.0f} +- {err:4.0f} ({(f - pred[b]) / err:4.1f} sigma)')
print(f'  W4 upper limit {W4_ul:.0f} uJy vs predicted star {pred["W4"]:.0f} uJy')

# ---- dust blackbody fit to the excess at 8, 12, 24 um ----
pts = [(lam['I4'], *ex['I4']), (lam['W3'], *ex['W3']), (lam['M1'], ex24, e_ex24)]
Ts = np.arange(40, 800, 1.0)
chi = []
for T in Ts:
    model = np.array([bnu(l, T) for l, _, _ in pts])
    y = np.array([p[1] for p in pts]); s = np.array([p[2] for p in pts])
    a = np.sum(model * y / s**2) / np.sum(model**2 / s**2)
    chi.append(np.sum(((y - a * model) / s)**2))
chi = np.array(chi)
Tbest = Ts[np.argmin(chi)]
ok = Ts[chi < chi.min() + 1]
print(f'--- dust blackbody ---\n  best T = {Tbest:.0f} K (1-sigma range {ok.min():.0f}-{ok.max():.0f} K), chi2 = {chi.min():.2f} for 1 dof')

# fit with only the solid 24 um point + 22 um limit can't fix T, so also report f for a range of T
L_star = 0.24  # Lsun, from Gaia: M_G = 6.45 after extinction, BC_G ~ -0.16
Teff = 4636.0

# star's bolometric flux from Gaia G = 15.483, A_G = 0.273, BC_G ~ -0.16 (Teff 4636 K, Andrae et al. 2018)
F_star = 2.518e-8 * 10**(-0.4 * (15.483 - 0.273 - 0.16))  # W/m^2

def frac_lum(T):
    # dust: blackbody scaled to the 24 um excess, integrated over all wavelengths
    om_d = ex24 * 1e-32 / bnu(lam['M1'], T)
    F_dust = om_d * 5.670e-8 * T**4 / np.pi
    return F_dust / F_star

def r_bb(T):
    return (278.3 / T)**2 * np.sqrt(L_star)

for T in sorted(set([int(Tbest), int(ok.min()), int(ok.max()), 100, 150, 200, 300])):
    print(f'  T = {T:4d} K: f = L_dust/L_star = {frac_lum(T):.1e}, blackbody distance from star = {r_bb(T):.2f} AU')

# ---- figure ----
fig, ax = plt.subplots(figsize=(7.5, 5))
lgrid = np.logspace(np.log10(1.0), np.log10(40), 300)
ph = np.where(lgrid >= 3.55, phot_from(I1, 3.55, np.maximum(lgrid, 3.55)) * scale,
              I1 * scale * bnu(lgrid, Teff) / bnu(3.55, Teff))
aT = ex24 / bnu(lam["M1"], Tbest)
dust = aT * bnu(lgrid, Tbest)
nu = lambda l: c / (l * 1e-6)
conv = lambda l, f: nu(l) * f * 1e-32  # nuFnu in W/m^2
ax.plot(lgrid, conv(lgrid, ph), color='0.45', lw=1.6, label='Star alone (predicted)')
ax.plot(lgrid, conv(lgrid, dust), color='#c2410c', lw=1.2, ls='--', label=f'Dust blackbody, {Tbest:.0f} K')
ax.plot(lgrid, conv(lgrid, ph + dust), color='#1d4ed8', lw=1.2, label='Star + dust')
pts_plot = [('2MASS', ['J', 'H', 'K'], [J, H, K], [eJ, eH, eK], 'o', '0.2'),
            ('WISE', ['W1', 'W2', 'W3'], [W1, W2, W3], [eW1, eW2, eW3], 's', '#7c3aed'),
            ('Spitzer IRAC', ['I1', 'I2', 'I3', 'I4'], [I1, I2, I3, I4], [eI1, eI2, eI3, eI4], 'D', '#0f766e'),
            ('Spitzer MIPS 24', ['M1'], [M1], [eM1], '*', '#b91c1c')]
for lab, bands, fl, er, mk, col in pts_plot:
    L = np.array([lam[b] for b in bands]); F = np.array(fl); E = np.array(er)
    ax.errorbar(L, conv(L, F), yerr=conv(L, E), fmt=mk, color=col, ms=9 if mk == '*' else 6,
                label=lab, capsize=2, lw=1)
ax.errorbar([lam['W4']], [conv(lam['W4'], W4_ul)], yerr=[conv(lam['W4'], W4_ul) * 0.35], uplims=[True],
            fmt='v', color='#7c3aed', ms=6, label='WISE 22 um (upper limit)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1, 40); ax.set_ylim(1e-17, 3e-14)
ax.set_xlabel('Wavelength (microns)'); ax.set_ylabel(r'$\nu F_\nu$ (W m$^{-2}$)')
ax.set_title(f'J2133+83: about {M1 / pred24:.0f}x brighter than the star alone at 24 microns')
ax.legend(fontsize=8, loc='lower left', frameon=False)
ax.grid(alpha=0.2, which='both')
plt.tight_layout()
plt.savefig('J2133+83_SED.png', dpi=160)
print('saved J2133+83_SED.png')
