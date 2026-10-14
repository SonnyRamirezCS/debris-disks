"""
Redo J2133+83's numbers (SSTSL2 J213307.73+830501.2) from the current pipeline output.

sed_fit_j2133.py has the Oct 9 numbers with the plain-star calibration typed in
(1.329 at 24 um, Ks-[24] = 0.263, from 28 plain stars in the 3500-5000 K sample).
This script works those out again from whatever data/kdwarf_sfr.csv is there now,
so the numbers follow the sample instead of being copied by hand.

The star's own measurements don't change (Spitzer, 2MASS, WISE, Gaia). What changes
with the sample is the calibration (how bright a plain K dwarf is at 24 um compared
to Rayleigh-Jeans), how many stars count as comparable, and the galaxy odds.

Plain stars = clean stars where Spitzer sees the star's own 24 um light clearly,
minus the ones with excess:
  clip     keep stars within 3 robust sigma of the median measured/RJ ratio, repeat
           until nothing changes
  ks075    Ks-[24] < 0.75 mag (what temp_groups.py uses; lets in stars up to ~1.6x
           the photosphere, so it reads high)
ADOPTED: clip, with the star-only 24 um flux at least 5x its error. Faint stars only
made it into the Spitzer catalog if their 24 um flux happened to scatter up, so
including stars down to 3x makes plain stars look too bright (the script prints
this check). "Comparable stars" still uses the 3x cut, same as the checks doc.

Run from the debris-disks folder:  python j2133_numbers.py
Makes figures/J2133_SED_rerun.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=str(HERE / 'data' / 'kdwarf_sfr.csv'))
parser.add_argument('--fig', default=str(HERE / 'figures' / 'J2133_SED_rerun.png'))
args = parser.parse_args()

CANDIDATE = 'SSTSL2 J213307.73+830501.2'
TEFF_MIN, TEFF_MAX, EDGES = 3930, 5380, [4400, 4700]
GROUPS = ['Late K', 'Mid K', 'Early K']
SEE_MIN = 3                     # star-only 24 um flux >= 3x the error = "Spitzer can see it"
CAL_SN = 5                      # stricter version for the calibration stars (see above)
KS_ZP, M24_ZP = 666.7, 7.17     # Jy
h, c, k_B = 6.626e-34, 2.998e8, 1.381e-23
LAM = {'J': 1.235, 'H': 1.662, 'K': 2.159, 'W1': 3.353, 'I1': 3.550, 'I2': 4.493, 'W2': 4.603,
       'I3': 5.731, 'I4': 7.872, 'W3': 11.56, 'W4': 22.08, 'M1': 23.68}
BANDS = {'I2': 'i2_f_ap2', 'I3': 'i3_f_ap2', 'I4': 'i4_f_ap2', 'M1': 'm1_f_psf'}

# Oct 9 numbers (checks doc / sed_fit_j2133.py), for the old vs new table
OLD = {'plain stars': 28, '24 um correction': 1.329, 'plain Ks-[24]': 0.263,
       'IRAC 3.6 prediction (uJy)': 50.5, 'WISE W1 prediction (uJy)': 55.2, 'Ks prediction (uJy)': 53.5,
       'excess ratio (adopted)': 10.0, '24 um excess (uJy)': 456, 'excess error (uJy)': 24, 'sigma': 19,
       'dust T (K)': 270, 'f = Ldust/Lstar': 5e-3, 'comparable stars': 52, 'clean stars': 144,
       'next biggest comparable': 2.4}


def bnu(lam_um, T):
    nu = c / (lam_um * 1e-6)
    return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k_B * T))


def mag(f_ujy, zp):
    return -2.5 * np.log10(f_ujy * 1e-6 / zp)


def as_bool(s):
    return s.astype(str).str.lower().isin(['true', '1', '1.0'])


# ---------------------------------------------------------------- load
df = pd.read_csv(args.data).copy()
df['clean'] = as_bool(df['clean'])
for b, col in BANDS.items():
    df['r_' + b] = df[col] / (df['i1_f_ap2'] * (LAM['I1'] / LAM[b])**2)   # measured / RJ from 3.6
df['rj24'] = df['i1_f_ap2'] * (LAM['I1'] / LAM['M1'])**2
df['Ks24'] = mag(df['k'], KS_ZP) - mag(df['m1_f_psf'], M24_ZP)
df['group'] = pd.cut(df['Teff'], [TEFF_MIN] + EDGES + [TEFF_MAX], right=False, labels=GROUPS)

row = df[df.objid == CANDIDATE]
if row.empty:
    raise SystemExit(f'{CANDIDATE} is not in {args.data}')
row = row.iloc[0]
clean = df[df.clean]
print(f'Sample file: {args.data}')
print(f'Stars: {len(df)} (Teff {df.Teff.min():.0f}-{df.Teff.max():.0f} K), clean: {len(clean)}')
print(f'J2133+83: clean = {row.clean}, Teff {row.Teff:.0f} K -> {row.group}, '
      f'{abs(row.Teff - 4700):.0f} K from the Mid/Early K boundary')

# ---------------------------------------------------------------- photometry (uJy)
J, eJ, H, eH, K, eK = row.j, row.dj, row.h, row.dh, row.k, row.dk
I1, eI1, I2, eI2 = row.i1_f_ap2, row.i1_df_ap2, row.i2_f_ap2, row.i2_df_ap2
I3, eI3, I4, eI4 = row.i3_f_ap2, row.i3_df_ap2, row.i4_f_ap2, row.i4_df_ap2
M1, eM1 = row.m1_f_psf, row.m1_df_psf
expected = {'K': 3901.0, 'I1': 1692.0, 'I4': 546.5, 'M1': 509.0}
for b, v in {'K': K, 'I1': I1, 'I4': I4, 'M1': M1}.items():
    if abs(v / expected[b] - 1) > 0.005:
        print(f'WARNING: {b} = {v} uJy here but {expected[b]} uJy in sed_fit_j2133.py')


def wise(m, zp):
    return zp * 10**(-m / 2.5) * 1e6


# AllWISE row at 0.3" (CDS portal), same as sed_fit_j2133.py. W4 has no error -> upper limit.
W1, W2, W3 = wise(12.936, 309.54), wise(12.935, 171.787), wise(11.83, 31.674)
eW1, eW2 = W1 * 0.023 / 1.0857, W2 * 0.024 / 1.0857
eW3 = (wise(11.83 - 0.225, 31.674) - wise(11.83 + 0.225, 31.674)) / 2
W4_ul = wise(9.4, 8.363)


# ---------------------------------------------------------------- calibration
def calibrate(sample, method, sn_min=CAL_SN, c24=1.329):
    """Median measured/RJ ratio of plain stars at 4.5, 5.8, 8, 24 um (+ robust scatter)."""
    s = sample[sample.objid != CANDIDATE]
    for _ in range(30):                                   # 'seen' depends on c24, so iterate
        seen = s[s.rj24 * c24 / s.m1_df_psf >= sn_min]
        if method == 'ks075':
            plain = seen[seen.Ks24 < 0.75]
        else:                                             # 'clip'
            plain = seen
            for _ in range(30):
                med = plain.r_M1.median()
                sig = 1.4826 * (plain.r_M1 - med).abs().median()
                keep = seen[(seen.r_M1 - med).abs() < 3 * sig]
                if len(keep) == len(plain):
                    break
                plain = keep
        new_c24 = plain.r_M1.median()
        if abs(new_c24 - c24) < 1e-4:
            break
        c24 = new_c24
    out = {'n_seen': len(seen), 'n': len(plain), 'Ks24': plain.Ks24.median(), 'plain': plain}
    for b in BANDS:
        v = plain['r_' + b]
        out[b] = v.median()
        out['sd_' + b] = 1.4826 * (v - v.median()).abs().median()
    return out


def corr_fn(cal):
    lam = np.array([LAM['I1'], LAM['I2'], LAM['I3'], LAM['I4'], LAM['M1']])
    val = np.array([1.0, cal['I2'], cal['I3'], cal['I4'], cal['M1']])
    frac = np.array([0.0] + [cal['sd_' + b] / cal[b] for b in ['I2', 'I3', 'I4', 'M1']])
    return (lambda l: np.interp(np.log(l), np.log(lam), val),
            lambda l: np.interp(np.log(l), np.log(lam), frac))


def predict(cal, ks24=None):
    """Star-only 24 um flux three ways (uJy)."""
    ks24 = cal['Ks24'] if ks24 is None else ks24
    a = I1 * (LAM['I1'] / LAM['M1'])**2 * cal['M1']
    b = W1 * (LAM['W1'] / LAM['I1'])**2 * (LAM['I1'] / LAM['M1'])**2 * cal['M1']
    k_ = M24_ZP * 10**(-(mag(K, KS_ZP) - ks24) / 2.5) * 1e6
    return np.array([a, b, k_])


ADOPTED = 'S/N >= 5, clipped, all K (ADOPTED)'
variants = {
    ADOPTED: calibrate(clean, 'clip'),
    'S/N >= 5, clipped, Mid K only': calibrate(clean[clean.group == 'Mid K'], 'clip'),
    'S/N >= 5, clipped, Mid + Early K': calibrate(clean[clean.group.isin(['Mid K', 'Early K'])], 'clip'),
    'S/N >= 3, clipped, all K': calibrate(clean, 'clip', sn_min=3),
    'S/N >= 3, Ks-[24] < 0.75 (temp_groups)': calibrate(clean, 'ks075', sn_min=3),
}
cal = variants[ADOPTED]

print('\n--- Plain K dwarf calibration (measured / Rayleigh-Jeans from 3.6 um) ---')
print(f"{'set':40s} {'seen':>5s} {'plain':>5s} {'4.5':>6s} {'5.8':>6s} {'8.0':>6s} {'24':>6s} {'sd24':>6s} {'Ks-[24]':>8s}")
for name, v in variants.items():
    print(f"{name:40s} {v['n_seen']:5d} {v['n']:5d} {v['I2']:6.3f} {v['I3']:6.3f} {v['I4']:6.3f} "
          f"{v['M1']:6.3f} {v['sd_M1']:6.3f} {v['Ks24']:8.3f}")
print(f"{'Oct 9 (typed into sed_fit_j2133.py)':40s} {'':>5s} {28:5d} {1.034:6.3f} {1.112:6.3f} {1.208:6.3f} "
      f"{1.329:6.3f} {0.225:6.3f} {0.263:8.3f}")

# Faint stars only got into SEIP if their 24 um flux fluctuated up, so the ratio should
# drift up at low signal to noise. If it does, the high S/N number is the honest one.
loose = variants['S/N >= 3, clipped, all K']
p = loose['plain'].assign(sn=lambda d: d.rj24 * loose['M1'] / d.m1_df_psf)
print('\nPlain-star 24 um ratio by how clearly Spitzer sees the star (flux-limit bias check;')
print('if faint stars come out brighter, they are biased and the S/N >= 5 number is the fair one):')
for lo, hi in [(3, 5), (5, 10), (10, np.inf)]:
    q = p[(p.sn >= lo) & (p.sn < hi)]
    if len(q):
        print(f'  star S/N {lo}-{hi}: {len(q):3d} stars, median ratio {q.r_M1.median():.3f}')

# ---------------------------------------------------------------- J2133 at 24 um
preds = predict(cal)
pred24 = preds.mean()
corr, frac = corr_fn(cal)
e_ex24 = np.sqrt(eM1**2 + (0.04 * M1)**2 + (frac(LAM['M1']) * pred24)**2)
ex24 = M1 - pred24
print('\n--- J2133+83 at 24 um (adopted calibration) ---')
for name, v in zip(['RJ from IRAC 3.6, calibrated', 'RJ from WISE W1, calibrated',
                    f"Ks-[24] = {cal['Ks24']:.2f} of plain stars"], preds):
    print(f'  {name:32s} {v:6.1f} uJy -> {M1 / v:4.1f}x')
print(f'  adopted {pred24:.1f} uJy -> {M1 / pred24:.1f}x  (range {M1 / preds.max():.1f}-{M1 / preds.min():.1f}x)')
print(f'  24 um excess {ex24:.0f} +- {e_ex24:.0f} uJy = {ex24 / e_ex24:.0f} sigma')

print('\n  How much the ratio moves with the calibration choice:')
for name, v in variants.items():
    pv = predict(v)
    print(f'    {name:40s} {M1 / pv.mean():5.1f}x  ({M1 / pv.max():.1f}-{M1 / pv.min():.1f}x)')
for kc in [0.0, 0.1]:
    pk = predict(cal, ks24=kc)[2]
    print(f"    {f'Ks-[24] = {kc:.1f} (photosphere near 0, from Ks)':40s} {M1 / pk:5.1f}x")

# ---------------------------------------------------------------- shorter wavelengths
scale = pred24 / preds[0]
meas = {'I2': (I2, eI2, 0.03), 'W2': (W2, eW2, 0.024), 'I3': (I3, eI3, 0.03),
        'I4': (I4, eI4, 0.03), 'W3': (W3, eW3, 0.024)}
print('\n--- Other bands (measured / star alone) ---')
ex = {}
for b, (f, e, calib) in meas.items():
    pr = I1 * (LAM['I1'] / LAM[b])**2 * corr(LAM[b]) * scale
    err = np.sqrt(e**2 + (calib * f)**2 + (max(frac(LAM[b]), 0.05) * pr)**2)
    ex[b] = (f - pr, err)
    print(f'  {b}: {f:7.1f} vs {pr:7.1f} uJy  ratio {f / pr:.2f}  excess {f - pr:6.0f} +- {err:4.0f} '
          f'({(f - pr) / err:4.1f} sigma)')
pr_w4 = I1 * (LAM['I1'] / LAM['W4'])**2 * corr(LAM['W4']) * scale
print(f'  W4: upper limit {W4_ul:.0f} uJy vs star {pr_w4:.0f} uJy')

# ---------------------------------------------------------------- dust blackbody
pts = [(LAM['I4'], *ex['I4']), (LAM['W3'], *ex['W3']), (LAM['M1'], ex24, e_ex24)]
y = np.array([q[1] for q in pts])
s = np.array([q[2] for q in pts])
Ts = np.arange(40, 800, 1.0)
chi2 = []
for T in Ts:
    model = np.array([bnu(q[0], T) for q in pts])
    amp = np.sum(model * y / s**2) / np.sum(model**2 / s**2)
    chi2.append(np.sum(((y - amp * model) / s)**2))
chi2 = np.array(chi2)
Tbest = Ts[np.argmin(chi2)]
ok = Ts[chi2 < chi2.min() + 1]
L_star = 0.24                                                # Lsun (Gaia, M_G 6.45, BC_G -0.16)
F_star = 2.518e-8 * 10**(-0.4 * (row.Gmag - row.AG - 0.16))  # W/m^2


def frac_lum(T):
    return ex24 * 1e-32 / bnu(LAM['M1'], T) * 5.670e-8 * T**4 / np.pi / F_star


def r_bb(T):
    return (278.3 / T)**2 * np.sqrt(L_star)


print('\n--- Dust blackbody fit to the 8, 12, 24 um excess ---')
print(f'  best T = {Tbest:.0f} K (1-sigma {ok.min():.0f}-{ok.max():.0f} K), chi2 = {chi2.min():.2f} for 1 dof')
for T in sorted({int(Tbest), int(ok.min()), int(ok.max()), 150, 200, 300}):
    print(f'  T = {T:4d} K: f = {frac_lum(T):.1e} ({100 * frac_lum(T):.2f}%), blackbody distance {r_bb(T):.2f} AU')

# ---------------------------------------------------------------- comparable stars + galaxy odds
seen = clean[clean.rj24 * cal['M1'] / clean.m1_df_psf >= SEE_MIN].copy()
seen['ratio'] = seen.m1_f_psf / (seen.rj24 * cal['M1'])
top = seen.sort_values('ratio', ascending=False)
others = top[top.objid != CANDIDATE]
print('\n--- J2133+83 among the comparable stars ---')
print(f'  comparable (clean, Spitzer sees the star alone at 24 um): {len(seen)} of {len(clean)} clean; '
      f'J2133+83 included: {CANDIDATE in set(seen.objid)}')
print('  biggest measured / calibrated ratios:')
for _, r in top.head(5).iterrows():
    print(f"    {r.objid.replace('SSTSL2 ', ''):22s} {r.Teff:5.0f} K {str(r.group):8s} {r.ratio:5.1f}x  chi_24 {r.chi_24:6.1f}")
nxt = others.ratio.iloc[0] if len(others) else np.nan

p_one = {r: 1600 * np.pi * (r / 3600)**2 for r in (1.0, 1.5, 2.0)}   # SWIRE: 1600 /deg^2 above 0.46 mJy
print('\n--- Background galaxy odds (SWIRE counts, 1600 per sq deg brighter than 0.46 mJy) ---')
print(f"  {'within':>8s} {'this star':>11s} {f'{len(seen)} comparable':>15s} {f'{len(clean)} clean':>11s}")
for r, p1 in p_one.items():
    print(f'  {r:6.1f}"  1 in {1 / p1:5.0f}   {1 - (1 - p1)**len(seen):12.1%}   {1 - (1 - p1)**len(clean):9.1%}')

# ---------------------------------------------------------------- old vs new
newv = {'plain stars': cal['n'], '24 um correction': cal['M1'], 'plain Ks-[24]': cal['Ks24'],
        'IRAC 3.6 prediction (uJy)': preds[0], 'WISE W1 prediction (uJy)': preds[1],
        'Ks prediction (uJy)': preds[2], 'excess ratio (adopted)': M1 / pred24,
        '24 um excess (uJy)': ex24, 'excess error (uJy)': e_ex24, 'sigma': ex24 / e_ex24,
        'dust T (K)': Tbest, 'f = Ldust/Lstar': frac_lum(Tbest), 'comparable stars': len(seen),
        'clean stars': len(clean), 'next biggest comparable': nxt}
print('\n--- Oct 9 checks doc vs this run ---')
for key, o in OLD.items():
    n = newv[key]
    fmt = '.1e' if key.startswith('f =') else ('.0f' if isinstance(o, int) else '.3g')
    print(f'  {key:28s} {o:{fmt}} -> {n:{fmt}}')
print(f"  galaxy odds, comparable      2-8% -> {1 - (1 - p_one[1.0])**len(seen):.0%}-{1 - (1 - p_one[2.0])**len(seen):.0%}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.5, 5))
lg = np.logspace(0, np.log10(40), 300)
ph = np.where(lg >= 3.55, I1 * (3.55 / np.maximum(lg, 3.55))**2 * corr(np.maximum(lg, 3.55)) * scale,
              I1 * scale * bnu(lg, row.Teff) / bnu(3.55, row.Teff))
dust = ex24 / bnu(LAM['M1'], Tbest) * bnu(lg, Tbest)
nufnu = lambda l, f: c / (l * 1e-6) * f * 1e-32
ax.plot(lg, nufnu(lg, ph), color='0.45', lw=1.6, label='Star alone (predicted)')
ax.plot(lg, nufnu(lg, dust), color='#c2410c', lw=1.2, ls='--', label=f'Dust blackbody, {Tbest:.0f} K')
ax.plot(lg, nufnu(lg, ph + dust), color='#1d4ed8', lw=1.2, label='Star + dust')
for lab, bands, fl, er, mk, col in [
        ('2MASS', ['J', 'H', 'K'], [J, H, K], [eJ, eH, eK], 'o', '0.2'),
        ('WISE', ['W1', 'W2', 'W3'], [W1, W2, W3], [eW1, eW2, eW3], 's', '#7c3aed'),
        ('Spitzer IRAC', ['I1', 'I2', 'I3', 'I4'], [I1, I2, I3, I4], [eI1, eI2, eI3, eI4], 'D', '#0f766e'),
        ('Spitzer MIPS 24', ['M1'], [M1], [eM1], '*', '#b91c1c')]:
    L = np.array([LAM[b] for b in bands])
    ax.errorbar(L, nufnu(L, np.array(fl)), yerr=nufnu(L, np.array(er)), fmt=mk, color=col,
                ms=9 if mk == '*' else 6, label=lab, capsize=2, lw=1)
ax.errorbar([LAM['W4']], [nufnu(LAM['W4'], W4_ul)], yerr=[nufnu(LAM['W4'], W4_ul) * 0.35], uplims=[True],
            fmt='v', color='#7c3aed', ms=6, label='WISE 22 um (upper limit)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 40)
ax.set_ylim(1e-17, 3e-14)
ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel(r'$\nu F_\nu$ (W m$^{-2}$)')
ax.set_title(f'J2133+83: about {M1 / pred24:.0f}x brighter than the star alone at 24 microns')
ax.legend(fontsize=8, loc='lower left', frameon=False)
ax.grid(alpha=0.2, which='both')
fig.tight_layout()
Path(args.fig).parent.mkdir(exist_ok=True)
fig.savefig(args.fig, dpi=160)
print(f'\nSaved {args.fig}')
