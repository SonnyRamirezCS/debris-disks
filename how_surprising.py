"""
How surprising is J2133+83? Expected number of J2133-like objects in our clean sample
from each explanation, using published rates.

Inputs (all approximate):
  - Bright warm dust (f ~ 0.5%, ~300 K) rate: ~1% of stars under 120 Myr, ~1e-4 of stars
    over 1 Gyr (Kennedy & Wyatt 2013, as used in our summary doc).
  - Fraction of field dwarfs younger than 120 Myr: ~1.2% (steady star formation over ~10 Gyr).
  - Background galaxies: 24 um integral counts from Papovich et al. (2004, Table 2).
    A galaxy at least as bright as J2133+83's excess (~0.46 mJy) inside radius r of the star.
"""
import numpy as np

# Papovich et al. 2004 Table 2 cumulative counts (log S in mJy, N(>S) per steradian)
logS = np.array([-1.1, -0.95, -0.8, -0.65, -0.5, -0.35, -0.2, -0.05])
NS = np.array([6.1e7, 3.9e7, 2.9e7, 1.8e7, 8.6e6, 4.4e6, 2.3e6, 1.2e6])
ARCSEC2_PER_SR = (180 / np.pi * 3600) ** 2


def n_per_arcsec2(S_mJy):
    return 10 ** np.interp(np.log10(S_mJy), logS, np.log10(NS)) / ARCSEC2_PER_SR


def p_blend(S_mJy, r_arcsec):
    return 1 - np.exp(-n_per_arcsec2(S_mJy) * np.pi * r_arcsec ** 2)


EXCESS_mJy = 0.509 - 0.046        # J2133+83: measured 24 um minus the star
RATE_OLD, RATE_YOUNG, F_YOUNG = 1e-4, 1e-2, 0.012
SAMPLES = {"171 clean K stars": 171, "202 comparable stars, all types (full SEIP run)": 202}

print(f"Galaxy density brighter than {EXCESS_mJy:.2f} mJy: {n_per_arcsec2(EXCESS_mJy)*3600:.2f} per arcmin^2")
for r in (0.5, 1.0, 2.0):
    print(f"  chance one sits within {r}\" of a given star: {p_blend(EXCESS_mJy, r)*100:.3f}%")

rows = []
for label, N in SAMPLES.items():
    old = N * (1 - F_YOUNG) * RATE_OLD
    young = N * F_YOUNG * RATE_YOUNG
    gal1 = N * p_blend(EXCESS_mJy, 1.0)
    tot = old + young + gal1
    print(f"\n{label} (N = {N}):")
    print(f"  expected real disk around an old star:   {old:.3f}")
    print(f"  expected real disk around a young star:  {young:.3f}")
    print(f"  expected galaxy hiding within 1\":        {gal1:.3f}")
    print(f"  chance of finding at least one J2133-like object at all: {(1-np.exp(-tot))*100:.1f}%")
    print(f"  given we found one: galaxy {gal1/tot*100:.0f}%, young-star disk {young/tot*100:.0f}%, old-star disk {old/tot*100:.0f}%")
    print(f"  if the star is confirmed old: galaxy {gal1/(gal1+old)*100:.0f}%, old-star disk {old/(gal1+old)*100:.0f}%")
    rows.append((label, old, young, gal1))
