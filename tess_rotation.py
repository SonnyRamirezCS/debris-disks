"""
tess_rotation.py: look for a starspot rotation period for J2133+83 in TESS full-frame images.

Why: young K dwarfs spin fast (days), old ones spin slowly (weeks). A clear, repeatable
period of about a week or less would point to a young star. A single TESS sector is only
~27 days long, so periods longer than ~13 days can't be measured here, and finding no
period does NOT prove the star is old.

Target: J2133+83 = SSTSL2 J213307.73+830501.2 = Gaia DR3 2300237806311269248 = TIC 264404592

Run on your laptop (needs internet for MAST):
    pip install lightkurve
    python tess_rotation.py                  # every sector TESS has
    python tess_rotation.py --max-sectors 4  # quick test first
    python tess_rotation.py --target J0641-51 --max-sectors 4   # second candidate

Outputs go to tess_rotation_out/: summary.csv plus one PNG per sector.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
import lightkurve as lk

# Gaia DR3 positions. J2133+83 is the default; pick another with --target
TARGETS = {
    "J2133+83": SkyCoord(323.2824367 * u.deg, 83.0837421 * u.deg),    # Gaia DR3 2300237806311269248
    "J0641-51": SkyCoord(100.3937213 * u.deg, -51.0497450 * u.deg),   # Gaia DR3 5502841708866016512
}
CUTOUT = 15            # cutout size in pixels (TESS pixels are 21 arcsec)
PMIN, PMAX = 0.3, 13.0 # period search range in days (one sector can't support longer)
N_PCA = 5              # background components removed with the regression corrector
OUTDIR = "tess_rotation_out"


def make_lightcurve(tpf):
    """Aperture photometry on a TESScut cutout, with scattered light removed by regressing
    against the pixels outside the aperture (the standard lightkurve recipe for FFI cutouts)."""
    good = np.isfinite(tpf.flux.value).all(axis=(1, 2)) & (tpf.quality == 0)
    tpf = tpf[good]
    aper = tpf.create_threshold_mask(threshold=3, reference_pixel="center")
    if aper.sum() == 0:
        aper = tpf.create_threshold_mask(threshold=1.5, reference_pixel="center")
    if aper.sum() == 0:
        raise ValueError("no aperture found around the target")
    raw = tpf.to_lightcurve(aperture_mask=aper)
    regressors = tpf.flux.value[:, ~aper]
    dm = lk.DesignMatrix(regressors, name="regressors").pca(N_PCA).append_constant()
    corrected = lk.RegressionCorrector(raw).correct(dm)
    lc = corrected.remove_nans().normalize().remove_outliers(sigma=5)
    return tpf, aper, lc


def find_period(time, flux, pmin=PMIN, pmax=PMAX):
    """Lomb-Scargle on a normalized light curve. Returns the periodogram and the best peak."""
    ls = LombScargle(time, flux)
    freq, power = ls.autopower(minimum_frequency=1 / pmax, maximum_frequency=1 / pmin,
                               samples_per_peak=10)
    i = int(np.argmax(power))
    best_f = freq[i]
    fap = float(ls.false_alarm_probability(power[i], minimum_frequency=1 / pmax,
                                           maximum_frequency=1 / pmin))
    tfit = np.linspace(0, 1 / best_f, 200)
    model = ls.model(tfit, best_f)
    semi_amp = (model.max() - model.min()) / 2
    return freq, power, 1 / best_f, fap, semi_amp


def plot_sector(sector, tpf, aper, lc, freq, power, best_p, fap, semi_amp, path):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    tpf.plot(ax=ax1, aperture_mask=aper, show_colorbar=False)
    ax1.set_title(f"Sector {sector} cutout, aperture in red")

    ax2 = fig.add_subplot(2, 3, (2, 3))
    t, f = lc.time.value, lc.flux.value
    ax2.plot(t, f, ".", ms=2, color="0.6")
    binned = lc.bin(time_bin_size=0.25)
    ax2.plot(binned.time.value, binned.flux.value, ".", ms=5, color="C0")
    ax2.set_xlabel("Time [BTJD]")
    ax2.set_ylabel("Normalized flux")
    ax2.set_title("Corrected light curve (blue = 6 h bins)")

    ax3 = fig.add_subplot(2, 3, (4, 5))
    ax3.plot(1 / freq, power, color="C0")
    ax3.axvline(best_p, color="C3", ls="--")
    ax3.set_xscale("log")
    ax3.set_xlabel("Period [days]")
    ax3.set_ylabel("Lomb-Scargle power")
    ax3.set_title(f"Best period {best_p:.2f} d, FAP {fap:.1e}, semi-amplitude {1e3 * semi_amp:.1f} ppt")

    ax4 = fig.add_subplot(2, 3, 6)
    phase = ((t - t[0]) / best_p) % 1
    ax4.plot(phase, f, ".", ms=2, color="0.6")
    pb = np.linspace(0, 1, 21)
    idx = np.digitize(phase, pb) - 1
    med = [np.nanmedian(f[idx == k]) if np.any(idx == k) else np.nan for k in range(20)]
    ax4.plot((pb[:-1] + pb[1:]) / 2, med, "o-", color="C3")
    ax4.set_xlabel("Phase")
    ax4.set_title(f"Folded at {best_p:.2f} d")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sectors", type=int, default=None)
    ap.add_argument("--target", default="J2133+83", choices=sorted(TARGETS))
    args = ap.parse_args()
    global OUTDIR
    if args.target != "J2133+83":
        OUTDIR = f"tess_rotation_out_{args.target}"
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Target: {args.target}, results go to {OUTDIR}/")

    search = lk.search_tesscut(TARGETS[args.target])
    print(f"TESS has {len(search)} sectors of full-frame images for this star")
    if len(search) == 0:
        return
    n = len(search) if args.max_sectors is None else min(len(search), args.max_sectors)

    rows = []
    for k in range(n):
        cols = search.table.colnames
        if "sequence_number" in cols:
            sector = int(search.table["sequence_number"][k])
        elif "mission" in cols:
            sector = str(search.table["mission"][k]).replace("TESS Sector", "").strip()
        else:
            sector = k
        try:
            tpf = search[k].download(cutout_size=CUTOUT)
            tpf, aper, lc = make_lightcurve(tpf)
            freq, power, best_p, fap, semi_amp = find_period(lc.time.value, lc.flux.value)
            rms = np.nanstd(lc.flux.value)
            plot_sector(sector, tpf, aper, lc, freq, power, best_p, fap, semi_amp,
                        os.path.join(OUTDIR, f"sector_{sector}.png"))
            rows.append((sector, len(lc), best_p, fap, 1e3 * semi_amp, 1e3 * rms, int(aper.sum())))
            print(f"sector {sector}: best period {best_p:.2f} d, FAP {fap:.1e}, "
                  f"semi-amplitude {1e3 * semi_amp:.1f} ppt, scatter {1e3 * rms:.1f} ppt")
        except Exception as e:
            print(f"sector {sector}: skipped ({e})")

    with open(os.path.join(OUTDIR, "summary.csv"), "w") as fh:
        fh.write("sector,n_points,best_period_d,fap,semi_amp_ppt,scatter_ppt,aperture_pixels\n")
        for r in rows:
            fh.write("{},{},{:.3f},{:.2e},{:.2f},{:.2f},{}\n".format(*r))

    print("\nHow to read this:")
    print("  A real rotation period should show up at about the same value (within ~10%)")
    print("  in at least 2 sectors, each with a low FAP. The FAP here is optimistic because")
    print("  TESS noise is correlated, so don't trust a period seen in only one sector.")
    print("  Peaks near 13 days or longer are usually leftover scattered light, not the star.")
    print("  If you do find a period, run tpf.plot_pixels(periodogram=True) on that sector to")
    print("  check the signal is centered on the star and not on a neighbor in the cutout.")


if __name__ == "__main__":
    main()
