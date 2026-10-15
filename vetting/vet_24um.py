#!/usr/bin/env python3
"""
vet_24um.py

Quick check of whether a Spitzer 24 micron excess really belongs to the star,
using the PBCD mosaics (maic.fits) from the Spitzer Heritage Archive.

What it does:
  1. Makes a 2x3 figure: IRAC 3.6, IRAC 8.0 and MIPS 24, once wide and once
     zoomed in, with the star marked. Saved as <name>_cutouts.png.
  2. Fits a 2D Gaussian to the 24 micron source near the star to get how far
     its center is from the star and how wide it is (FWHM).
  3. Does aperture photometry on the 24 micron mosaic (MJy/sr -> microJy),
     applies an aperture correction, and compares it to the catalog flux.
  4. Lists other sources close to the star in each band (possible blends).
  5. Pointing check: matches other 24 micron sources in the field to their
     IRAC counterparts to see if the whole MIPS image is shifted. If it is,
     a small 24 micron offset on the star is just that shift, not a problem.
  6. Prints a short report and saves it as <name>_24um_vetting.txt.

Defaults are set for J0845-78 (SSTSL2 J084517.41-785937.1), so if the three
files are in the folder you run it from, this is all you need:

    python vet_24um.py

To use IRAC 4.5 instead of 3.6 (labels are read from the file header):

    python vet_24um.py --irac1 SPITZER_I2_3958016_0000_6_E8758331_maic.fits

For another star or other files:

    python vet_24um.py --mips M1.fits --irac1 I1.fits --irac4 I4.fits \
        --ra 131.32254 --dec -78.99364 --name J0845-78 \
        --ks 12.633 --cat-flux 510.40 --cat-err 15.561

Needs: numpy, astropy, matplotlib, scipy
    pip install numpy astropy matplotlib scipy

Notes:
  * Default aperture is 7 arcsec with a 20-32 arcsec sky annulus and an
    aperture correction of 1.61. Double check that value against the 24 micron
    aperture correction table in the MIPS Instrument Handbook, and change
    --apcorr if you change --ap / --ann-in / --ann-out.
  * Mosaic pixels are correlated, so the error printed here is a lower limit.
  * The flags are hints for where to look, not final answers. Always look at
    the PNG.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import binary_dilation, maximum_filter

MIPS24_FWHM = 5.9          # arcsec, MIPS 24 micron PSF FWHM
ARCSEC_PER_RAD = 206264.806


# ----------------------------------------------------------------------------
# Loading and basic helpers
# ----------------------------------------------------------------------------

def load_image(path):
    """Return (data, wcs, header) for the first 2D image HDU in a FITS file."""
    if not os.path.exists(path):
        sys.exit(f"Can't find file: {path}")
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                data = np.array(hdu.data, dtype=float)
                header = hdu.header.copy()
                break
        else:
            sys.exit(f"No 2D image found in {path}")
    wcs = WCS(header, naxis=2)
    return data, wcs, header


def pixscale(wcs):
    """Mean pixel scale in arcsec."""
    return float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)


def check_units(header, label, warnings):
    bunit = str(header.get("BUNIT", "")).replace(" ", "").lower()
    if bunit != "mjy/sr":
        warnings.append(f"{label}: BUNIT is '{header.get('BUNIT', '')}', "
                        f"expected MJy/sr. Flux conversion may be wrong.")


def star_on_image(data, wcs, coord):
    """Is the star inside the image and on real (non-NaN) data?"""
    x, y = wcs.world_to_pixel(coord)
    x, y = float(x), float(y)
    ny, nx = data.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < nx and 0 <= iy < ny):
        return False, x, y
    return bool(np.isfinite(data[iy, ix])), x, y


# ----------------------------------------------------------------------------
# Photometry
# ----------------------------------------------------------------------------

def aperture_sum(data, x0, y0, r_pix, sub=5):
    """Sum inside a circle, with sub-pixel sampling for partial pixels.

    Returns (sum, area_used_pix, n_missing_pix).
    """
    ny, nx = data.shape
    x_lo = max(int(np.floor(x0 - r_pix - 1)), 0)
    x_hi = min(int(np.ceil(x0 + r_pix + 1)), nx - 1)
    y_lo = max(int(np.floor(y0 - r_pix - 1)), 0)
    y_hi = min(int(np.ceil(y0 + r_pix + 1)), ny - 1)
    ys, xs = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]

    offs = (np.arange(sub) + 0.5) / sub - 0.5
    dx = xs[..., None, None] + offs[None, None, None, :] - x0
    dy = ys[..., None, None] + offs[None, None, :, None] - y0
    frac = ((dx ** 2 + dy ** 2) <= r_pix ** 2).mean(axis=(2, 3))

    vals = data[y_lo:y_hi + 1, x_lo:x_hi + 1]
    good = np.isfinite(vals)
    n_missing = int(np.sum((frac > 0) & ~good))
    total = float(np.sum(frac[good] * vals[good]))
    area = float(np.sum(frac[good]))
    return total, area, n_missing


def annulus_stats(data, x0, y0, r_in, r_out):
    """Sigma-clipped sky level and scatter in an annulus (pixel units)."""
    ny, nx = data.shape
    x_lo = max(int(np.floor(x0 - r_out - 1)), 0)
    x_hi = min(int(np.ceil(x0 + r_out + 1)), nx - 1)
    y_lo = max(int(np.floor(y0 - r_out - 1)), 0)
    y_hi = min(int(np.ceil(y0 + r_out + 1)), ny - 1)
    ys, xs = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
    d = np.hypot(xs - x0, ys - y0)
    vals = data[y_lo:y_hi + 1, x_lo:x_hi + 1]
    mask = (d >= r_in) & (d <= r_out) & np.isfinite(vals)
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    _, med, std = sigma_clipped_stats(vals[mask], sigma=3.0)
    return float(med), float(std), int(mask.sum())


def photometry(data, wcs, coord, ap, ann_in, ann_out, apcorr):
    """Forced aperture photometry at coord. Returns dict in microJy."""
    ps = pixscale(wcs)
    x, y = wcs.world_to_pixel(coord)
    x, y = float(x), float(y)
    r, r1, r2 = ap / ps, ann_in / ps, ann_out / ps

    sky, sky_std, n_ann = annulus_stats(data, x, y, r1, r2)
    raw, area, n_missing = aperture_sum(data, x, y, r)
    net = raw - sky * area                         # MJy/sr * pixels

    pix_sr = (ps / ARCSEC_PER_RAD) ** 2
    to_ujy = pix_sr * 1e12                         # (MJy/sr * pix) -> microJy

    err = sky_std * np.sqrt(area + area ** 2 / max(n_ann, 1))
    return {
        "x": x, "y": y, "pixscale": ps,
        "sky": sky, "sky_std": sky_std, "n_ann": n_ann,
        "area_pix": area, "n_missing": n_missing,
        "flux_raw_ujy": net * to_ujy,
        "flux_ujy": net * to_ujy * apcorr,
        "err_ujy": err * to_ujy * apcorr,
    }


def fit_source(data, wcs, coord, sky, sky_std, fit_radius=8.0,
               fwhm0=MIPS24_FWHM, verbose=True):
    """Fit a 2D Gaussian (plus constant) to the source near coord."""
    ps = pixscale(wcs)
    x0, y0 = wcs.world_to_pixel(coord)
    x0, y0 = float(x0), float(y0)
    half = int(np.ceil(fit_radius / ps))
    ix, iy = int(round(x0)), int(round(y0))
    ny, nx = data.shape
    x_lo, x_hi = max(ix - half, 0), min(ix + half, nx - 1)
    y_lo, y_hi = max(iy - half, 0), min(iy + half, ny - 1)

    ys, xs = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
    z = data[y_lo:y_hi + 1, x_lo:x_hi + 1] - (sky if np.isfinite(sky) else 0.0)
    good = np.isfinite(z)
    if good.sum() < 9:
        return None
    xs, ys, z = xs[good], ys[good], z[good]

    near = np.hypot(xs - x0, ys - y0) <= max(2.0, 3.0 / ps)
    amp0 = float(np.max(z[near])) if near.any() else float(np.max(z))
    sig0 = (fwhm0 / 2.3548) / ps

    g = models.Gaussian2D(amplitude=amp0, x_mean=x0, y_mean=y0,
                          x_stddev=sig0, y_stddev=sig0, theta=0.0)
    g.x_mean.bounds = (x0 - half, x0 + half)
    g.y_mean.bounds = (y0 - half, y0 + half)
    g.x_stddev.bounds = (0.3 * sig0, 5 * sig0)
    g.y_stddev.bounds = (0.3 * sig0, 5 * sig0)
    model = g + models.Const2D(amplitude=0.0)

    fitter = fitting.TRFLSQFitter() if hasattr(fitting, "TRFLSQFitter") \
        else fitting.LevMarLSQFitter()
    try:
        fit = fitter(model, xs, ys, z)
    except Exception as exc:                        # noqa: BLE001
        if verbose:
            print(f"  Gaussian fit failed: {exc}")
        return None

    gx, gy = float(fit[0].x_mean.value), float(fit[0].y_mean.value)
    sx, sy = abs(float(fit[0].x_stddev.value)), abs(float(fit[0].y_stddev.value))
    center = wcs.pixel_to_world(gx, gy)
    return {
        "x": gx, "y": gy, "center": center,
        "offset": float(coord.separation(center).arcsec),
        "pa": float(coord.position_angle(center).deg),
        "fwhm_major": 2.3548 * max(sx, sy) * ps,
        "fwhm_minor": 2.3548 * min(sx, sy) * ps,
        "peak_snr": float(fit[0].amplitude.value) / sky_std
        if np.isfinite(sky_std) and sky_std > 0 else np.nan,
    }


# ----------------------------------------------------------------------------
# Neighbor search
# ----------------------------------------------------------------------------

def find_neighbors(data, wcs, coord, radius, nsig=5.0, box=60.0):
    """Local maxima above nsig*sigma within `radius` arcsec of the star.

    Peaks right next to a data gap (NaN pixels) are skipped, because the
    edge of a gap that cuts through a source makes fake "peaks". Peaks that
    are really the same source (flat tops) are merged.

    Returns (list of (sep_arcsec, pa_deg, snr) closest first,
             n_skipped_near_gap).
    """
    try:
        cut = Cutout2D(data, coord, box * u.arcsec, wcs=wcs,
                       mode="partial", fill_value=np.nan)
    except NoOverlapError:
        return [], 0
    img = cut.data
    finite = np.isfinite(img)
    if finite.sum() < 50:
        return [], 0
    ps = pixscale(cut.wcs)
    _, med, std = sigma_clipped_stats(img[finite], sigma=3.0)
    filled = np.where(finite, img, med)
    size = 3 if ps > 1.5 else 5
    peaks = (filled == maximum_filter(filled, size=size)) \
        & (filled > med + nsig * std) & finite

    # pixels within ~2" (IRAC) or ~1 pixel+ (MIPS) of a gap are not trusted
    gap_pix = max(2, int(np.ceil((2.0 if ps < 1.5 else 3.0) / ps)))
    near_gap = binary_dilation(~finite, iterations=gap_pix)

    merge_r = 1.5 if ps < 1.5 else 4.0          # arcsec
    cand, skipped = [], 0
    for y, x in zip(*np.nonzero(peaks)):
        sky = cut.wcs.pixel_to_world(x, y)
        sep = float(coord.separation(sky).arcsec)
        if sep > radius:
            continue
        if near_gap[y, x]:
            skipped += 1
            continue
        cand.append((float((filled[y, x] - med) / std), sky, sep))

    out, kept = [], []
    for snr, sky, sep in sorted(cand, key=lambda c: -c[0]):
        if any(sky.separation(k).arcsec < merge_r for k in kept):
            continue
        kept.append(sky)
        out.append((sep, float(coord.position_angle(sky).deg), snr))
    return sorted(out), skipped


def band_label(header, default):
    """Name the band from the header (IRAC CHNLNUM), else use default."""
    inst = str(header.get("INSTRUME", "")).upper()
    ch = header.get("CHNLNUM")
    if "IRAC" in inst and ch in (1, 2, 3, 4):
        return {1: "IRAC 3.6", 2: "IRAC 4.5", 3: "IRAC 5.8", 4: "IRAC 8.0"}[ch]
    return default


# ----------------------------------------------------------------------------
# MIPS pointing check
# ----------------------------------------------------------------------------

IRAC_FWHM = 1.8   # arcsec, rough IRAC PSF FWHM
SMALL_OFFSET = 1.0  # arcsec; leftover 24um offsets below this are "probably fine"


def offset_verdict(pc):
    """'ok', 'small', 'big', or None (pointing check couldn't measure it)."""
    if pc is None or "chi" not in pc:
        return None
    if pc["chi"] < 2.5:
        return "ok"
    if pc["residual_total"] < SMALL_OFFSET:
        return "small"
    return "big"


def _peaks(img, nsig, size, gap_pix, edge_pix=0, med=None, std=None):
    """Pixel (x, y) of local maxima above med + nsig*std, away from gaps."""
    finite = np.isfinite(img)
    if med is None or std is None:
        _, med, std = sigma_clipped_stats(img[finite], sigma=3.0)
    filled = np.where(finite, img, med)
    pk = (filled == maximum_filter(filled, size=size)) \
        & (filled > med + nsig * std) & finite
    pk &= ~binary_dilation(~finite, iterations=gap_pix)
    if edge_pix:
        pk[:edge_pix, :] = pk[-edge_pix:, :] = False
        pk[:, :edge_pix] = pk[:, -edge_pix:] = False
    ys, xs = np.nonzero(pk)
    return list(zip(xs, ys)), med, std


def _irac_fit(irac, wirac, coord):
    """Gaussian center of an IRAC source near coord (or None)."""
    ps = pixscale(wirac)
    x, y = wirac.world_to_pixel(coord)
    sky, sstd, _ = annulus_stats(irac, float(x), float(y), 6.0 / ps, 10.0 / ps)
    f = fit_source(irac, wirac, coord, sky, sstd, fit_radius=3.0,
                   fwhm0=IRAC_FWHM, verbose=False)
    if f is None or f["offset"] > 1.5 or not 0.8 < f["fwhm_major"] < 4.5:
        return None
    return f["center"]


def pointing_check(m24, w24, irac, wirac, coord, target_24_center,
                   exclude=10.0, match_r=4.0, nsig=5.0):
    """Is the whole MIPS image shifted relative to IRAC?

    Finds other 24 um sources in the MIPS mosaic, finds each one's single
    IRAC counterpart, and measures the 24um-minus-IRAC position shift. The
    median shift is the overall MIPS pointing error. Then it compares the
    target's own 24um-minus-IRAC shift to that.
    """
    ps24 = pixscale(w24)
    psi = pixscale(wirac)
    res = {"n_det": 0, "n_used": 0, "n_nomatch": 0, "n_ambig": 0,
           "n_badfit": 0, "offsets": []}

    # IRAC noise level from a big patch around the star
    try:
        big = Cutout2D(irac, coord, 300 * u.arcsec, wcs=wirac,
                       mode="partial", fill_value=np.nan)
    except NoOverlapError:
        return None
    fin = np.isfinite(big.data)
    if fin.sum() < 1000:
        return None
    _, imed, istd = sigma_clipped_stats(big.data[fin], sigma=3.0)

    pk24, _, _ = _peaks(m24, nsig, size=3, gap_pix=3, edge_pix=5)
    for x, y in pk24:
        s = w24.pixel_to_world(x, y)
        if coord.separation(s).arcsec < exclude:
            continue                                 # skip the target
        res["n_det"] += 1
        sky, sstd, _ = annulus_stats(m24, float(x), float(y),
                                     15.0 / ps24, 25.0 / ps24)
        f = fit_source(m24, w24, s, sky, sstd, verbose=False)
        if (f is None or f["offset"] > 3.0 or f["peak_snr"] < nsig
                or not 3.5 < f["fwhm_major"] < 9.0):
            res["n_badfit"] += 1
            continue
        mc = f["center"]

        # IRAC peaks near the 24 um position
        try:
            cut = Cutout2D(irac, mc, (2 * match_r + 6) * u.arcsec, wcs=wirac,
                           mode="partial", fill_value=np.nan)
        except NoOverlapError:
            res["n_nomatch"] += 1
            continue
        if np.isfinite(cut.data).sum() < 20:
            res["n_nomatch"] += 1
            continue
        pki, _, _ = _peaks(cut.data, nsig, size=5,
                           gap_pix=max(2, int(np.ceil(2.0 / psi))),
                           med=imed, std=istd)
        cands = []
        for px, py in pki:
            c = cut.wcs.pixel_to_world(px, py)
            if mc.separation(c).arcsec <= match_r and \
                    all(c.separation(k).arcsec > 1.5 for k in cands):
                cands.append(c)
        if not cands:
            res["n_nomatch"] += 1
            continue
        if len(cands) > 1:
            res["n_ambig"] += 1
            continue
        ic = _irac_fit(irac, wirac, cands[0])
        if ic is None:
            res["n_badfit"] += 1
            continue
        de, dn = ic.spherical_offsets_to(mc)
        res["offsets"].append((de.to_value(u.arcsec), dn.to_value(u.arcsec)))

    res["n_used"] = len(res["offsets"])

    # target: 24 um center minus the star's own IRAC center
    star_ic = _irac_fit(irac, wirac, coord)
    res["star_irac_ok"] = star_ic is not None
    ref = star_ic if star_ic is not None else coord
    te, tn = ref.spherical_offsets_to(target_24_center)
    res["target"] = (te.to_value(u.arcsec), tn.to_value(u.arcsec))

    if res["n_used"] >= 3:
        arr = np.array(res["offsets"])
        med = np.median(arr, axis=0)
        mad = 1.4826 * np.median(np.abs(arr - med), axis=0)
        mad = np.maximum(mad, 0.1)                   # floor, avoid /0
        res["median"] = (float(med[0]), float(med[1]))
        res["scatter"] = (float(mad[0]), float(mad[1]))
        re_, rn = res["target"][0] - med[0], res["target"][1] - med[1]
        res["residual"] = (float(re_), float(rn))
        res["residual_total"] = float(np.hypot(re_, rn))
        res["chi"] = float(np.hypot(re_ / mad[0], rn / mad[1]))
    return res


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def plot_panels(images, coord, name, wide, zoom, ap, ann_in, ann_out,
                fit24, outpath):
    fig = plt.figure(figsize=(14, 9.6))
    for row, size in enumerate((wide, zoom)):
        for col, (label, data, wcs) in enumerate(images):
            try:
                cut = Cutout2D(data, coord, size * u.arcsec, wcs=wcs,
                               mode="partial", fill_value=np.nan)
            except NoOverlapError:
                cut = None
            if cut is None:
                ax = fig.add_subplot(2, 3, row * 3 + col + 1)
                vals = np.array([])
            else:
                ax = fig.add_subplot(2, 3, row * 3 + col + 1,
                                     projection=cut.wcs)
                vals = cut.data[np.isfinite(cut.data)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "no data here", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{label} ({size:.0f}\")")
                continue
            vmin, vmax = ZScaleInterval().get_limits(vals)
            if row == 0:
                # wide view: plain zscale, best for spotting faint stuff
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
            else:
                # zoom view: asinh up to the brightest pixel, so a close
                # neighbor doesn't merge into one black blob with the star
                vmax = float(np.max(vals))
                norm = ImageNormalize(vmin=vmin,
                                      vmax=max(vmax, vmin + 1e-6),
                                      stretch=AsinhStretch(0.05))
            ax.imshow(cut.data, origin="lower", cmap="gray_r", norm=norm,
                      interpolation="nearest")
            ny, nx = cut.data.shape

            ps = pixscale(cut.wcs)
            xs, ys = cut.wcs.world_to_pixel(coord)
            ax.plot(xs, ys, "+", color="red", ms=16, mew=1.5)

            if label.startswith("MIPS"):
                ax.add_patch(Circle((xs, ys), ap / ps, fill=False,
                                    ec="red", lw=1.2))
                for r in (ann_in, ann_out):
                    ax.add_patch(Circle((xs, ys), r / ps, fill=False,
                                        ec="orange", lw=1, ls="--"))
                if fit24 is not None:
                    fx, fy = cut.wcs.world_to_pixel(
                        wcs.pixel_to_world(fit24["x"], fit24["y"]))
                    ax.plot(fx, fy, "x", color="blue", ms=10, mew=1.5)
            else:
                # 6 arcsec circle = roughly where a source would blend into
                # the 24 micron beam
                ax.add_patch(Circle((xs, ys), 6.0 / ps, fill=False,
                                    ec="deepskyblue", lw=1.2, ls="--"))

            # keep the view on the cutout even if circles stick out of it
            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(-0.5, ny - 0.5)
            ax.set_title(f"{label} ({size:.0f}\" box)")
            ax.coords[0].set_axislabel("RA")
            ax.coords[1].set_axislabel("Dec")
            for c in ax.coords:
                try:
                    c.set_ticklabel(exclude_overlapping=True, size=7)
                except TypeError:
                    pass

    fig.suptitle(
        f"{name}   RA {coord.ra.deg:.5f}  Dec {coord.dec.deg:.5f}\n"
        "red + = star | red circle = 24um aperture | orange = sky annulus | "
        "blue x = fitted 24um center | cyan dashed = 6\" radius",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mips", default="SPITZER_M1_3987712_0003_9_E6473816_maic.fits")
    p.add_argument("--irac1", default="SPITZER_I1_3958016_0000_6_E8757921_maic.fits",
                   help="IRAC 3.6 (or 4.5) mosaic")
    p.add_argument("--irac4", default="SPITZER_I4_3958016_0000_6_E8758033_maic.fits",
                   help="IRAC 8.0 mosaic")
    p.add_argument("--ra", type=float, default=131.32254, help="deg, J2000")
    p.add_argument("--dec", type=float, default=-78.99364, help="deg, J2000")
    p.add_argument("--name", default="J0845-78")
    p.add_argument("--wide", type=float, default=90.0, help="wide box, arcsec")
    p.add_argument("--zoom", type=float, default=30.0, help="zoom box, arcsec")
    p.add_argument("--ap", type=float, default=7.0, help="aperture radius, arcsec")
    p.add_argument("--ann-in", type=float, default=20.0, help="arcsec")
    p.add_argument("--ann-out", type=float, default=32.0, help="arcsec")
    p.add_argument("--apcorr", type=float, default=1.61,
                   help="24um aperture correction for the chosen ap/annulus")
    p.add_argument("--zp", type=float, default=7.17,
                   help="MIPS 24 zero point in Jy")
    p.add_argument("--ks", type=float, default=12.633,
                   help="2MASS Ks mag (use 0 to skip the Ks - [24] lines)")
    p.add_argument("--cat-flux", type=float, default=510.40,
                   help="catalog 24um flux in microJy (e.g. SSTSL2 F24-PSF)")
    p.add_argument("--cat-err", type=float, default=15.561, help="microJy")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    coord = SkyCoord(a.ra * u.deg, a.dec * u.deg, frame="icrs")
    warnings, flags, lines = [], [], []

    def say(s=""):
        print(s)
        lines.append(s)

    print("Loading mosaics (the IRAC ones are big, give it a few seconds)...")
    m24, w24, h24 = load_image(a.mips)
    i1, w1, h1 = load_image(a.irac1)
    i4, w4, h4 = load_image(a.irac4)
    lab1 = band_label(h1, "IRAC 3.6")
    lab4 = band_label(h4, "IRAC 8.0")
    for lab, h in (("MIPS 24", h24), (lab1, h1), (lab4, h4)):
        check_units(h, lab, warnings)

    say(f"=== {a.name}  RA {a.ra}  Dec {a.dec} ===")
    say(f"MIPS 24 : {os.path.basename(a.mips)}")
    say(f"{lab1}: {os.path.basename(a.irac1)}")
    say(f"{lab4}: {os.path.basename(a.irac4)}")
    say()

    # --- coverage -----------------------------------------------------------
    say("Coverage (is the star on real data?)")
    covered = {}
    for lab, d, w in (("MIPS 24", m24, w24), (lab1, i1, w1), (lab4, i4, w4)):
        ok, x, y = star_on_image(d, w, coord)
        covered[lab] = ok
        say(f"  {lab:8s}: {'yes' if ok else 'NO'}  (pixel x={x:.1f}, y={y:.1f}, "
            f"scale {pixscale(w):.3f}\"/pix)")
        if not ok:
            flags.append(f"Star is not on valid data in {lab} (off the image "
                         "or in a data gap), so that band can't be used to "
                         "check for neighbors.")
    say()

    if not covered["MIPS 24"]:
        say("The star is not on the MIPS 24 mosaic, so there's nothing to "
            "measure. Check the file and the coordinates.")
        txt = os.path.join(a.outdir, f"{a.name}_24um_vetting.txt")
        with open(txt, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Report saved: {txt}")
        sys.exit(1)

    # --- 24 um photometry ---------------------------------------------------
    ph = photometry(m24, w24, coord, a.ap, a.ann_in, a.ann_out, a.apcorr)
    say("MIPS 24 aperture photometry (forced at the star position)")
    say(f"  aperture {a.ap}\", sky annulus {a.ann_in}-{a.ann_out}\", "
        f"aperture correction {a.apcorr}")
    say(f"  sky = {ph['sky']:.5f} MJy/sr, sky scatter = {ph['sky_std']:.5f} "
        f"MJy/sr ({ph['n_ann']} pix)")
    say(f"  flux before correction = {ph['flux_raw_ujy']:.1f} uJy")
    say(f"  flux after correction  = {ph['flux_ujy']:.1f} +/- "
        f"{ph['err_ujy']:.1f} uJy  (error is a lower limit)")
    if ph["n_missing"]:
        flags.append(f"{ph['n_missing']} pixel(s) inside the 24um aperture "
                     "have no data (edge or gap).")

    if a.cat_flux > 0 and ph["flux_ujy"] > 0:
        ratio = ph["flux_ujy"] / a.cat_flux
        say(f"  catalog flux = {a.cat_flux:.2f} +/- {a.cat_err:.2f} uJy  ->  "
            f"yours / catalog = {ratio:.2f}")
        if not 0.8 <= ratio <= 1.2:
            flags.append(f"Your 24um flux differs from the catalog by "
                         f"{(ratio - 1) * 100:+.0f}%. Check aperture, annulus, "
                         "correction, and nearby sources.")
    if ph["flux_ujy"] > 0:
        m = -2.5 * np.log10(ph["flux_ujy"] * 1e-6 / a.zp)
        say(f"  [24] = {m:.3f} mag (zero point {a.zp} Jy)")
        if a.ks:
            say(f"  Ks - [24] = {a.ks - m:.3f} mag  "
                "(a bare K dwarf photosphere is close to 0)")
    if a.cat_flux > 0:
        mc = -2.5 * np.log10(a.cat_flux * 1e-6 / a.zp)
        ks_part = f", Ks - [24] = {a.ks - mc:.3f}" if a.ks else ""
        say(f"  catalog: [24] = {mc:.3f}{ks_part}")
    say()

    # --- 24 um source fit ---------------------------------------------------
    fit24 = fit_source(m24, w24, coord, ph["sky"], ph["sky_std"])
    say("MIPS 24 source fit (2D Gaussian)")
    if fit24 is None:
        say("  fit failed, check the image by eye")
        flags.append("24um Gaussian fit failed.")
    else:
        say(f"  center offset from star = {fit24['offset']:.2f}\" "
            f"(PA {fit24['pa']:.0f} deg)")
        say(f"  FWHM = {fit24['fwhm_major']:.1f}\" x {fit24['fwhm_minor']:.1f}\" "
            f"(MIPS 24 point source is about {MIPS24_FWHM}\")")
        say(f"  peak S/N per pixel ~ {fit24['peak_snr']:.1f}")
        if fit24["fwhm_major"] > 1.3 * MIPS24_FWHM:
            flags.append(f"24um source looks wider than a point source "
                         f"({fit24['fwhm_major']:.1f}\"). Could be extended "
                         "(galaxy) or a blend.")
    say()

    # --- neighbors ----------------------------------------------------------
    say("Other sources near the star (local peaks above 5 sigma)")
    say("  sep = arcsec from star, PA = deg east of north, S/N is rough")
    for lab, d, w, rad in ((lab1, i1, w1, 12.0),
                           (lab4, i4, w4, 12.0),
                           ("MIPS 24", m24, w24, 25.0)):
        if not covered[lab]:
            say(f"  {lab}: skipped, the star is not on valid data here")
            continue
        nb, n_gap = find_neighbors(d, w, coord, rad)
        star_match = 1.5 if lab.startswith("IRAC") else 3.0
        others = [n for n in nb if n[0] > star_match]
        has_star = any(n[0] <= star_match for n in nb)
        gap_note = (f" ({n_gap} peak(s) next to a data gap ignored)"
                    if n_gap else "")
        say(f"  {lab}: star detected = {'yes' if has_star else 'no'}; "
            f"{len(others)} other peak(s) within {rad:.0f}\"{gap_note}")
        for sep, pa, snr in others:
            say(f"      sep {sep:5.1f}\"  PA {pa:5.0f}  S/N {snr:6.1f}")
        if lab.startswith("IRAC"):
            close = [n for n in others if n[0] <= 6.0]
            if close:
                flags.append(f"{len(close)} other {lab} source(s) within 6\" "
                             "of the star. These could blend into the 24um "
                             "beam.")
    say()

    # --- MIPS pointing check ------------------------------------------------
    notes = []
    pc = None
    if fit24 is not None:
        pc_lab, pc_img, pc_wcs = ((lab1, i1, w1) if covered[lab1]
                                  else (lab4, i4, w4))
        say(f"MIPS pointing check (is the whole 24um image shifted vs {pc_lab}?)")
        if not covered[pc_lab]:
            say("  skipped, the star is not on valid IRAC data")
        else:
            print("  matching field sources, this can take a minute...")
            pc = pointing_check(m24, w24, pc_img, pc_wcs, coord,
                                fit24["center"])
        if pc is None and covered[pc_lab]:
            say("  skipped, not enough IRAC data around the star")
        elif pc is not None:
            say(f"  other 24um sources found: {pc['n_det']}; matched to one "
                f"IRAC source: {pc['n_used']}")
            say(f"  (skipped: {pc['n_nomatch']} no IRAC match, "
                f"{pc['n_ambig']} more than one IRAC match, "
                f"{pc['n_badfit']} bad fit)")
            te, tn = pc["target"]
            ref = "its IRAC position" if pc["star_irac_ok"] else "catalog position"
            say(f"  {a.name} 24um center vs {ref}: east {te:+.2f}\", "
                f"north {tn:+.2f}\" (total {np.hypot(te, tn):.2f}\")")
            if "median" in pc:
                me, mn = pc["median"]
                se, sn = pc["scatter"]
                re_, rn = pc["residual"]
                say(f"  whole-image shift (median of field sources): east "
                    f"{me:+.2f}\", north {mn:+.2f}\" (total "
                    f"{np.hypot(me, mn):.2f}\")")
                say(f"  scatter of field sources: east {se:.2f}\", "
                    f"north {sn:.2f}\"")
                say(f"  {a.name} after removing the shift: east {re_:+.2f}\", "
                    f"north {rn:+.2f}\" (total {pc['residual_total']:.2f}\", "
                    f"{pc['chi']:.1f}x the field scatter)")
                verdict = offset_verdict(pc)
                if verdict == "ok":
                    say("  -> the leftover offset is within normal scatter. "
                        "The 24um offset is explained by pointing/measurement "
                        "error.")
                elif verdict == "small":
                    say(f"  -> small leftover offset (under {SMALL_OFFSET:.0f}\", "
                        "less than half a MIPS pixel). Probably measurement "
                        "error. Worth a note, not a red flag.")
                else:
                    say("  -> still offset after removing the shift. The 24um "
                        "light may not be centered on the star.")
            else:
                say("  too few matched sources (need 3+) to measure the "
                    "shift, so this check can't settle it")
        say()

    if fit24 is not None:
        verdict = offset_verdict(pc) if pc is not None else None
        if verdict == "ok":
            if fit24["offset"] > 2.0:
                notes.append(f"24um center is {fit24['offset']:.1f}\" from the "
                             "star, but the pointing check says that's normal "
                             "(whole-image shift plus scatter).")
        elif verdict == "small":
            notes.append(f"Small 24um offset of {pc['residual_total']:.2f}\" "
                         "after the pointing correction (under "
                         f"{SMALL_OFFSET:.0f}\"). Probably measurement error, "
                         "not a red flag.")
        elif verdict == "big":
            flags.append(f"24um center is {pc['residual_total']:.1f}\" from the "
                         "star after removing the image shift "
                         f"({pc['chi']:.1f}x the field scatter). The 24um light "
                         "may not come from the star.")
        elif fit24["offset"] > 2.0:
            flags.append(f"24um center is {fit24['offset']:.1f}\" from the "
                         "star (more than ~2\" is worth a close look; pointing "
                         "check couldn't settle it).")

    # --- figure -------------------------------------------------------------
    png = os.path.join(a.outdir, f"{a.name}_cutouts.png")
    plot_panels([(lab1, i1, w1), (lab4, i4, w4),
                 ("MIPS 24", m24, w24)],
                coord, a.name, a.wide, a.zoom, a.ap, a.ann_in, a.ann_out,
                fit24, png)

    # --- summary ------------------------------------------------------------
    say("Flags")
    if flags:
        for f in flags:
            say(f"  * {f}")
    else:
        say("  none. Still look at the PNG before trusting this.")
    if notes:
        say("Notes")
        for nmsg in notes:
            say(f"  * {nmsg}")
    if warnings:
        say("Warnings")
        for wmsg in warnings:
            say(f"  * {wmsg}")
    say()
    say(f"Figure saved: {png}")

    txt = os.path.join(a.outdir, f"{a.name}_24um_vetting.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Report saved: {txt}")


if __name__ == "__main__":
    main()
