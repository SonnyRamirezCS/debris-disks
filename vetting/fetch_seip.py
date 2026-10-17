#!/usr/bin/env python3
"""
fetch_seip.py

Downloads Spitzer image cutouts around a star straight from IRSA, so you don't
have to click through the Spitzer Heritage Archive (which doesn't reliably mark
the target). Then it can run vet_24um.py on them for you.

It uses the SEIP super mosaics, which are the same images the SSTSL2 catalog
photometry was measured on. Only the cutout gets downloaded (a few MB per band),
not the whole mosaic.

What it does:
  1. Asks IRSA which SEIP mosaics cover the star (IRAC 3.6, 4.5, 8.0, MIPS 24).
  2. For each band, reads just the patch around the star and checks the star is
     on real data. If several mosaics cover it, the one with the catalog's
     smid is tried first.
  3. Saves the cutouts as data/<name>_seip_<band>.fits (git ignores data/*.fits).
  4. With --vet, runs vet_24um.py on them and saves the PNG and report in
     vetting/results/.

Our two candidates are built in:

    python vetting/fetch_seip.py J2133+83 --vet
    python vetting/fetch_seip.py J0845-78 --vet

Any other star:

    python vetting/fetch_seip.py --name MyStar --ra 131.32256 --dec -78.99365 \
        --ks 12.6 --cat-flux 510.4 --cat-err 15.6 --vet

If it can't find a band, run it with --list to see everything IRSA sent back.

Needs: astroquery, astropy, numpy, fsspec, aiohttp
    pip install astroquery astropy numpy fsspec aiohttp
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.wcs import WCS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

COLLECTION = "spitzer_seip"

# Values from data/kdwarf_sfr.csv. cat_flux/cat_err are the SSTSL2 24 um PSF
# fluxes in microJy. J2133's Ks is its k flux (3901 uJy) converted with a
# 666.7 Jy zero point; J0845's Ks is the one already in vet_24um.py.
PRESETS = {
    "J2133+83": dict(objid="SSTSL2 J213307.73+830501.2", ra=323.282217,
                     dec=83.083677, smid="40021360", ks=13.08,
                     cat_flux=509.0, cat_err=9.289),
    "J0845-78": dict(objid="SSTSL2 J084517.41-785937.1", ra=131.322558,
                     dec=-78.993645, smid="40010750", ks=12.633,
                     cat_flux=510.4, cat_err=15.56),
}

# band -> (instrument, IRAC channel number, wavelength range in microns)
BANDS = {
    "I1": ("IRAC", 1, (3.1, 4.0)),
    "I2": ("IRAC", 2, (4.0, 5.1)),
    "I3": ("IRAC", 3, (5.1, 6.6)),
    "I4": ("IRAC", 4, (6.6, 9.8)),
    "M1": ("MIPS", None, (18.0, 30.0)),
}
BAND_NAMES = {"I1": "IRAC 3.6", "I2": "IRAC 4.5", "I3": "IRAC 5.8",
              "I4": "IRAC 8.0", "M1": "MIPS 24"}


# ----------------------------------------------------------------------------
# Asking IRSA what covers the star
# ----------------------------------------------------------------------------

def query_irsa(coord):
    """SEIP images that contain the star, as an astropy Table."""
    from astroquery.ipac.irsa import Irsa
    try:
        res = Irsa.query_sia(pos=(coord, 1 * u.arcsec), collection=COLLECTION)
    except Exception as exc:                        # noqa: BLE001
        sys.exit(f"IRSA query failed: {exc}\n"
                 "Check your internet connection and try again in a bit.")
    if hasattr(res, "to_table"):
        res = res.to_table()
    return res


def col(row, name, default=""):
    return row[name] if name in row.colnames else default


def text_of(row):
    """Everything in the row that might name the band, joined together."""
    parts = [str(col(row, c)) for c in ("energy_bandpassname", "obs_id",
                                        "obs_publisher_did", "access_url",
                                        "instrument_name")]
    return " ".join(parts)


def band_of(row):
    """Which band an image is: from its wavelength range, else from its name."""
    try:
        lo = float(col(row, "em_min", np.nan)) * 1e6     # meters -> microns
        hi = float(col(row, "em_max", np.nan)) * 1e6
        mid = 0.5 * (lo + hi)
        if np.isfinite(mid):
            for band, (_, _, (a, b)) in BANDS.items():
                if a <= mid < b:
                    return band
    except (TypeError, ValueError):
        pass
    t = re.sub(r"[\s._\-]", "", text_of(row).upper())
    m = re.search(r"IRAC([1-4])(?!\d)", t)
    if m:
        return f"I{m.group(1)}"
    if re.search(r"MIPS(24|1)(?!\d)", t):
        return "M1"
    return None


def is_science(row):
    """True for the actual image, not the uncertainty or coverage maps."""
    sub = str(col(row, "dataproduct_subtype", "")).strip().lower()
    if sub:
        return sub == "science"
    fname = str(col(row, "access_url", "")).lower().split("?")[0].split("/")[-1]
    return not any(k in fname for k in ("unc", "cov", "std", "mask", "err"))


def candidates_by_band(table, coord, smid):
    """{band: [rows]} with the most likely mosaic first."""
    out = {}
    for row in table:
        band = band_of(row)
        if band is None or not is_science(row):
            continue
        out.setdefault(band, []).append(row)

    def rank(row):
        has_smid = bool(smid) and str(smid) in text_of(row)
        try:
            c = SkyCoord(float(col(row, "s_ra")) * u.deg,
                         float(col(row, "s_dec")) * u.deg)
            sep = coord.separation(c).deg
        except (TypeError, ValueError):
            sep = 999.0
        return (not has_smid, sep)

    for band in out:
        out[band].sort(key=rank)
    return out


def print_table(table):
    print(f"IRSA returned {len(table)} rows from {COLLECTION}:")
    for row in table:
        print(f"  band={band_of(row) or '?':3s} "
              f"science={'yes' if is_science(row) else 'no ':3s} "
              f"bandpass={col(row, 'energy_bandpassname')!s:12s} "
              f"subtype={col(row, 'dataproduct_subtype')!s:12s} "
              f"{col(row, 'access_url')}")


# ----------------------------------------------------------------------------
# Cutting out the patch around the star
# ----------------------------------------------------------------------------

WCS_KEY = re.compile(r"^((CD|PC)\d_\d|(CRPIX|CRVAL|CDELT|CTYPE|CUNIT|CROTA)\d|"
                     r"PV\d_\d+|PS\d_\d+|LONPOLE|LATPOLE|EQUINOX|EPOCH|"
                     r"RADESYS|RADECSYS|WCSAXES|BSCALE|BZERO|BLANK)$")


def cutout_header(orig, cut_wcs, band, url, size):
    """The mosaic's header with the cutout's WCS in place of the old one."""
    hdr = fits.Header()
    for card in orig.cards:
        key = card.keyword
        if not key or key in ("COMMENT", "HISTORY", "") or WCS_KEY.match(key):
            continue
        if key.startswith("NAXIS") or key in ("SIMPLE", "BITPIX", "EXTEND"):
            continue
        try:
            hdr[key] = (card.value, card.comment)
        except (ValueError, TypeError):
            pass
    hdr.update(cut_wcs.to_header())
    inst, ch, _ = BANDS[band]
    hdr["INSTRUME"] = inst            # vet_24um.py reads these to label panels
    if ch is not None:
        hdr["CHNLNUM"] = ch
    hdr["HISTORY"] = f"{size:.0f} arcsec cutout made by fetch_seip.py"
    hdr["HISTORY"] = f"from {url}"
    return hdr


def fetch_cutout(row, coord, size, band):
    """Read the patch around the star. Returns (data, header) or (None, reason)."""
    url = str(col(row, "access_url"))
    try:
        with fits.open(url, use_fsspec=True, lazy_load_hdus=True) as hdul:
            hdu = next((h for h in hdul if h.header.get("NAXIS", 0) == 2), None)
            if hdu is None:
                return None, "no 2D image in the file"
            wcs = WCS(hdu.header, naxis=2)
            try:
                cut = Cutout2D(hdu.section, coord, size * u.arcsec, wcs=wcs,
                               mode="trim")
            except NoOverlapError:
                return None, "star is off this mosaic"
            data = np.array(cut.data, dtype=np.float32)
            header = cutout_header(hdu.header, cut.wcs, band, url, size)
    except Exception as exc:                        # noqa: BLE001
        return None, f"couldn't read it ({type(exc).__name__}: {exc})"

    x, y = cut.wcs.world_to_pixel(coord)
    ix, iy = int(round(float(x))), int(round(float(y)))
    ny, nx = data.shape
    if not (0 <= ix < nx and 0 <= iy < ny) or not np.isfinite(data[iy, ix]):
        return None, "star lands in a gap or on the edge"
    if np.all(np.nan_to_num(data) == 0):
        return None, "cutout is all zeros (no coverage)"
    return (data, header), None


def get_band(cands, coord, size, band, outpath):
    """Try each mosaic for this band until one covers the star. Saves it."""
    for row in cands.get(band, []):
        url = str(col(row, "access_url"))
        print(f"  {BAND_NAMES[band]}: reading {url.split('/')[-1]} ...")
        got, why = fetch_cutout(row, coord, size, band)
        if got is None:
            print(f"    skipped: {why}")
            continue
        data, header = got
        fits.PrimaryHDU(data=data, header=header).writeto(outpath, overwrite=True)
        print(f"    saved {os.path.relpath(outpath)} ({data.shape[1]}x{data.shape[0]} px)")
        return outpath
    return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("star", nargs="?", choices=sorted(PRESETS),
                   help="one of our candidates (fills in everything below)")
    p.add_argument("--name")
    p.add_argument("--ra", type=float, help="deg, J2000")
    p.add_argument("--dec", type=float, help="deg, J2000")
    p.add_argument("--smid", default="", help="SEIP smid from the catalog (optional)")
    p.add_argument("--ks", type=float, default=0.0, help="2MASS Ks mag, for vet_24um")
    p.add_argument("--cat-flux", type=float, default=0.0, help="SSTSL2 24 um flux, uJy")
    p.add_argument("--cat-err", type=float, default=0.0, help="uJy")
    p.add_argument("--size", type=float, default=480.0,
                   help="cutout box in arcsec (vet_24um's pointing check wants "
                        "300\"+ of IRAC around the star)")
    p.add_argument("--datadir", default=os.path.join(REPO, "data"))
    p.add_argument("--outdir", default=os.path.join(HERE, "results"),
                   help="where vet_24um.py puts its PNG and report")
    p.add_argument("--vet", action="store_true", help="run vet_24um.py afterwards")
    p.add_argument("--list", action="store_true",
                   help="just print what IRSA has for this star")
    a = p.parse_args()

    if a.star:
        pre = PRESETS[a.star]
        a.name = a.name or a.star
        a.ra = a.ra if a.ra is not None else pre["ra"]
        a.dec = a.dec if a.dec is not None else pre["dec"]
        a.smid = a.smid or pre["smid"]
        a.ks = a.ks or pre["ks"]
        a.cat_flux = a.cat_flux or pre["cat_flux"]
        a.cat_err = a.cat_err or pre["cat_err"]
    if a.ra is None or a.dec is None or not a.name:
        p.error("give a candidate name (J2133+83 or J0845-78) or --name, --ra and --dec")

    coord = SkyCoord(a.ra * u.deg, a.dec * u.deg, frame="icrs")
    print(f"=== {a.name}  RA {a.ra}  Dec {a.dec} ===")
    print(f"Asking IRSA ({COLLECTION}) what covers the star...")
    table = query_irsa(coord)
    if a.list or len(table) == 0:
        print_table(table)
        if len(table) == 0:
            sys.exit("No SEIP images cover this position. Check the RA/Dec.")
        return

    cands = candidates_by_band(table, coord, a.smid)
    found = ", ".join(f"{BAND_NAMES[b]} ({len(r)})" for b, r in sorted(cands.items()))
    print(f"Mosaics found: {found or 'none'}")

    os.makedirs(a.datadir, exist_ok=True)
    path = lambda b: os.path.join(a.datadir, f"{a.name}_seip_{b}.fits")  # noqa: E731

    # IRAC 3.6 and 4.5 don't always cover the same sky, so 4.5 is the backup
    short = get_band(cands, coord, a.size, "I1", path("I1")) \
        or get_band(cands, coord, a.size, "I2", path("I2"))
    long8 = get_band(cands, coord, a.size, "I4", path("I4"))
    mips = get_band(cands, coord, a.size, "M1", path("M1"))

    missing = [n for n, f in (("IRAC 3.6/4.5", short), ("IRAC 8.0", long8),
                              ("MIPS 24", mips)) if f is None]
    if missing:
        print(f"\nCouldn't get: {', '.join(missing)}. Run again with --list "
              "to see what IRSA has.")
        sys.exit(1)

    vet_cmd = [sys.executable, os.path.join(HERE, "vet_24um.py"),
               "--mips", mips, "--irac1", short, "--irac4", long8,
               "--ra", str(a.ra), "--dec", str(a.dec), "--name", a.name,
               "--ks", str(a.ks), "--cat-flux", str(a.cat_flux),
               "--cat-err", str(a.cat_err), "--outdir", a.outdir]
    if not a.vet:
        print("\nDone. To vet these, run again with --vet, or:")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in vet_cmd))
        return

    print("\nRunning vet_24um.py...\n")
    sys.exit(subprocess.call(vet_cmd))


if __name__ == "__main__":
    main()
