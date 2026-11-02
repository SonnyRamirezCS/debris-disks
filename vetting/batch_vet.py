#!/usr/bin/env python3
"""
batch_vet.py

Runs our 24 micron image check (fetch_seip.py + vet_24um.py) on every star
with a real 24 um excess in excess_shape.py (the 116 stars in Part B), then
compares the results to the KMeans shape groups (warm vs cool/flat).

The question: do the stars whose 24 um light is off-center end up in the
cool/flat group, like J0845-78 does? If they do, the cool/flat group is
probably full of background galaxy blends, and the clustering is telling
disk-like shapes apart from galaxy-like ones.

For each star it saves (in vetting/results/batch_vet.csv):
  offset_raw      how far the 24 um center is from the star, before fixing
                  the image pointing (arcsec)
  offset          same, after removing the whole-image shift (arcsec)
  offset_x_scatter   that offset divided by the normal scatter of field
                  sources (the "6.0x" number for J0845-78)
  offset_x_noise  same, but also counting the star's own centering noise
                  (faint 24 um sources wobble more, this is fairer to them)
  irac_within_6   other IRAC sources within 6" that could blend into the
                  24 um beam
  ...plus flux check, source width, S/N, and a status column.

Before looking at any results, these are the rules (same as vet_24um.py,
plus a stricter one at about J0845-78's level):
  off_center         offset_x_scatter >= 2.5 and offset >= 1.0"
                     (what vet_24um.py flags; J2133+83 just trips it at 1.06")
  off_center_strong  offset_x_scatter >= 5 and offset >= 2.0"
                     (clearly off, like J0845-78 at 6.0x and 2.4")
  possible_blend     another IRAC 3.6/4.5 or 8.0 source within 6"

Run from the debris-disks folder, in the disks environment:

    python vetting/batch_vet.py --test     # just J2133+83 and J0845-78 first
    python vetting/batch_vet.py            # all 116 (an hour or more)
    python vetting/batch_vet.py --summary  # redo only the group comparison
    python vetting/batch_vet.py --control  # same check on plain stars (no excess)

The control group: stars in excess_shape.csv with NO 24 um excess (under
3 sigma) whose 24 um light Spitzer still detects at S/N >= 5. Their 24 um
light comes from the star itself, so they show how often the check calls a
star "off-center" just from noise. If the excess stars are off-center much
more often than these, the offsets are real.

It saves after every star, so if it stops (internet drops, you close the
window), run the same command again and it picks up where it left off.
Downloaded cutouts go in data/ (git ignores data/*.fits), about 5 MB a star.
Cutout PNGs go in vetting/results/batch/ unless you add --no-png.

Needs: the disks environment (astroquery, astropy, numpy, pandas, scipy,
matplotlib, fsspec, aiohttp)
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fetch_seip as F   # noqa: E402
import vet_24um as V     # noqa: E402

# astropy's "fit may be unsuccessful" warning shows up for faint sources; the
# status and numbers in the CSV already show when a fit didn't work
warnings.filterwarnings("ignore", module="astropy.modeling")

DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(HERE, "results")
CSV_OUT = os.path.join(RESULTS, "batch_vet.csv")
PNG_DIR = os.path.join(RESULTS, "batch")
FIG_OUT = os.path.join(REPO, "figures", "batch_vet_groups.png")

# same settings vet_24um.py uses by default
AP, ANN_IN, ANN_OUT, APCORR, ZP = 7.0, 20.0, 32.0, 1.61, 7.17
SIZE = 480.0                 # cutout box, arcsec
KS_ZP = 666.7                # 2MASS Ks zero point, Jy

# rules, set before looking at the results
OFF_CHI, OFF_ARCSEC = 2.5, 1.0
STRONG_CHI, STRONG_ARCSEC = 5.0, 2.0
BLEND_R = 6.0                # arcsec
BRIGHT_SN = 10               # 24 um S/N cut for the "bright stars only" check

GROUP_NAME = {1: "warm", 2: "cool/flat", 0: "control"}
NAMED = {"SSTSL2 J213307.73+830501.2": "J2133+83",
         "SSTSL2 J084517.41-785937.1": "J0845-78"}
# Oct 22 single-star results, to check the batch reproduces them
OCT22 = {"J2133+83": dict(offset=1.06, chi=3.4, n_used=42),
         "J0845-78": dict(offset=2.38, chi=6.0, n_used=37)}


# ----------------------------------------------------------------------------
# Which stars
# ----------------------------------------------------------------------------

def load_targets(control=False):
    shape_csv = os.path.join(DATA, "excess_shape.csv")
    sfr_csv = os.path.join(DATA, "kdwarf_sfr.csv")
    for p in (shape_csv, sfr_csv):
        if not os.path.exists(p):
            sys.exit(f"Can't find {p}. Run excess_shape.py first.")
    shape = pd.read_csv(shape_csv)
    sfr = pd.read_csv(sfr_csv, low_memory=False,
                      usecols=["objid", "ra", "dec", "k", "m1_f_psf",
                               "m1_df_psf", "smid"])
    if control:
        # plain stars: no real 24 um excess, but the star itself is seen at 24 um
        shape = shape[shape["shape_group"].isna()].merge(
            sfr[["objid", "m1_f_psf", "m1_df_psf"]].drop_duplicates("objid"), on="objid")
        shape = shape[(shape["exsig_M1"] < 3) &
                      (shape["m1_f_psf"] / shape["m1_df_psf"] >= 5)].copy()
        shape = shape.drop(columns=["m1_f_psf", "m1_df_psf"])
        shape["shape_group"] = 0
    else:
        shape = shape[shape["shape_group"].notna()].copy()
    shape["shape_group"] = shape["shape_group"].astype(int)
    t = shape.merge(sfr.drop_duplicates("objid"), on="objid", how="left")
    if t["ra"].isna().any():
        bad = t.loc[t["ra"].isna(), "objid"].tolist()
        sys.exit(f"No coordinates in kdwarf_sfr.csv for: {bad}")
    t["short"] = t["objid"].map(NAMED).fillna("")
    # file-safe name, e.g. J102607.38+673823.2
    t["fname"] = t["objid"].str.replace("SSTSL2 ", "", regex=False)
    t["ks"] = -2.5 * np.log10(t["k"] * 1e-6 / KS_ZP)
    t["smid"] = t["smid"].apply(lambda s: "" if pd.isna(s) else str(int(s)))
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Getting the cutouts (reuses fetch_seip.py)
# ----------------------------------------------------------------------------

def cutout_paths(fname):
    p = lambda b: os.path.join(DATA, f"{fname}_seip_{b}.fits")  # noqa: E731
    short = p("I1") if os.path.exists(p("I1")) else (p("I2") if os.path.exists(p("I2")) else None)
    long8 = p("I4") if os.path.exists(p("I4")) else None
    mips = p("M1") if os.path.exists(p("M1")) else None
    return short, long8, mips


def query_with_retries(coord, waits=(15, 45)):
    """IRSA sometimes fails for a moment ("Unable to access the capabilities
    endpoint"), so try up to 3 times before giving up on a star."""
    for attempt in range(len(waits) + 1):
        try:
            return F.query_irsa(coord), None
        except SystemExit as exc:          # fetch_seip exits on IRSA errors
            why = str(exc).splitlines()[0]
            if attempt < len(waits):
                print(f"  IRSA didn't answer, trying again in {waits[attempt]} s...")
                time.sleep(waits[attempt])
    return None, why


def fetch(star, coord, refetch=False):
    """Returns (short, long8, mips, status, note)."""
    short, long8, mips = cutout_paths(star["fname"])
    if short and long8 and mips and not refetch:
        return short, long8, mips, "ok", "cutouts already downloaded"
    table, why = query_with_retries(coord)
    if table is None:
        return None, None, None, "irsa_error", why
    if len(table) == 0:
        return None, None, None, "no_seip_images", "IRSA has no SEIP images here"
    cands = F.candidates_by_band(table, coord, star["smid"])
    p = lambda b: os.path.join(DATA, f"{star['fname']}_seip_{b}.fits")  # noqa: E731
    if refetch:                            # don't let an old I2 file stand in for I1
        for b in ("I1", "I2", "I4", "M1"):
            if os.path.exists(p(b)):
                os.replace(p(b), p(b).replace(".fits", "_old.fits"))   # still git-ignored
    short = long8 = mips = None
    for attempt in range(2):               # second pass only for bands that failed
        if attempt:
            print("  some bands didn't download, trying those once more in 10 s...")
            time.sleep(10)
        short = short or F.get_band(cands, coord, SIZE, "I1", p("I1")) \
            or F.get_band(cands, coord, SIZE, "I2", p("I2"))
        long8 = long8 or F.get_band(cands, coord, SIZE, "I4", p("I4"))
        mips = mips or F.get_band(cands, coord, SIZE, "M1", p("M1"))
        if short and long8 and mips:
            break
    missing = [n for n, f in (("IRAC 3.6/4.5", short), ("IRAC 8.0", long8),
                              ("MIPS 24", mips)) if f is None]
    if missing:
        return short, long8, mips, "missing_band", "no usable " + ", ".join(missing)
    return short, long8, mips, "ok", ""


# ----------------------------------------------------------------------------
# The image check (same steps and settings as vet_24um.py's main)
# ----------------------------------------------------------------------------

def vet(star, coord, short, long8, mips, make_png):
    row = {}
    m24, w24, _ = V.load_image(mips)
    i1, w1, h1 = V.load_image(short)
    i4, w4, h4 = V.load_image(long8)
    lab1 = V.band_label(h1, "IRAC 3.6")
    lab4 = V.band_label(h4, "IRAC 8.0")
    row["irac_short_band"] = lab1

    covered = {}
    for lab, d, w in (("MIPS 24", m24, w24), (lab1, i1, w1), (lab4, i4, w4)):
        covered[lab] = V.star_on_image(d, w, coord)[0]
    if not covered["MIPS 24"]:
        row["status"], row["note"] = "not_on_mips", "star is not on valid MIPS 24 data"
        return row

    ph = V.photometry(m24, w24, coord, AP, ANN_IN, ANN_OUT, APCORR)
    row["flux_ujy"] = ph["flux_ujy"]
    row["flux_err_ujy"] = ph["err_ujy"]
    row["flux_vs_catalog"] = (ph["flux_ujy"] / star["m1_f_psf"]
                              if star["m1_f_psf"] > 0 else np.nan)
    sn = ph["flux_ujy"] / ph["err_ujy"] if ph["err_ujy"] > 0 else np.nan
    row["sn_24"] = sn

    fit24 = V.fit_source(m24, w24, coord, ph["sky"], ph["sky_std"], verbose=False)
    if fit24 is None:
        row["status"], row["note"] = "fit_failed", "24 um Gaussian fit failed"
    else:
        row["offset_raw"] = fit24["offset"]
        row["pa_raw"] = fit24["pa"]
        row["fwhm"] = fit24["fwhm_major"]
        row["peak_snr"] = fit24["peak_snr"]
        row["wide"] = fit24["fwhm_major"] > 1.3 * V.MIPS24_FWHM

    # neighbors
    for key, lab, d, w, rad in (("irac_short", lab1, i1, w1, 12.0),
                                ("irac8", lab4, i4, w4, 12.0),
                                ("mips", "MIPS 24", m24, w24, 25.0)):
        if not covered[lab]:
            row[f"{key}_star_seen"] = np.nan
            continue
        nb, _ = V.find_neighbors(d, w, coord, rad)
        star_match = 1.5 if key.startswith("irac") else 3.0
        others = [n for n in nb if n[0] > star_match]
        row[f"{key}_star_seen"] = any(n[0] <= star_match for n in nb)
        if key.startswith("irac"):
            close = [n for n in others if n[0] <= BLEND_R]
            row[f"{key}_within_6"] = len(close)
            row[f"{key}_nearest"] = others[0][0] if others else np.nan
        else:
            row["mips_others_25"] = len(others)
    # the same neighbor usually shows up in both IRAC bands, so take the larger count
    row["irac_within_6"] = int(max(row.get("irac_short_within_6", 0),
                                   row.get("irac8_within_6", 0)))
    row["possible_blend"] = row["irac_within_6"] > 0

    # pointing check
    if fit24 is not None:
        pc_lab, pc_img, pc_wcs = ((lab1, i1, w1) if covered[lab1] else (lab4, i4, w4))
        pc = None
        if covered[pc_lab]:
            pc = V.pointing_check(m24, w24, pc_img, pc_wcs, coord, fit24["center"])
        if pc is not None:
            row["field_found"] = pc["n_det"]
            row["field_used"] = pc["n_used"]
            row["star_irac_ok"] = pc["star_irac_ok"]
        if pc is not None and "chi" in pc:
            me, mn = pc["median"]
            se, sn_ = pc["scatter"]
            re_, rn = pc["residual"]
            row["shift_total"] = float(np.hypot(me, mn))
            row["scatter_e"], row["scatter_n"] = se, sn_
            row["offset_e"], row["offset_n"] = re_, rn
            row["offset"] = pc["residual_total"]
            row["offset_x_scatter"] = pc["chi"]
            # the star's own centering noise per axis, rough rule 0.6 * FWHM / S/N
            if np.isfinite(sn) and sn > 0:
                sig_t = 0.6 * V.MIPS24_FWHM / sn
                row["center_noise"] = sig_t
                row["offset_x_noise"] = float(np.hypot(re_ / np.hypot(se, sig_t),
                                                       rn / np.hypot(sn_, sig_t)))
            row["verdict"] = V.offset_verdict(pc)
            row["status"] = "ok"
        elif "status" not in row:
            row["status"] = "no_pointing_check"
            row["note"] = ("fewer than 3 field sources matched, so the offset "
                           "can't be compared to the normal scatter")

    if make_png:
        os.makedirs(PNG_DIR, exist_ok=True)
        png = os.path.join(PNG_DIR, f"{star['fname']}_cutouts.png")
        V.plot_panels([(lab1, i1, w1), (lab4, i4, w4), ("MIPS 24", m24, w24)],
                      coord, star["short"] or star["fname"], 90.0, 30.0,
                      AP, ANN_IN, ANN_OUT, fit24, png)
    return row


def add_flags(df):
    nan = pd.Series(np.nan, index=df.index)
    chi = pd.to_numeric(df.get("offset_x_scatter", nan), errors="coerce")
    off = pd.to_numeric(df.get("offset", nan), errors="coerce")
    for col, c_min, a_min in (("off_center", OFF_CHI, OFF_ARCSEC),
                              ("off_center_strong", STRONG_CHI, STRONG_ARCSEC)):
        flag = ((chi >= c_min) & (off >= a_min)).astype(object)
        flag[chi.isna()] = None              # not measured, so neither yes nor no
        df[col] = flag
    return df


# ----------------------------------------------------------------------------
# Comparing to the shape groups
# ----------------------------------------------------------------------------

def count_true(s):
    return int(s.map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).sum())


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else f"{a}/0"


def compare(df, label):
    from scipy.stats import fisher_exact, mannwhitneyu
    print(f"\n--- {label} ---")
    m = df[df["offset_x_scatter"].notna()]
    groups = [g for g in (1, 2, 0) if (m["shape_group"] == g).any()]
    rows = []
    for g in groups:
        s = m[m["shape_group"] == g]
        rows.append((g, len(s), count_true(s["off_center"]),
                     count_true(s["off_center_strong"]),
                     count_true(s["possible_blend"]),
                     s["offset_x_scatter"].median(), s["offset_x_noise"].median()))
    print(f"{'group':10s} {'checked':>7s}  {'off-center':>14s}  {'clearly off':>14s}  "
          f"{'possible blend':>14s}  {'median x scatter':>16s}  {'median x noise':>14s}")
    for g, n, a, b, c, md, mdn in rows:
        print(f"{GROUP_NAME[g]:10s} {n:7d}  {pct(a, n):>14s}  {pct(b, n):>14s}  "
              f"{pct(c, n):>14s}  {md:16.1f}  {mdn:14.1f}")
    by = {r[0]: r for r in rows}
    if 0 in by:
        _, n0, a0, b0, _, _, _ = by[0]
        for g in (1, 2):
            if g in by and by[g][1] and n0:
                _, n, a, b, _, _, _ = by[g]
                for name, x, x0 in (("off-center", a, a0), ("clearly off", b, b0)):
                    _, p = fisher_exact([[x, n - x], [x0, n0 - x0]])
                    print(f"  {name}: {GROUP_NAME[g]} {100 * x / n:.0f}% vs control "
                          f"{100 * x0 / n0:.0f}%, Fisher exact p = {p:.3f}")
    if 1 not in by or 2 not in by:
        return rows
    (_, nw, aw, bw, cw, _, _), (_, nc, ac, bc, cc, _, _) = by[1], by[2]
    if nw and nc:
        for name, xw, xc in (("off-center", aw, ac), ("clearly off", bw, bc),
                             ("possible blend", cw, cc)):
            _, p = fisher_exact([[xc, nc - xc], [xw, nw - xw]])
            print(f"  {name}: cool/flat {100 * xc / nc:.0f}% vs warm {100 * xw / nw:.0f}%, "
                  f"Fisher exact p = {p:.3f}")
        for col in ("offset_x_scatter", "offset_x_noise"):
            a = m.loc[m["shape_group"] == 2, col].dropna()
            b = m.loc[m["shape_group"] == 1, col].dropna()
            if len(a) and len(b):
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                print(f"  {col}: are cool/flat offsets bigger overall? "
                      f"Mann-Whitney p = {p:.3f}")
    return rows


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    if not os.path.exists(CSV_OUT):
        sys.exit(f"No results yet ({CSV_OUT}). Run without --summary first.")
    df = add_flags(pd.read_csv(CSV_OUT))
    t = load_targets()
    n_ctrl = int((df["shape_group"] == 0).sum())
    print("\n========== Image check vs shape groups ==========")
    print(f"Stars with a real 24 um excess: {len(t)} "
          f"(warm {int((t['shape_group'] == 1).sum())}, "
          f"cool/flat {int((t['shape_group'] == 2).sum())})")
    print(f"Checked so far: {len(df)}" + (f" (including {n_ctrl} control stars)" if n_ctrl else "") +
          ". Status counts:")
    for st, n in df["status"].value_counts().items():
        print(f"  {st}: {n}")

    print("\nThe rules (set before looking):")
    print(f"  off-center  = offset >= {OFF_ARCSEC}\" and >= {OFF_CHI}x the field scatter")
    print(f"  clearly off = offset >= {STRONG_ARCSEC}\" and >= {STRONG_CHI}x the field scatter")
    print(f"  possible blend = another IRAC source within {BLEND_R:.0f}\"")
    compare(df, "All checked stars")
    bright = df[pd.to_numeric(df["sn_24"], errors="coerce") >= BRIGHT_SN]
    compare(bright, f"Only stars with 24 um S/N >= {BRIGHT_SN} (centers are less noisy)")

    print("\nNamed stars:")
    for objid, nm in NAMED.items():
        r = df[df["objid"] == objid]
        if r.empty:
            print(f"  {nm}: not checked yet")
            continue
        r = r.iloc[0]
        print(f"  {nm}: {GROUP_NAME.get(int(r['shape_group']), '?')} group, "
              f"offset {r.get('offset', np.nan):.2f}\", "
              f"{r.get('offset_x_scatter', np.nan):.1f}x field scatter "
              f"({r.get('offset_x_noise', np.nan):.1f}x counting its own noise), "
              f"IRAC sources within 6\": {r.get('irac_within_6', np.nan):.0f}")

    # figure
    m = df[df["offset_x_scatter"].notna()]
    if m.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    rng = np.random.default_rng(0)
    colors = {1: "#d95f02", 2: "#1b9e77", 0: "#7570b3"}
    fig_groups = [g for g in (1, 2, 0) if (m["shape_group"] == g).any()]
    for i, g in enumerate(fig_groups):
        s = m[m["shape_group"] == g]
        x = i + rng.uniform(-0.18, 0.18, len(s))
        ax1.scatter(x, s["offset_x_scatter"].clip(lower=0.1), s=22, alpha=0.75,
                    color=colors[g], label=f"{GROUP_NAME[g]} ({len(s)})")
        ax2.scatter(s["sn_24"], s["offset"], s=22, alpha=0.75, color=colors[g],
                    label=f"{GROUP_NAME[g]}")
    for objid, nm in NAMED.items():
        r = m[m["objid"] == objid]
        if r.empty:
            continue
        r = r.iloc[0]
        i = fig_groups.index(int(r["shape_group"]))
        ax1.annotate(nm, (i, max(r["offset_x_scatter"], 0.1)), xytext=(i + 0.28, r["offset_x_scatter"]),
                     fontsize=9, va="center", arrowprops=dict(arrowstyle="-", color="0.4"))
        ax2.annotate(nm, (r["sn_24"], r["offset"]), xytext=(6, 6),
                     textcoords="offset points", fontsize=9)
    ax1.axhline(OFF_CHI, color="0.5", ls="--", lw=1)
    ax1.axhline(STRONG_CHI, color="0.3", ls=":", lw=1)
    ax1.text(-0.48, OFF_CHI, "off-center", va="bottom", fontsize=8, color="0.4")
    ax1.text(-0.48, STRONG_CHI, "clearly off", va="bottom", fontsize=8, color="0.3")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.set_xticks(range(len(fig_groups)), [GROUP_NAME[g] for g in fig_groups])
    ax1.set_xlim(-0.5, len(fig_groups) - 0.2)
    ax1.set_ylabel("24 um offset / normal field scatter")
    ax1.set_title("A. How off-center is the 24 um light, by shape group", loc="left", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.axhline(OFF_ARCSEC, color="0.5", ls="--", lw=1)
    ax2.axhline(STRONG_ARCSEC, color="0.3", ls=":", lw=1)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 30, 50, 100]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("24 um S/N (brighter to the right)")
    ax2.set_ylabel("24 um offset after pointing fix (arcsec)")
    ax2.set_title("B. Offset vs brightness (faint sources wobble more)", loc="left", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved {os.path.relpath(FIG_OUT, REPO)}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true",
                   help="only J2133+83 and J0845-78, and compare to the Oct 22 runs")
    p.add_argument("--control", action="store_true",
                   help="check the plain (no excess) stars instead, as a control group")
    p.add_argument("--summary", action="store_true",
                   help="skip the image check, just redo the group comparison")
    p.add_argument("--redo", action="store_true",
                   help="recheck stars already in batch_vet.csv")
    p.add_argument("--retry", action="store_true",
                   help="also retry stars that came back missing_band or error "
                        "(for example after an internet hiccup)")
    p.add_argument("--refetch", action="store_true",
                   help="download the cutouts again even if they're already in data/")
    p.add_argument("--no-png", action="store_true", help="don't save cutout PNGs")
    p.add_argument("--limit", type=int, default=0, help="stop after this many new stars")
    a = p.parse_args()

    if a.summary:
        summary()
        return

    t = load_targets(control=a.control)
    if a.test:
        t = t[t["short"] != ""]
    done = pd.read_csv(CSV_OUT) if os.path.exists(CSV_OUT) else pd.DataFrame()
    if not done.empty and a.retry:
        again = done.loc[done["status"].isin(["missing_band", "error"]), "objid"]
        done = done[~done["objid"].isin(again)]
    if not done.empty and not a.redo:
        todo = t[~t["objid"].isin(done["objid"])]
    else:
        todo = t
        if a.redo and not done.empty:
            done = done[~done["objid"].isin(t["objid"])]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(t)} stars selected, {len(t) - len(todo)} already done, "
          f"{len(todo)} to check.")
    os.makedirs(RESULTS, exist_ok=True)

    fails_in_a_row = 0
    t0 = time.time()
    for i, (_, star) in enumerate(todo.iterrows(), 1):
        label = star["short"] or star["fname"]
        coord = SkyCoord(star["ra"] * u.deg, star["dec"] * u.deg, frame="icrs")
        print(f"\n[{i}/{len(todo)}] {label}  ({GROUP_NAME[star['shape_group']]} group)")
        row = {"objid": star["objid"], "name": label,
               "shape_group": star["shape_group"],
               "group_name": GROUP_NAME[star["shape_group"]],
               "comparable": star["comparable"], "ratio_M1": star["ratio_M1"],
               "T_dust": star["T_dust"], "ra": star["ra"], "dec": star["dec"],
               "cat_flux_ujy": star["m1_f_psf"], "cat_err_ujy": star["m1_df_psf"]}
        try:
            short, long8, mips, status, note = fetch(star, coord, a.refetch)
            if status == "ok":
                row.update(vet(star, coord, short, long8, mips, not a.no_png))
                row.setdefault("status", "ok")
            else:
                row["status"], row["note"] = status, note
        except KeyboardInterrupt:
            print("\nStopped. Everything before this star is saved; run the same "
                  "command again to continue.")
            break
        except Exception as exc:                        # noqa: BLE001
            row["status"] = "error"
            row["note"] = f"{type(exc).__name__}: {exc}"

        if row["status"] == "irsa_error":
            fails_in_a_row += 1
            print(f"  IRSA error: {row.get('note', '')}")
            if fails_in_a_row >= 5:
                print("\n5 IRSA errors in a row, so IRSA or your internet is "
                      "probably down. Nothing was saved for these. Try again "
                      "later with the same command.")
                break
            continue                      # don't save, so a rerun tries again
        fails_in_a_row = 0

        msg = row["status"]
        if row["status"] == "ok":
            msg = (f"offset {row['offset']:.2f}\" = {row['offset_x_scatter']:.1f}x "
                   f"field scatter ({row.get('offset_x_noise', np.nan):.1f}x with its own noise), "
                   f"S/N {row['sn_24']:.0f}, IRAC within 6\": {row['irac_within_6']}")
        elif row.get("note"):
            msg += f": {row['note']}"
        print(f"  -> {msg}")
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        add_flags(done).to_csv(CSV_OUT, index=False)
        el = time.time() - t0
        print(f"  ({el / 60:.1f} min so far, about {el / i * (len(todo) - i) / 60:.0f} min left)")

    if a.test and not done.empty:
        print("\n========== Test: batch vs the Oct 22 single-star runs ==========")
        for nm, ref in OCT22.items():
            r = done[done["name"] == nm]
            if r.empty or r.iloc[0]["status"] != "ok":
                print(f"  {nm}: no result")
                continue
            r = r.iloc[0]
            print(f"  {nm}: offset {r['offset']:.2f}\" (Oct 22: {ref['offset']}), "
                  f"{r['offset_x_scatter']:.1f}x scatter (Oct 22: {ref['chi']}), "
                  f"field sources used {int(r['field_used'])} (Oct 22: {ref['n_used']})")
        print("If these match, run the full batch:  python vetting/batch_vet.py")
        return

    print(f"\nSaved {os.path.relpath(CSV_OUT, REPO)}")
    summary()


if __name__ == "__main__":
    main()
