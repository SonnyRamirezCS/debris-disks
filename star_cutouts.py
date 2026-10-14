"""
3-panel cutouts (3.6, 8.0, 24 um) centered on our candidate star.
Based on Tyler's Fits_cropping_template.py, with these changes:
  - crops around the star's RA/Dec using the image WCS, not the image center
    (the SHA mosaics are centered on the white dwarf WD 2136+828, ~2' away)
  - box size is set in arcseconds, so every band shows the same patch of sky
  - N and E arrows come from the WCS, so they're right even if the image is rotated
  - a small circle marks the star's position

Any star, from the command line (defaults are J2133+83 and the paths below):
    python star_cutouts.py --name J0845-78 --ra 131.32254 --dec -78.99364 --box 120 ^
        --i1 path\\to\\SPITZER_I1_..._maic.fits --i4 path\\to\\SPITZER_I4_..._maic.fits ^
        --m1 path\\to\\SPITZER_M1_..._maic.fits
(--box 120 shows a wider patch, handy for spotting spikes or rings from a bright star nearby)
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- STEP 1: the star and the box size ---
STAR_NAME = 'J2133+83'
STAR_RA, STAR_DEC = 323.2822, 83.0837
BOX_ARCSEC = 49.0  # same 49" box Tyler used

# --- STEP 2: paste your local file paths (the _maic.fits files from SHA) ---
file_3_6 = r'3.6 um file goes here'   # IRAC ch1, SPITZER_I1_..._maic.fits
file_8_0 = r'8.0 um file goes here'   # IRAC ch4, SPITZER_I4_..._maic.fits
file_24  = r'24.0 um file goes here'  # MIPS 24,  SPITZER_M1_..._maic.fits

# Command line options override the values above
parser = argparse.ArgumentParser()
parser.add_argument('--name', default=STAR_NAME)
parser.add_argument('--ra', type=float, default=STAR_RA, help='degrees')
parser.add_argument('--dec', type=float, default=STAR_DEC, help='degrees')
parser.add_argument('--box', type=float, default=BOX_ARCSEC, help='box size in arcsec')
parser.add_argument('--i1', default=file_3_6, help='3.6 um (or 4.5 um) file')
parser.add_argument('--i4', default=file_8_0, help='8.0 um file')
parser.add_argument('--m1', default=file_24, help='24 um file')
args = parser.parse_args()
STAR_NAME, BOX_ARCSEC = args.name, args.box
STAR = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame='icrs')
file_3_6, file_8_0, file_24 = args.i1, args.i4, args.m1

# Heads up: IRAC's 3.6/5.8 and 4.5/8.0 cameras look at two side-by-side patches of sky.
# If the 3.6 file doesn't cover the star, use the 4.5 file (I2) from the same AOR instead.
# The panel label is picked automatically from the file name.
file_configs = [{'path': file_3_6}, {'path': file_8_0}, {'path': file_24}]

BAND_LABELS = {
    '_I1_': r'3.6 $\mu$m: Reference Photosphere',
    '_I2_': r'4.5 $\mu$m: Reference Photosphere',
    '_I3_': r'5.8 $\mu$m: Mid-IR',
    '_I4_': r'8.0 $\mu$m: Mid-IR Emissive Flux',
    '_M1_': r'24 $\mu$m: Far-IR Emissive Peak',
}
for cfg in file_configs:
    name = cfg['path'].replace('\\', '/').split('/')[-1].upper()
    cfg['label'] = next((v for k, v in BAND_LABELS.items() if k in name), name)


def load_image(path):
    """Return (2D data, WCS) from the first HDU that has image data."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = hdu.data
                if data.ndim == 3:
                    data = data[0]
                return data.astype(float), WCS(hdu.header).celestial
    raise ValueError('No image data found in this FITS file.')


def sky_arrow(wcs, coord, length_px):
    """Pixel directions of North and East at `coord`, scaled to length_px."""
    x0, y0 = wcs.world_to_pixel(coord)
    out = {}
    for name, offset in (('N', coord.directional_offset_by(0 * u.deg, 5 * u.arcsec)),
                         ('E', coord.directional_offset_by(90 * u.deg, 5 * u.arcsec))):
        x1, y1 = wcs.world_to_pixel(offset)
        v = np.array([x1 - x0, y1 - y0], dtype=float)
        out[name] = v / np.linalg.norm(v) * length_px
    return out


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, cfg in zip(axes, file_configs):
    try:
        data, wcs = load_image(cfg['path'])
        cut = Cutout2D(data, STAR, (BOX_ARCSEC * u.arcsec, BOX_ARCSEC * u.arcsec),
                       wcs=wcs, mode='partial', fill_value=np.nan)
        if not np.any(np.isfinite(cut.data)):
            ax.text(0.5, 0.5, 'No data at the star\nin this file', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(cfg['label'], fontsize=11, fontweight='bold', pad=12)
            ax.set_xticks([]); ax.set_yticks([])
            print(f"{cfg['label']}: this file doesn't cover the star")
            continue
        img = np.nan_to_num(cut.data, nan=np.nanmin(cut.data))

        # log stretch, same idea as Tyler's version
        log_data = np.log10(img - np.min(img) + 1.0)
        im = ax.imshow(log_data, cmap='inferno', origin='lower')
        ax.set_title(cfg['label'], fontsize=11, fontweight='bold', pad=12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Log(Intensity)', fontsize=9, labelpad=5)
        cbar.ax.tick_params(labelsize=8)

        # mark the star
        sx, sy = cut.wcs.world_to_pixel(STAR)
        ny, nx = img.shape
        ax.add_patch(plt.Circle((sx, sy), radius=0.12 * nx, fill=False,
                                color='cyan', lw=1.2, ls='--'))

        # arcsecond tick labels from the real pixel scale
        scale = (cut.wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value
        ticks = np.linspace(0, nx - 1, 5)
        labels = [f'{(t - (nx - 1) / 2) * scale:+.1f}"' for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Sky Offset (Arcseconds)', fontsize=10)
        ax.set_ylabel('Sky Offset (Arcseconds)', fontsize=10)

        # N and E arrows from the WCS
        base = np.array([0.2 * nx, 0.2 * ny])
        arrows = sky_arrow(cut.wcs, STAR, 0.15 * nx)
        for name, vec in arrows.items():
            tip = base + vec
            ax.annotate('', xy=tip, xytext=base,
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5,
                                        shrinkA=0, shrinkB=0))
            ax.text(*(base + vec * 1.25), name, color='white', fontsize=10,
                    fontweight='bold', ha='center', va='center')

        print(f"{cfg['label']}: {nx}x{ny} px at {scale:.2f}\"/px")

    except Exception as e:
        ax.text(0.5, 0.5, f'Error\n{type(e).__name__}', ha='center', va='center',
                color='red', transform=ax.transAxes)
        ax.axis('off')
        print(f"Error handling {cfg['label']}: {e}")

fig.suptitle(f'{STAR_NAME}  ({BOX_ARCSEC:.0f}" box, dashed circle = star)', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(wspace=0.55)
plt.savefig(f'{STAR_NAME}_cutouts.png', dpi=150)
plt.show()
