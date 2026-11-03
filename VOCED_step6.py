import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

fits_file = r'data\SPITZER_I1_22839808_0000_6_E8143254_maic.fits'

# ---------- 1. open the FITS file and look at the header ----------
hdul = fits.open(fits_file)
hdul.info()

header = hdul[0].header
data = hdul[0].data
print('Image size (rows, columns):', data.shape)
print('Telescope:', header.get('TELESCOP'), ' Instrument:', header.get('INSTRUME'), ' Channel:', header.get('CHNLNUM'))
print('Units:', header.get('BUNIT'))
print('Reference pixel:', header.get('CRPIX1'), header.get('CRPIX2'), ' -> RA, Dec:', header.get('CRVAL1'), header.get('CRVAL2'))

# ---------- 2. view the whole image ----------
# most pixels are dark sky, so stretch the colors between the 1st and 99th percentile
vmin, vmax = np.nanpercentile(data, [1, 99])

plt.figure()
plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(label=f"Surface brightness ({header.get('BUNIT')})")
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Full Spitzer IRAC 3.6 micron image')
plt.savefig(r'figures\step6_full_image.png', dpi=200)

# ---------- 3. use the WCS to go between pixels and sky coordinates ----------
wcs = WCS(header)

# pixel -> sky: what RA and Dec is the middle of the image?
ny, nx = data.shape
middle = wcs.pixel_to_world(nx / 2, ny / 2)
print('Middle of the image (pixel', nx // 2, ny // 2, ') is at RA, Dec =', middle.to_string('hmsdms'))

# sky -> pixel: where is our star?
# SSTSL2 J012412.12-741735.2 -> RA 01h24m12.12s, Dec -74d17m35.2s
star = SkyCoord('01h24m12.12s', '-74d17m35.2s')
x_star, y_star = wcs.world_to_pixel(star)
print(f'Star is at pixel x = {x_star:.2f}, y = {y_star:.2f}')

# check: convert that pixel back to the sky and see how far off it lands
back = wcs.pixel_to_world(x_star, y_star)
print('Converted back to RA, Dec:', back.to_string('hmsdms'),
      ' difference:', round(back.separation(star).to(u.arcsec).value, 4), 'arcsec')

# ---------- 4. crop a 30 x 30 pixel square with the star in the middle ----------
size = 30
# the middle of a 30 pixel box is 14.5, so start the box 14.5 pixels before the star
x_start = int(round(float(x_star) - 14.5))
y_start = int(round(float(y_star) - 14.5))
cutout = data[y_start:y_start + size, x_start:x_start + size]
wcs_cut = wcs[y_start:y_start + size, x_start:x_start + size]   # the WCS for just the cropped piece
print('Cutout size:', cutout.shape)

x_in_cut = x_star - x_start
y_in_cut = y_star - y_start
print(f'Star position inside the cutout: x = {x_in_cut:.2f}, y = {y_in_cut:.2f} (the center is 14.5, 14.5)')

plt.figure()
ax = plt.subplot(projection=wcs_cut)
plt.imshow(cutout, origin='lower', cmap='inferno')
plt.colorbar(label=f"Surface brightness ({header.get('BUNIT')})")
plt.plot(x_in_cut, y_in_cut, '+', color='cyan', markersize=14, label='SSTSL2 J012412.12-741735.2')
# the image is rotated on the sky, so let the WCS put the RA and Dec labels where those ticks are
ax.coords[0].set_axislabel('RA')
ax.coords[1].set_axislabel('Dec')
ax.coords.grid(color='white', alpha=0.3, linestyle='dotted')
plt.title('30 x 30 pixel cutout at 3.6 microns', pad=14)
plt.legend(loc='upper right', fontsize=8)
plt.savefig(r'figures\step6_cutout.png', dpi=200)
plt.show()

hdul.close()
