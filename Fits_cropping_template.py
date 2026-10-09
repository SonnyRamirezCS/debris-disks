import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable 


# --- STEP 1: Enter your 3 explicit local file paths for 3.6, 8.0, 24.0 um ---
file_3_6  = '''3.6 um file goes here'''   # 3.6 um
file_8_0  = '''8.0 um file goes here'''   # 8.0 um
file_24   = '''24.0 um file goes here'''   # 24.0 um

file_configs = [
    {'path': file_3_6, 'scale': 0.60, 'label': '3.6 $\mu$m: Reference Photosphere'},
    {'path': file_8_0, 'scale': 0.60, 'label': '8.0 $\mu$m: Mid-IR Emissive Flux'},
    {'path': file_24,  'scale': 2.45, 'label': '24 $\mu$m: Far-IR Emissive Peak'}
]

file_configs = [
    {'path': file_3_6, 'scale': 0.60, 'label': '3.6 $\mu$m: Reference Photosphere'},
    {'path': file_8_0, 'scale': 0.60, 'label': '8.0 $\mu$m: Mid-IR Emissive Flux'},
    {'path': file_24,  'scale': 2.45, 'label': '24 $\mu$m: Far-IR Emissive Peak'}
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, cfg in zip(axes, file_configs):
    try:
        with fits.open(cfg['path']) as hdul:
            # --- FIXED LAYER LOGIC: Access hdul[0] or hdul[1] data fields directly ---
            if hdul[0].data is not None:
                raw_data = hdul[0].data
            elif len(hdul) > 1 and hdul[1].data is not None:
                raw_data = hdul[1].data
            else:
                raise ValueError("No valid data array found in FITS extensions.")
                
            data = np.nan_to_num(raw_data)
            if data.ndim == 3:
                data = data[0, :, :]
                
        height, width = data.shape
        center_y, center_x = height // 2, width // 2
        
        # --- Dynamic Slicing for Identical 49" Sky View ---
        if 'm1' in cfg['path'] or '24' in cfg['path']:
            crop_half = 10  # 20x20 pixels total (~49 arcseconds box)
        else:
            crop_half = 41  # 82x82 pixels total (~49 arcseconds box)
            
        cropped_data = data[center_y - crop_half : center_y + crop_half, 
                            center_x - crop_half : center_x + crop_half]
        
        # Logarithmic intensity scale mapping
        min_val = np.min(cropped_data)
        log_data = np.log10(cropped_data - min_val + 1.0)
        
        # Render the cropped matrix
        im = ax.imshow(log_data, cmap='inferno', origin='lower')
        ax.set_title(cfg['label'], fontsize=11, fontweight='bold', pad=12)
        
        # Colorbar attachment
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Log(Intensity)", fontsize=9, labelpad=5)
        cbar.ax.tick_params(labelsize=8)
        
        # --- Uniform Arcsecond Labeling (-24.5" to +24.5") ---
        total_pixels = cropped_data.shape[0] # Grab integer dimension shape safely
        pixel_ticks = np.linspace(0, total_pixels, 5)
        
        # Calculate sky boundaries uniformly
        arcsec_labels = [(tick - (total_pixels / 2)) * cfg['scale'] for tick in pixel_ticks]
        formatted_labels = [f"{val:+.1f}\"" if val != 0 else "0\"" for val in arcsec_labels]
        
        ax.set_xticks(pixel_ticks)
        ax.set_xticklabels(formatted_labels, fontsize=9)
        ax.set_yticks(pixel_ticks)
        ax.set_yticklabels(formatted_labels, fontsize=9)
        
        ax.set_xlabel("Sky Offset (Arcseconds)", fontsize=10)
        ax.set_ylabel("Sky Offset (Arcseconds)", fontsize=10)
        
        arrow_base_x = total_pixels * 0.25
        arrow_base_y = total_pixels * 0.12
        arrow_len = total_pixels * 0.15
        
        # North Arrow (Points straight Up in Y)
        ax.annotate('', xy=(arrow_base_x, arrow_base_y + arrow_len), xytext=(arrow_base_x, arrow_base_y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5, shrinkA=0, shrinkB=0))
        ax.text(arrow_base_x, arrow_base_y + arrow_len + (total_pixels * 0.02), 'N', 
                color='white', fontsize=10, fontweight='bold', ha='center', va='bottom')
        
        # East Arrow (Points straight Left in X - Astronomical Standard)
        ax.annotate('', xy=(arrow_base_x - arrow_len, arrow_base_y), xytext=(arrow_base_x, arrow_base_y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5, shrinkA=0, shrinkB=0))
        ax.text(arrow_base_x - arrow_len - (total_pixels * 0.02), arrow_base_y, 'E', 
                color='white', fontsize=10, fontweight='bold', ha='right', va='center')
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error\n{type(e).__name__}", ha='center', va='center', color='red')
        ax.axis('off')
        print(f"Error handling {cfg['label']}: {e}")

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()