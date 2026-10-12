from pathlib import Path
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  # Moved to top
import seaborn as sns 
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(Path(__file__).with_name("SEIP_data.csv")) 

# STEP 1: Cleaning and Filtering the Data 
cols_to_clean = ['Teff', 'logg', 'm1_f_psf', 'm1_snr','i2_f_ap2','i2_snr', 'i4_f_ap2','i4_snr','i1_f_ap2'] 
df[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].median()) 

# Isolating K-Dwarfs 
mask = ( 
    (df['mips_obstype'] == 0) & 
    (df['Teff'] > 3930) & (df['Teff'] < 5380) & 
    (df['logg'] > 3.8) 
) 
df_kdwarf = df[mask].copy() 

# STEP 2: Feature Engineering for ML: Calculating Chi Values
def add_infrared_features(df): 
    RJ_24 = (3.6 / 24.0)**2 
    RJ_8 = (3.6 / 8.0)**2 
    RJ_45 = (3.6 / 4.5)**2 
    RJ_36 = (3.6 / 3.6)**2  
    
    # Calculate Chi_24 
    df['m1_err'] = df['m1_f_psf'] / df['m1_snr'].replace(0, np.nan) 
    df['m1_pred'] = df['i1_f_ap2'] * RJ_24 
    df['chi_24'] = (df['m1_f_psf'] - df['m1_pred']) / df['m1_err'] 
    
    # Calculate Chi_8 
    df['i4_err'] = df['i4_f_ap2'] / df['i4_snr'].replace(0, np.nan) 
    df['i4_pred'] = df['i1_f_ap2'] * RJ_8 
    df['chi_8'] = (df['i4_f_ap2'] - df['i4_pred']) / df['i4_err'] 
    
    # Calculate Chi_4.5 
    df['i2_err'] = df['i2_f_ap2'] / df['i2_snr'].replace(0, np.nan) 
    df['i2_pred'] = df['i1_f_ap2'] * RJ_45 
    df['chi_45'] = (df['i2_f_ap2'] - df['i2_pred']) / df['i2_err'] 

    # Calculate Chi_3.6 (Baseline)
    df['i1_err'] = df['i1_f_ap2'] / df['i1_snr'].replace(0, np.nan) 
    df['i1_pred'] = df['i1_f_ap2'] * RJ_36 
    df['chi_36'] = (df['i1_f_ap2'] - df['i1_pred']) / df['i1_err']
    
    # Log Transforms 
    df['log_chi24'] = np.log10(df['chi_24'].clip(lower=0.1)) 
    df['log_chi8'] = np.log10(df['chi_8'].clip(lower=0.1)) 
    df['log_chi45'] = np.log10(df['chi_45'].clip(lower=0.1)) 
    df['log_chi36'] = np.log10(df['chi_36'].clip(lower=0.1)) 
    # Flags 
    df['reliable_detection'] = (df['m1_fluxtype'] == 1) & (df['i1_fluxtype'] == 1) 
    return df 

df_kdwarf = add_infrared_features(df_kdwarf) 
df_kdwarf = df_kdwarf[df_kdwarf['reliable_detection']].copy() 

# STEP 3: Normalization and Scaling
# Scale Teff (Min-Max) 
t_min, t_max = df_kdwarf['Teff'].min(), df_kdwarf['Teff'].max() 
df_kdwarf['Teff_scaled'] = (df_kdwarf['Teff'] - t_min) / (t_max - t_min) 

# Scale Log_Chi Features
scaler = RobustScaler() 
df_kdwarf[['norm_chi24', 'norm_chi8', 'norm_chi45', 'norm_chi36']] = scaler.fit_transform(df_kdwarf[['log_chi24', 'log_chi8', 'log_chi45', 'log_chi36']]) 

# Define 'is_candidate' column 
df_kdwarf['is_candidate'] = ((df_kdwarf['norm_chi24'] > 1.0) & (df_kdwarf['norm_chi8'] > 1.0)).astype(int) 

# --- Outlier Data Isolation & Logging --- 
outliers_high24 = df_kdwarf[df_kdwarf['norm_chi24'] > 1.05].copy() 
outliers_high8 = df_kdwarf[df_kdwarf['norm_chi8'] > 1.10].copy() 
top_candidates = df_kdwarf[(df_kdwarf['norm_chi24'] > 1.0) & (df_kdwarf['norm_chi8'] > 1.0)].copy() 

top_52 = df_kdwarf[(df_kdwarf['norm_chi24'] > 1.0) | (df_kdwarf['norm_chi8'] > 1.0)].copy() # Added .copy()
top_52['total_excess_score'] = top_52['norm_chi24'] + top_52['norm_chi8'] 
top_50 = top_52.sort_values(by='total_excess_score', ascending=False).head(50) 

print(f"Found {len(outliers_high24)} high outliers in norm_chi24 and {len(outliers_high8)} high outliers in norm_chi8.") 

identifier_cols24 = ['objid', 'Teff','chi_8', 'norm_chi8', 'chi_24', 'norm_chi24', 'norm_chi45'] 

print("--- Identified Outlier Stars ---") 
print(outliers_high24[identifier_cols24]) 
print("\n\n--- Identified Outlier Stars (8μm) ---") 
print(outliers_high8[identifier_cols24]) 
print("\n\n--- Top Candidates with Both 8μm and 24μm Excess ---") 
print(top_candidates[identifier_cols24]) 
print("\n\n--- Top 50 Candidates by Combined Excess Score ---") 
print(top_50[identifier_cols24]) 


target_star_id = 'SSTSL2 J213307.73+830501.2'

star_data = df_kdwarf[df_kdwarf['objid'] == target_star_id].copy()

if not star_data.empty:
    print(f"\n--- Data for Star {target_star_id} ---")
    print(star_data[['objid', 'chi_24', 'norm_chi24', 'chi_8', 'norm_chi8', 'chi_45', 'norm_chi45', 'chi_36', 'norm_chi36']])
else:
    print(f"\nStar {target_star_id} was not found in the final K-dwarf dataset.")
    # Optional: Check if it was removed during the cleaning or filtering steps
    if target_star_id in df['objid'].values:
        print("Note: This star exists in the raw data but was filtered out (e.g., not a K-dwarf or lacked a reliable detection).")

export_csv = Path(__file__).with_name("SEIP_data_cleaned.csv")
top_50.to_csv("RJ_Candidates.csv", index=False)
