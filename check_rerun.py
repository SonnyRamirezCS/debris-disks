"""
Check the pipeline rerun (tyler_kstar_features_v2, 3930-5380 K) against the last run.

Run from the debris-disks folder AFTER the full pipeline:
    export_coords -> SIMBAD XMatch -> young_star_check -> sfr_cut -> isoforest_v2 -> temp_groups
    python check_rerun.py

It reads the new data/*.csv files and compares them to the last run
(data/kdwarf_sfr.csv as committed in 50a6599, the 3500-5000 K sample).
It only reads files, it doesn't change anything.

Each line starts with:
    OK    looks right
    WARN  probably a problem, look at it before using the results
    INFO  a number worth knowing (paste these back)
"""
import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OLD_COMMIT = '50a6599'          # last run before the v2 switch
CANDIDATE = 'SSTSL2 J213307.73+830501.2'
TEFF_MIN, TEFF_MAX, OLD_TEFF_MAX = 3930, 5380, 5000
FEATURES = ['norm_chi24', 'norm_chi8', 'norm_chi45']
RJ_CORRECTION = 1.329           # what temp_groups.py currently assumes

parser = argparse.ArgumentParser()
parser.add_argument('--old', help='path to the old kdwarf_sfr.csv (default: read it from git)')
parser.add_argument('--seeds', type=int, default=50, help='Isolation Forest seeds for the stability check')
args = parser.parse_args()


def say(tag, msg):
    print(f'{tag:5s} {msg}')


def load_old():
    if args.old:
        return pd.read_csv(args.old)
    try:
        out = subprocess.run(['git', 'show', f'{OLD_COMMIT}:data/kdwarf_sfr.csv'], cwd=HERE,
                             capture_output=True, check=True)
        return pd.read_csv(io.BytesIO(out.stdout))
    except Exception as e:  # no git, or commit not there
        say('WARN', f"couldn't read the old run from git ({e.__class__.__name__}); "
                    f"pass --old path/to/old_kdwarf_sfr.csv to compare")
        return None


def as_bool(d, cols):
    for c in cols:
        if c in d.columns:
            d[c] = d[c].astype(str).str.lower().isin(['true', '1', '1.0'])
    return d


new = as_bool(pd.read_csv(DATA / 'kdwarf_sfr.csv'),
              ['young', 'evolved', 'clean', 'in_sfr', 'in_mc', 'in_plane', 'in_known', 'in_simbad'])
old = load_old()
if old is not None:
    old = as_bool(old, ['young', 'evolved', 'clean', 'in_sfr', 'in_mc', 'in_plane', 'in_known', 'in_simbad'])

# ======================================================================
print('\n=== 1. The K star sample (tyler_kstar_features_v2) ===')
say('INFO', f'stars in sample: {len(new)}, Teff {new.Teff.min():.0f} - {new.Teff.max():.0f} K')
if new.Teff.min() <= TEFF_MIN or new.Teff.max() >= TEFF_MAX:
    say('WARN', f'Teff outside {TEFF_MIN}-{TEFF_MAX} K, so the old feature script may have run')
else:
    say('OK', f'all stars inside {TEFF_MIN}-{TEFF_MAX} K')
if new.Teff.max() < TEFF_MAX - 200:
    say('WARN', f'hottest star is {new.Teff.max():.0f} K, the 5000-5380 K stars are missing')
n_hot = int((new.Teff >= OLD_TEFF_MAX).sum())
say('INFO', f'new hot stars (5000-5380 K): {n_hot}')

# v2 fills missing Teff/logg with the median of the WHOLE SEIP file before the K star cut.
# Stars with no Gaia Teff would all get the same value and could sneak into the sample.
for col in ['Teff', 'logg']:
    vc = new[col].value_counts()          # exact repeats only
    if vc.iloc[0] >= 5:
        say('WARN', f'{col} = {vc.index[0]} appears {vc.iloc[0]} times. That is probably the median '
                    f'fill in tyler_kstar_features_v2 (fillna before the cut), so those stars have no real {col}')
    else:
        say('OK', f'most repeated {col} value shows up {vc.iloc[0]} times (no big median-fill spike; '
                  f'last run: 3 for Teff, 4 for logg)')

# v2 log step = log10(chi clipped at 0.1). v1 was log10(chi + 1).
v2_log = np.log10(new['chi_24'].clip(lower=0.1))
if np.allclose(new['log_chi24'], v2_log, equal_nan=True):
    say('OK', 'log_chi24 = log10(chi_24 clipped at 0.1), so this is the v2 log step')
else:
    say('WARN', 'log_chi24 does not match the v2 log step, check which feature script ran')
for b in ['24', '8', '45']:
    n_floor = int((new[f'chi_{b}'] <= 0.1).sum())
    say('INFO', f'chi_{b} <= 0.1 (all squashed onto the same floor value): {n_floor}')

if old is not None:
    o = old[(old.Teff > TEFF_MIN) & (old.Teff < TEFF_MAX)]
    both = set(o.objid) & set(new.objid)
    gone = set(o.objid) - set(new.objid)
    extra = set(new.loc[new.Teff < OLD_TEFF_MAX, 'objid']) - set(o.objid)
    say('INFO', f'old run stars in 3930-5000 K: {len(o)}, found in new run: {len(both)}')
    if gone or extra:
        say('WARN', f'{len(gone)} old stars missing and {len(extra)} unexpected new stars below 5000 K. '
                    f'Either SEIP_data.csv changed or a cut changed')
    else:
        say('OK', 'same 3930-5000 K stars as last time, plus the new hot ones')
    m = o.merge(new, on='objid', suffixes=('_old', '_new'))
    d = (m['chi_24_old'] - m['chi_24_new']).abs().max()
    say('OK' if d < 1e-6 else 'WARN', f'chi_24 for the shared stars changed by at most {d:.2g} (should be 0)')

# ======================================================================
print('\n=== 2. SIMBAD cross match (young_star_check) ===')
coords_f, simbad_f = DATA / 'kdwarf_coords.csv', DATA / 'kdwarf_simbad.csv'
if coords_f.exists() and simbad_f.exists():
    coords, sim = pd.read_csv(coords_f), pd.read_csv(simbad_f)
    stale = set(sim.objid) - set(coords.objid)
    if stale:
        say('WARN', f'kdwarf_simbad.csv has {len(stale)} stars that are not in kdwarf_coords.csv. '
                    f'It is probably the OLD XMatch download. Redo the XMatch with the new coords file')
    else:
        say('OK', 'every star in kdwarf_simbad.csv is in the new coords file')
    say('INFO', f'XMatch rows: {len(sim)}, stars matched: {sim.objid.nunique()} of {len(coords)}, '
                f'largest match distance {sim.angDist.max():.2f} arcsec')
hot = new[new.Teff >= OLD_TEFF_MAX]
cool = new[new.Teff < OLD_TEFF_MAX]
if len(hot):
    fh, fc = hot['in_simbad'].mean(), cool['in_simbad'].mean()
    say('WARN' if fh < 0.5 * fc else 'OK',
        f'found in SIMBAD: {fh:.0%} of the hot stars vs {fc:.0%} of the others '
        f'(a big gap means the XMatch was not redone)')
say('INFO', f"young: {int(new.young.sum())} ({new.young.mean():.0%}), giants: {int(new.evolved.sum())}")

# ======================================================================
print('\n=== 3. Cuts (sfr_cut) ===')
say('INFO', f'clean stars: {int(new.clean.sum())} '
            f'({int((new.clean & (new.Teff < OLD_TEFF_MAX)).sum())} below 5000 K, '
            f'{int((new.clean & (new.Teff >= OLD_TEFF_MAX)).sum())} hot)')
if old is not None:
    m = old.merge(new[['objid', 'clean', 'in_sfr', 'n_any_near', 'n_young_near']], on='objid',
                  suffixes=('_old', '_new'))
    m = m[(m.Teff > TEFF_MIN) & (m.Teff < TEFF_MAX)]
    became = m[~m.clean_old & m.clean_new]
    lost = m[m.clean_old & ~m.clean_new]
    say('INFO', f'last run had {int(old.clean.sum())} clean, {int(m.clean_old.sum())} of them inside 3930-5380 K')
    say('INFO', f'shared stars that became clean: {len(became)}, stopped being clean: {len(lost)} '
                f'(neighbor counts change when the sample changes)')
    for _, r in lost.iterrows():
        say('INFO', f"  no longer clean: {r.objid.replace('SSTSL2 ', '')} "
                    f"neighbors {r.n_any_near_old:.0f} -> {r.n_any_near_new:.0f}, "
                    f"young neighbors {r.n_young_near_old:.0f} -> {r.n_young_near_new:.0f}")
j = new[new.objid == CANDIDATE]
if j.empty:
    say('WARN', 'J2133+83 is not in the new sample at all')
else:
    j = j.iloc[0]
    say('OK' if j.clean else 'WARN',
        f"J2133+83 clean = {j.clean} (neighbors {j.n_any_near:.0f}, young neighbors {j.n_young_near:.0f}, "
        f"b = {j.gal_b:.1f}, chi_24 = {j.chi_24:.1f})")
    rank = int((new.loc[new.clean, 'chi_24'] > j.chi_24).sum()) + 1
    say('INFO', f'J2133+83 is #{rank} by chi_24 among the clean stars')

# ======================================================================
print('\n=== 4. Isolation Forest (isoforest_v2) ===')
s = new[new.clean].copy()
n_feat = int(s.dropna(subset=FEATURES).shape[0])
say('OK' if n_feat == len(s) else 'WARN',
    f'clean stars with all three features: {n_feat} of {len(s)} '
    f'(v2 has no missing logs, so these should match)')
s = s.dropna(subset=FEATURES)
s = s[s.m1_f_psf / s.m1_df_psf >= 5].copy()
picks_f = DATA / 'isoforest_v2_picks.csv'
if picks_f.exists():
    picks = pd.read_csv(picks_f)
    say('INFO', f'Run B picks: {len(picks)}: ' +
        ', '.join(p.replace('SSTSL2 ', '') for p in picks.objid))
    say('INFO', f'J2133+83 in the picks: {CANDIDATE in set(picks.objid)} (with the v2 log step this is close to '
                f'a coin flip, see the seed check below; it does not change the size of its excess)')
    floor = picks[picks.chi_8 <= 0.1]
    if len(floor):
        say('WARN', f'{len(floor)} picks have chi_8 <= 0.1 (8 um fainter than Rayleigh-Jeans, which a normal '
                    f'K dwarf is not). They likely got flagged for sitting on the log floor, not for a disk: ' +
            ', '.join(p.replace('SSTSL2 ', '') for p in floor.objid))
if CANDIDATE in set(s.objid):
    flagged, ranks = 0, []
    for seed in range(args.seeds):
        iso = IsolationForest(contamination=0.05, random_state=seed).fit(s[FEATURES])
        w = -iso.score_samples(s[FEATURES])
        jw = w[(s.objid == CANDIDATE).to_numpy()][0]
        ranks.append(int((w > jw).sum()) + 1)
        flagged += int(iso.predict(s.loc[s.objid == CANDIDATE, FEATURES])[0] == -1)
    say('INFO', f'J2133+83 flagged in {flagged} of {args.seeds} random seeds, weirdness rank median '
                f'{int(np.median(ranks))} of {len(s)} (range {min(ranks)}-{max(ranks)}). '
                f'Last run: flagged 47 of 50, rank 4 of 104')

# ======================================================================
print('\n=== 5. Temperature groups (temp_groups) ===')
tg_f = DATA / 'clean_temp_groups.csv'
if tg_f.exists():
    tg = pd.read_csv(tg_f)
    tg['can_see_star'] = tg['can_see_star'].astype(str).str.lower().isin(['true', '1'])
    say('OK' if len(tg) == int(new.clean.sum()) else 'WARN',
        f'temp groups has {len(tg)} stars, clean count is {int(new.clean.sum())} (should match now)')
    say('INFO', 'stars per group: ' + ', '.join(f'{g} {n}' for g, n in tg.temp_group.value_counts().items()))
    say('INFO', f"Spitzer can see the star alone (comparable stars, with the {RJ_CORRECTION} correction): "
                f"{int(tg.can_see_star.sum())} (last run: 52)")
    plain = tg[tg.can_see_star & (tg.Ks_minus_24 < 0.75)]
    med = plain.ratio_rj.median()
    say('WARN' if abs(med - RJ_CORRECTION) > 0.05 else 'OK',
        f'plain stars (seen, Ks-[24] < 0.75): {len(plain)}, median measured/RJ = {med:.3f} '
        f'vs RJ_CORRECTION = {RJ_CORRECTION} hard-coded in temp_groups.py and sed_fit_j2133.py')
    jt = tg[tg.objid == CANDIDATE]
    if len(jt):
        jt = jt.iloc[0]
        say('INFO', f"J2133+83: {jt.temp_group} ({jt.spectral_type}), Teff {jt.Teff:.0f} K, "
                    f"near_edge = {jt.near_edge} (4700 K boundary is {abs(jt.Teff - 4700):.0f} K away)")
else:
    say('WARN', 'no data/clean_temp_groups.csv yet, run temp_groups.py')

print('\nNext: python j2133_numbers.py   (redoes J2133+83\'s numbers with this sample)')
