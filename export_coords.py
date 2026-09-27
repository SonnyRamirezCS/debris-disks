import contextlib, io, warnings
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    from tyler_kstar_features import df_kdwarf
df_kdwarf[['objid', 'ra', 'dec']].to_csv(r'data\kdwarf_coords.csv', index=False)
print(len(df_kdwarf), "stars saved")