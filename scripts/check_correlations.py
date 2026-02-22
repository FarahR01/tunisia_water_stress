import pandas as pd
import sys
candidates=['data/processed/processed_tunisia.csv','data/cleaned_water_stress.csv','data/raw/environment_tun.csv']
df=None
for p in candidates:
    try:
        df=pd.read_csv(p, index_col=None)
        print('Loaded:',p)
        break
    except Exception as e:
        pass
if df is None:
    print('No processed file found. Aborting.')
    sys.exit(2)
cols=[c.lower() for c in df.columns]
if 'indicator name' in cols and 'year' in cols and 'value' in cols:
    # world bank long form
    if 'country name' in cols:
        df=df[df['Country Name'].str.contains('Tunisia', na=False) | df.get('Country Name').isnull()]
    pivot=df.pivot_table(index='Year', columns='Indicator Name', values='Value')
    dfp=pivot.sort_index()
else:
    if 'year' in cols:
        try:
            df=df.set_index('Year')
        except Exception:
            pass
    dfp=df.copy()
# convert to numeric
dfp=dfp.apply(pd.to_numeric, errors='coerce')
# identify target
targets=[c for c in dfp.columns if 'water stress' in str(c).lower()]
if not targets:
    print('No water-stress column name found in columns. Columns sample:')
    print(list(dfp.columns[:40]))
    sys.exit(1)
target=targets[0]
print('Using target column:',target)
# correlation
corr=dfp.corr().abs()
if target not in corr:
    print('Target not in correlation matrix; abort')
    sys.exit(1)
series=corr[target].sort_values(ascending=False)
leakers=series[series>=0.90]
print('\nFeatures with |corr| >= 0.90 to target:\n')
print(leakers.to_string())
print('\nFull top-10 correlations:\n')
print(series.head(10).to_string())
