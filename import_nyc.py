"""Optional converter for NYC DOT Automated Traffic Volume Counts.
Download a CSV from https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/data
Choose a date range of at least 10 days and several locations. Export filtered CSV.
Run: python import_nyc.py raw_nyc.csv normalized_nyc.csv
Then: python traffic_project.py normalized_nyc.csv
"""
import sys
import pandas as pd
if len(sys.argv)!=3:
    raise SystemExit('Usage: python import_nyc.py raw_nyc.csv normalized_nyc.csv')
d=pd.read_csv(sys.argv[1]); d.columns=d.columns.str.strip().str.lower()
cols=['boro','yr','m','d','hh','mm','vol','segmentid','street','direction','requestid']
d=d[cols].dropna().drop_duplicates().copy()
for c in ['yr','m','d','hh','mm','vol']: d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.dropna(); d=d[(d.vol>=0)&d.mm.isin([0,15,30,45])]
year=d.yr.where(d.yr>=100,d.yr+2000)
d['timestamp']=pd.to_datetime(dict(year=year,month=d.m,day=d.d,hour=d.hh),errors='coerce')
d['location']=d.boro.astype(str)+' / '+d.street.astype(str)+' / '+d.segmentid.astype(str)+' / '+d.direction.astype(str)
d=d.dropna(subset=['timestamp'])
# Average repeated counts at the same location, hour and quarter; use complete hours only.
q=d.groupby(['timestamp','location','mm'],as_index=False).vol.mean()
h=q.groupby(['timestamp','location'],as_index=False).agg(traffic_volume=('vol','sum'),quarters=('mm','nunique'))
h=h[h.quarters==4].drop(columns='quarters').sort_values(['timestamp','location'])
if h.timestamp.dt.normalize().nunique()<10: raise ValueError('Export a broader time range with at least 10 complete dates.')
h.to_csv(sys.argv[2],index=False)
print('Saved',len(h),'complete location-hours across',h.location.nunique(),'locations.')
