# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import j24.visualization as vis
from os import path
from j24 import home
from sklearn import decomposition
from sklearn.cluster import KMeans
import sonde

datadir = path.join(home(), 'DATA', 'pangaea', 'sonde')
fname = 'NYA_UAS_2016.tab'
fpath = path.join(datadir, fname)

data = sonde.read(fpath)
data = sonde.prepare(data)
times = sonde.launch_times(data)

km = KMeans(init='k-means++', n_clusters=4, n_init=40, n_jobs=-1)

t = sonde.select_var(data, 't')
tt = sonde.select_var(data, 't', resample=False).ffill()

fig, ax = vis.heatmap(t, cmap='jet')
vis.fmt_axis_date(ax.xaxis)
ax.set_ylabel('Height, m')
ax.set_xlabel('Time')
ax.set_title('Temperature soundings from 2016')

km.fit(tt.T)
cen = pd.DataFrame(km.cluster_centers_.T, index=tt.index)
cfig, cax = vis.heatmap(cen)
vis.fmt_axis_str(cax.xaxis, locations=cen.columns.values, fmt='{x:.0f}')