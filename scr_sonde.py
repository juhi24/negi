# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
"""script for playing around with sounding clustering"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustering as cl
import sonde
from os import path
from sklearn.cluster import KMeans
from j24 import home, learn
import datetime


np.random.seed(42)
plt.ion()
plt.close('all')

datadir = path.join(home(), 'DATA', 'pangaea', 'sonde')
storage_path = path.join(datadir, '96-16prep.h5')
#data = sonde.read_all_data(datadir)
#data.to_hdf(storage_path, 'data')
data = pd.read_hdf(storage_path, 'data')
#times = sonde.launch_times(data)

ww = cl.wind4clus(data)
tt = cl.t4clus(data, col='t')
hh = cl.t4clus(data, col='rh')
clus_vars = (ww, tt, hh)
km = KMeans(init='k-means++', n_clusters=6, n_init=40, n_jobs=-1)
wtr, isplit = cl.concat(clus_vars)
classes = learn.fit_predict(wtr, km)
cen = learn.centroids(wtr, km)
cw, ct, ch = cl.split(cen, isplit)

start = datetime.datetime(2015, 12, 31)
end = datetime.datetime(2017, 1, 1)
t = sonde.resample_transpose(tt)
selection = (t.columns<end) & (t.columns>start)
t = t.loc[:,selection]
#h = sonde.resample_transpose(hh)
cla = classes.resample('1D').asfreq()
sel = (cla.index<end) & (cla.index>start)
cla = cla.loc[sel]

sonde.heatmap(t, classes=cla)
#sonde.heatmap(h, classes=cla)

# wy: positive: wind from west
# wx: positive: wind from south

