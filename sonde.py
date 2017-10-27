# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import datetime
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from j24 import home
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.dates as mdates


plt.ion()
plt.close('all')


def launch_times(data):
    g = data.groupby('id')
    return g['time'].first().sort_values()

def heatmap(t, xlocator=None, datefmt='%b', **kws):
    date_formatter = mdates.DateFormatter(datefmt)
    xloc = xlocator or mdates.MonthLocator()
    mesh = plt.pcolormesh(t.columns, t.index, t, **kws)
    ax = mesh.axes
    fig = ax.get_figure()
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(date_formatter)
    plt.colorbar()
    return fig, ax


datadir = path.join(home(), 'DATA', 'pangaea', 'sonde')
fname = 'NYA_UAS_2016.tab'
fpath = path.join(datadir, fname)

cols = ['time', 'id', 'h', 'lat', 'lon', 'p', 't', 'rh', 'ws', 'wd']
data = pd.read_csv(fpath, sep='\t', skiprows=24, parse_dates=['time'], names=cols)
data = data[data['h'] < 11001]
data = data[data['h'] > 40]
data['date'] = data['time'].apply(lambda tt: tt.date())

start_times = launch_times(data)
id_selection = start_times.apply(lambda tt: (tt.time()>datetime.time(10)) & (tt.time()<datetime.time(12)))
selected_ids = start_times[id_selection].index
data = data[data['id'].isin(selected_ids)]
times = launch_times(data)

km = KMeans(init='k-means++', n_clusters=10, n_init=40, n_jobs=-1)

t = data.pivot_table(values='t', columns='date', index='h').sort_index(ascending=False)
t.columns=pd.DatetimeIndex(t.columns)
t = t.T.resample('1D').asfreq().T

fig, ax = heatmap(t, cmap='jet')
ax.set_ylabel('Height, m')
ax.set_xlabel('Time')
ax.set_title('Temperature soundings from 2016')
#fig.autofmt_xdate()