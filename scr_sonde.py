# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import j24.visualization as vis
from os import path
from j24 import home, learn
#from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from scipy.spatial.distance import cdist
import sonde


def heatmap_t(t, classes=None, **kws):
    fig, ax = sonde.heatmap(t, cmap='jet')
    vis.fmt_axis_date(ax.xaxis)
    ax.set_xlabel('Time')
    ax.set_title('Temperature soundings from 2016')
    if classes is not None:
        vis.class_colors(classes, ax=ax)
    return fig, ax


def heatmap_cluster_cen(cen):
    fig, ax = sonde.heatmap(cen, cmap='jet')
    vis.fmt_axis_str(ax.xaxis, locations=cen.columns.values, fmt='{x:.0f}')
    vis.class_colors(cen.columns.values, ax=ax)
    return fig, ax


def t4clus(data):
    return sonde.select_var(data, 't', resample=False).ffill().bfill()


def wind_comp4clus(data, col='wx'):
    wd = sonde.select_var(data, col, resample=False)
    wd = wd.loc[wd.index > 151]
    return wd.ffill().bfill()


def wind4clus(data):
    wx = wind_comp4clus(data, col='wx')
    wy = wind_comp4clus(data, col='wy')
    return pd.concat((wx, wy))


def sep_xy(xy_series, cols=('x', 'y')):
    isplit = int(xy_series.size/2)
    x = xy_series.iloc[:isplit].copy()
    x.name = cols[0]
    y = xy_series.iloc[isplit:].copy()
    y.name = cols[1]
    return pd.concat((x, y), axis=1)


def plot_wprofile(prof, ax=None, **kws):
    x = prof['wx']
    y = prof['wy']
    z = prof.index.values
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.plot(x, y, z, **kws)
    sonde.fmt_m2km(ax.zaxis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Height, km')
    return ax


def plot_w_centroids(cen, ax=None):
    for i in cen.columns:
        prof = sep_xy(cen.loc[:,i], cols=('wx', 'wy'))
        ax = plot_wprofile(prof, ax=ax, color=vis.class_color(i))
    plot_wprofile(prof*0, ax=ax, color='black')
    return ax


def plot_scores(data, ns=[2,3,4,5,6,7,8,9,10,11],
                score_func=silhouette_score, ax=None):
    ax = ax or plt.gca()
    scores = []
    for n in ns:
        km = KMeans(init='k-means++', n_clusters=n, n_init=40, n_jobs=-1)
        classes = learn.fit_predict(data, km)
        scores.append(score_func(data.T, classes))
    ax.plot(ns, scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Number of clusters')
    return ax


plt.ion()
plt.close('all')

datadir = path.join(home(), 'DATA', 'pangaea', 'sonde')
fname = 'NYA_UAS_2016.tab'
fpath = path.join(datadir, fname)
data = sonde.read(fpath)
data = sonde.prepare(data)
times = sonde.launch_times(data)

ww = wind4clus(data)
km = KMeans(init='k-means++', n_clusters=6, n_init=40, n_jobs=-1)
#tt = t4clus(data)
#t = sonde.select_var(data, 't')
classes = learn.fit_predict(ww, km)
cen = learn.centroids(ww, km)
plot_scores(ww)


