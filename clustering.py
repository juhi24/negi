# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import j24.visualization as vis
import sonde
#from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score#, calinski_harabaz_score
#from scipy.spatial.distance import cdist
from j24 import learn


def heatmap_t(t, **kws):
    fig, ax = sonde.heatmap(t, cmap='jet', **kws)
    vis.fmt_axis_date(ax.xaxis)
    ax.set_xlabel('Time')
    ax.set_title('Temperature soundings from 2016')
    return fig, ax


def heatmap_cluster_cen(cen, cmap='jet', **kws):
    fig, ax = sonde.heatmap(cen, cmap=cmap)
    vis.fmt_axis_str(ax.xaxis, locations=cen.columns.values, fmt='{x:.0f}')
    vis.class_colors(cen.columns.values, ax=ax, cmap=sonde.DEFAULT_DISCRETE_CMAP)
    return fig, ax


def t4clus(data, col='t'):
    return sonde.select_var(data, col, resample=False).ffill().bfill()


def wind_comp4clus(data, col='u'):
    wd = sonde.select_var(data, col, resample=False)
    wd = wd.loc[wd.index > 151]
    return wd.ffill().bfill()


def wind4clus(data):
    wx = wind_comp4clus(data, col='u')
    wy = wind_comp4clus(data, col='v')
    return pd.concat((wx, wy))


def split_half(xy_series, cols=('u', 'v')):
    """Split a Series into DataFrame with two columns"""
    isplit = int(xy_series.size/2)
    x = xy_series.iloc[:isplit].copy()
    x.name = cols[0]
    y = xy_series.iloc[isplit:].copy()
    y.name = cols[1]
    return pd.concat((x, y), axis=1)


def plot_wprofile(prof, ax=None, **kws):
    x = prof['u']
    y = prof['v']
    z = prof.index.values
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.plot(x, y, z, **kws)
    sonde.fmt_m2km(ax.zaxis)
    ax.set_xlabel('$u$, ms$^{-1}$')
    ax.set_ylabel('$v$, ms$^{-1}$')
    ax.set_zlabel('Height, km')
    return ax


def plot_w_centroids(cen, linewidths=None, ax=None):
    if linewidths is None:
        linewidths = np.ones(cen.shape[1])
    for i in cen.columns:
        prof = split_half(cen.loc[:,i], cols=('u', 'v'))
        color = vis.class_color(i, cmap=sonde.DEFAULT_DISCRETE_CMAP)
        label = 'Class {}'.format(i)
        ax = plot_wprofile(prof, ax=ax, color=color, linewidth=linewidths[i], label=label)
    plot_wprofile(prof*0, ax=ax, color='black')
    #ax.legend()
    return ax


def scores(data, n_clus=[2,3,4,5,6,7,8,9,10,11],
                score_func=silhouette_score, ax=None):
    scores = []
    for n in n_clus:
        km = KMeans(init='k-means++', n_clusters=n, n_init=10, n_jobs=-1)
        classes = learn.fit_predict(data, km)
        scores.append(score_func(data.T, classes))
    return scores


def plot_scores(ns, scores, ax=None):
    ax = ax or plt.gca()
    ax.plot(ns, scores)
    ax.set_ylabel('Silhouette score')
    ax.set_xlabel('Number of clusters')
    return ax


def concat(dfs):
    sizes = [kk.shape[0] for kk in dfs]
    isplit = np.array(sizes).cumsum()
    catenated = pd.concat(dfs).dropna(axis=1)
    return catenated, isplit


def split(df, isplit):
    out = []
    prev = 0
    for i in isplit:
        out.append(df.iloc[prev:i])
        prev = i
    return out


def plot_profiles(ct, ax=None, linewidths=None):
    if linewidths is None:
        linewidths = np.ones(ct.shape[1])
    ax = ax or plt.gca()
    for i in range(ct.shape[1]):
        color = vis.class_color(i, cmap=sonde.DEFAULT_DISCRETE_CMAP)
        label = 'Class {}'.format(i)
        ax.plot(ct.iloc[:,i], ct.index, color=color, label=label, linewidth=linewidths[i])
    sonde.fmt_m2km(ax.yaxis)
    ax.set_ylabel('Height, km')
    ax.legend()
    return ax



