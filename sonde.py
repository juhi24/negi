# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import j24.visualization as vis
from j24 import math


def heatmap(*args, **kws):
    """j24.visualization.heatmap wrapper for sounding data"""
    fig, ax = vis.heatmap(*args, **kws)
    fmt_m2km(ax.yaxis)
    ax.set_ylabel('Height, km')
    return fig, ax


def m2km(m, pos):
    '''formatting m in km'''
    return '{:.0f}'.format(m*1e-3)


def fmt_m2km(axis):
    axis.set_major_formatter(mticker.FuncFormatter(m2km))


def launch_times(data):
    """List launch times by sounding id."""
    g = data.groupby('id')
    return g['time'].first().sort_values()


def read(file_path, **kws):
    """Read tab separated sounding data."""
    cols = ['time', 'id', 'h', 'lat', 'lon', 'p', 't', 'rh', 'ws', 'wd']
    return pd.read_csv(file_path, sep='\t', skiprows=24, parse_dates=['time'],
                       names=cols, **kws)


def between_time(data, hour_start=10, hour_end=12):
    """Select soundings that start between selected hours of day."""
    out = data.copy()
    start_times = launch_times(out)
    h_start = datetime.time(hour_start)
    h_end = datetime.time(hour_end)
    check_between = lambda tt: (tt.time()>h_start) & (tt.time()<h_end)
    id_selection = start_times.apply(check_between)
    selected_ids = start_times[id_selection].index
    return out[out['id'].isin(selected_ids)]


def between_altitude(data, h_start=50, h_end=11000):
    """inclusive altitude range"""
    out = data.copy()
    out = out[out['h'] < h_end+1]
    return out[out['h'] > h_start-1]


def select_var(data, var, columns='date', resample=True):
    """Select one variable from each sounding."""
    t = data.pivot_table(values=var, columns=columns, index='h')
    t.sort_index(ascending=False, inplace=True)
    if columns=='date':
        t.columns = pd.DatetimeIndex(t.columns)
        if resample:
            t = t.T.resample('1D').asfreq().T
    return t


def prepare(data):
    """Apply default altitude and time of day ranges and add date column."""
    out = data.copy()
    out = between_altitude(out)
    out['date'] = out['time'].apply(lambda tt: tt.date())
    out = between_time(out)
    xy = math.pol2cart_df(out.ws, np.deg2rad(out.wd), cols=('wx', 'wy'))
    return pd.concat((out, xy), axis=1)



