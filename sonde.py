# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
from os import path
from j24 import home

datadir = path.join(home(), 'DATA', 'pangaea', 'sonde')
fname = 'NYA_UAS_2016.tab'
fpath = path.join(datadir, fname)

cols = ['time', 'id', 'h', 'lat', 'lon', 'p', 't', 'rh', 'ws', 'wd']
data = pd.read_csv(fpath, sep='\t', skiprows=24, parse_dates=['time'], names=cols,
                   index_col=['id', 'h'])

