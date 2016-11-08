import os

from addons import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import skewtest
from scipy.stats import describe
from scipy.stats import boxcox

import multiprocessing

work_dir = os.getcwd()

# Clean and transform the raw data, creating train and test sets
target, df_train = pre_process_data(work_dir, is_train=True, reduce=False, impute=False, dummies=False)
df_test = pre_process_data(work_dir, is_train=False, reduce=False, impute=False, dummies=False)

# Select numeric and continuous features only
int64 = select_type(df_train, 'int64')
float64 = select_type(df_train, 'float64')
numeric = [l for l1 in [int64, float64] for l in l1]
df_num = df_train.loc[:, numeric]

# Temp dropping due to a very skewed distribution
to_drop = ['X24']
df_num = df_num.drop(to_drop, axis=1)

# Analysis of features with missing values, when found decide to either drop
# or interpolate, depending on the ratio
missing_cutoff = .30
for c in df_num.columns:
    s = df_num[c]
    stf = pd.isnull(s).value_counts()
    if True in stf.index:
        stf_share = stf[True] / len(s)
        # If a share of missing values > allowed threshold drop, otherwise interpolate
        if stf_share >= missing_cutoff:
            df_num = df_num.drop(c, axis=1)
        else:
            if c in ['X30']:
                max_ind = df_num.loc[df_num[c] > 1, 'X30'].index
                df_num.loc[max_ind, c] = np.nan
                x30 = pd.Series(df_num[c]).interpolate(method='nearest')
                df_num.loc[:, c] = x30
df_org = df_num.copy()

# Analysis of skewness and kurtosis, decide whether or not the distribution is skewed and its direction,
# either left, right, or normal
stats = dict()
right_skewed = []
left_skewed = []
normal = []
for c in df_num.columns:
    s = df_num[c]
    scaler = StandardScaler()
    s = scaler.fit_transform(s.reshape(-1, 1))
    s_stats = describe(s)
    stats[c] = s_stats
    # df_num.loc[:, c] = s
    if s_stats.skewness > 0:
        if abs(s_stats.skewness) < 1:
            normal.append(c)
        elif abs(s_stats.skewness) > 1:
            right_skewed.append(c)
    elif s_stats.skewness < 0:
        if abs(s_stats.skewness) < 1:
            normal.append(c)
        elif abs(s_stats.skewness) > 1:
            left_skewed.append(c)

# Scale symmetric features using StandardScaler
for c in normal:
    s = df_num[c]
    scaler = StandardScaler()
    s = scaler.fit_transform(s.reshape(-1, 1))
    df_num.loc[:, c] = s

# Transform skewed features using various methods (see below):
skewed = [l for l1 in [left_skewed, right_skewed] for l in l1]
quart_root = ['X31']
sqr_root = ['X6']
box_cox = ['X4', 'X5', 'X21', 'X27', 'X30']
nat_log = ['X13', 'X22', 'X28', 'X29']
for c in df_num.columns:
    s = df_num[c]
    if c in quart_root:
        print("Transforming {} by application of QUART_ROOT".format(c))
        s_trans = np.power(s, .25)
        s_stats = describe(s_trans)
        df_num[c] = s_trans
        print("{}:\n{}".format(c, s_stats))
    elif c in sqr_root:
        print("Transforming {} by application of SQRT".format(c))
        s_trans = np.sqrt(s)
        s_stats = describe(s_trans)
        df_num[c] = s_trans
        print("{}:\n{}".format(c, s_stats))
    elif c in box_cox:
        print("Transforming {} by application of BOXCOX method".format(c))
        s_trans = boxcox(s+1)[0]
        s_stats = describe(s_trans)
        df_num[c] = s_trans
        print("{}:\n{}".format(c, s_stats))
    elif c in nat_log:
        print("Transforming {} by application of LN".format(c))
        s_trans = np.log1p(s)
        s_stats = describe(s_trans)
        df_num[c] = s_trans
        print("{}:\n{}".format(c, s_stats))