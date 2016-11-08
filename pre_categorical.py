import os

from addons import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import multiprocessing

work_dir = os.getcwd()

# Clean and transform the raw data, creating train and test sets
target, df_train = pre_process_data(work_dir, is_train=True, reduce=False, impute=False, dummies=False)
df_test = pre_process_data(work_dir, is_train=False, reduce=False, impute=False, dummies=False)

# Keep categorical features only
categories = select_type(df_train, 'category')
df_cat = df_train.loc[:, categories]

# Encode ordinal
ordinal = ['X8', 'X9', 'X11', 'X12', 'X14']
for c in ordinal:
    s = df_cat[c]
    if c in ['X8', 'X9']:
        cats = sorted(list(set(s)))
        maps = {label: i for i, label in enumerate(cats)}
        df_cat[c] = df_cat[c].map(maps)
    elif c in ['X12']:
        maps = {'OWN': 7,
                'MORTGAGE': 6,
                'RENT': 5,
                'UNKNOWN': 4,
                'OTHER': 3,
                'NONE': 2,
                'ANY': 1
                }
        df_cat[c] = df_cat[c].map(maps)
    elif c in ['X14']:
        maps = {'VERIFIED - income': 3,
                'VERIFIED - income source': 2,
                'not verified': 1
        }
        df_cat[c] = df_cat[c].map(maps)
    elif c in ['X11']:
        df_cat.loc[:, c] = df_cat.loc[:, c].astype('int')

# Encode nominal features using k-to-n method
nominal = df_cat.columns.difference(ordinal)
encoded = pd.get_dummies(df_cat.loc[:, nominal])
df_cat = df_cat.drop(nominal, axis=1)
for c in encoded.columns:
    df_cat[c] = encoded[c]