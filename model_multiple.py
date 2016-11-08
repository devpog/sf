import os

from addons_sep import *

import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import explained_variance_score

import multiprocessing

work_dir = os.getcwd()
num_cpu = multiprocessing.cpu_count()

# Clean and transform the raw data, creating train and test sets
target, df_train = pre_process_data(work_dir, is_train=True, reduce=False, dummies=True)
df_test = pre_process_data(work_dir, is_train=False, reduce=False, dummies=True)

# Add missing features as a result of creating dummy vars while encoding categotical features
extra_columns = df_train.columns.difference(df_test.columns)
for e in extra_columns:
    df_test[e] = 0

# Apply MinMax scaler to normalize features
mms = MinMaxScaler()
X_train = mms.fit_transform(df_train)
X_test = mms.fit_transform(df_test)

l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
model = ElasticNetCV(l1_ratio=l1_ratio, eps=0.001, n_alphas=100, alphas=None,
                   normalize=False, precompute='auto', max_iter=1000, tol=0.0001, cv=10,
                   copy_X=True, verbose=1, n_jobs=num_cpu, positive=False, random_state=None,
                   selection='cyclic')
model.fit(X_train, target)


# Fit the model
cross = model.predict(X_train)
predicted = model.predict(X_test)
explained_variance_score(target, predicted)


# Predicting unseen data and saving the result
np.savetxt('result_RandomForest.csv', predicted)
