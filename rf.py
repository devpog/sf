import os
import multiprocessing

from addons import *

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV

from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score

work_dir = os.getcwd()
num_cpu = multiprocessing.cpu_count()

# Clean and transform the raw data, creating train and test sets
target, df_train = pre_process_data(work_dir, is_train=True, reduce=False, impute=False, dummies=False)
df_test = pre_process_data(work_dir, is_train=False, reduce=False, impute=False, dummies=False)

# Transform features in train set
df_train_tr = transform_numeric(df_train)
df_train_tr = transform_categorical(df_train_tr)

# Transform features in test set
df_test_tr = transform_numeric(df_test)
df_test_tr = transform_categorical(df_test_tr)

# Add missing features as a result of creating dummy vars while encoding categorical features
for e in df_train_tr.columns.difference(df_test_tr.columns):
    df_test_tr[e] = 0

# Apply ElasticNet with 5-fold CV and PCA
l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
model_en = ElasticNetCV(l1_ratio=l1_ratios, precompute=True, max_iter=num_cpu*100, n_jobs=num_cpu, cv=5, verbose=2).fit(df_train_tr, target)
predicted_en = model_en.predict(df_test_tr)
pd.DataFrame(predicted_en, columns = ['X1_predicted']).to_csv('result_en_Kyrylo_Pogrebenko.csv', index=False)


# Apply Random Forest Regressor with 5-fold CV
model_rf = RandomForestRegressor(n_estimators=num_cpu*10, criterion='mse',
                              max_features='log2', n_jobs=num_cpu, verbose=2).fit(df_train_tr, target)
scores_rf = cross_val_score(model_rf, df_train_tr, target, scoring='r2', cv=5, n_jobs=num_cpu, verbose=2)
print('Average score for RF: {}'.format(np.mean(scores_rf)))
predicted_rf = model_rf.predict(df_test_tr)
pd.DataFrame(predicted_rf, columns = ['X1_predicted']).to_csv('result_rf_Kyrylo_Pogrebenko.csv', index=False)



