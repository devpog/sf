import os
import re

import pandas as pd
import numpy as np

work_dir = os.getcwd()

columns = ['X2', 'X3', 'X15', 'X23', 'X32']
df_train = pre_process_data(work_dir, is_train=True, transform=False).loc[:, columns]
df_test = pre_process_data(work_dir, is_train=False, transform=False).loc[:, columns]

to_dates = ['X15', 'X23']
#df = df_test.copy()
df = df_test[df_test.X23 == 'Dec-68'].copy()


def year_trans(x):
    if bool(re.search("[A-Za-z]-\d+", x)):
        year = int(re.sub('\D+', '', re.search("[A-Za-z]?-(\\d+)", x).group()))
        if year < 16:
            res = pd.to_datetime(x, format='%b-%y')
        else:
            res = pd.to_datetime(x, format='%b-%y') - pd.DateOffset(years=100)
        return res
    elif bool(re.search("\d+-[A-Za-z]", x)):
        res = pd.to_datetime(x, dayfirst=True, format="%d-%b") + pd.DateOffset(years=114)
        return res

for c in to_dates:
    if c in ['X15']:
        df.loc[:, c] = df.loc[:, c].apply(lambda x: pd.to_datetime(x, format='%y-%b'))
    elif c in ['X23']:
        df.loc[:, c] = df.loc[:, c].apply(lambda x: year_trans(x))

df.loc[:, 'X33'] = [int(i.days) for i in (df.X15 - df.X23)]