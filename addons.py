def year_trans(x):
    """
    This method is used to transform a string object into datetime format,
        depending on the input, either 'month-year', or 'day-month'
    """
    import re
    import pandas as pd

    if bool(re.search("[A-Za-z]-\d+", x)):
        year = int(re.sub('\D+', '', re.search("[A-Za-z]?-(\\d+)", x).group()))
        if year < 16:
            res = pd.to_datetime(x, format='%b-%y')
        else:
            res = pd.to_datetime(x, format='%b-%y') - pd.DateOffset(years=100)
        return res
    elif bool(re.search("\d+-[A-Za-z]", x)):
        return pd.to_datetime(x, dayfirst=True, format="%d-%b") + pd.DateOffset(years=114)


def scale_features(df):
    """
    MinMax Scaler for numeric variables
    """
    import pandas as pd

    from sklearn.preprocessing import scale, OneHotEncoder, FunctionTransformer, MinMaxScaler
    scaler = MinMaxScaler()

    types = df.dtypes
    columns = [ind for ind in types.index if (str(types[ind]) == 'int64' or str(types[ind]) == 'float64')]
    df = pd.DataFrame(scaler.fit_transform(df), columns=columns)

    return df


def cast_into(x, new_type):
    """
    Method for casting columns into specified type
    """
    import pandas as pd
    import numpy as np

    ind = pd.isnull(x).values
    if ind.any():
        ind = np.invert(ind)
        x.iloc[ind] = x.iloc[ind].astype(new_type)
    return x


def nan_share(data):
    """
    Method to calculate the share of missing/NaNs values
    """
    import pandas as pd

    for column in data.columns:
        ps = pd.isnull(data[column])
        print("Column: {}\nCounts: {}\n\n".format(column, ps.value_counts()))


def nan_share_pcnt(data):
    """
    Method to calculate the share of missing/NaNs values, displayed in percentage
    """
    import pandas as pd
    rows = len(data)
    nans = dict()
    for column in data.columns:
        ps = pd.Series(pd.isnull(data[column]).values).value_counts()
        try:
            if len(ps) > 1:
                print("{}\t{}%".format(column, ps[1]/rows))
                nans[column] = data[column]
        except IndexError as err:
            print(err)
    return nans


def reduce_nan(data, columns = ['X1']):
    """
    Method to reduce the rows with missing values
    """
    import pandas as pd
    import numpy as np

    for column in columns:
        ind = pd.isnull(data[column]).values
        data = data.iloc[np.invert(ind)]

    return data


def select_type(df, type='category', columns_only=True):
    """
    Method to select columns with specified data type
    """
    types = df.dtypes
    select = [ind for ind in types.index if str(types[ind]) == type]
    if columns_only:
        return select
    else:
        return df[select]



def pre_process_data(work_dir, is_train=True, transform=True, reduce=True, drop=True, dummies=True, impute=True):
    """
    Method to read and pre-process data by dropping unnecessary features and filling in missing values
    """
    import os
    import re

    import pandas as pd
    import numpy as np

    from sklearn.feature_extraction import DictVectorizer

    if is_train:
        data_set = 'train'
    else:
        data_set = 'test'

    print("Reading {} data set, dropping unnecessary features...".format(data_set))
    # Read in train and test data sets, both in raw forms
    for root, dirs, files in os.walk(work_dir):
        for d in dirs:
            if re.match(r'input', d):
                input_dir = os.path.join(root, d)
        for f in files:
            if re.match(r'^data', f):
                data_file = os.path.join(input_dir, f)
                data_raw = pd.read_csv(data_file, low_memory=False)
                #if not transform: return data_raw
            elif re.match(r'^test', f):
                test_file = os.path.join(input_dir, f)
                test_raw = pd.read_csv(test_file, low_memory=False)
                #if not transform: return test_raw
    if is_train:
        if not transform: return data_raw
    else:
        if not transform: return test_raw

    # Remove a row with all NaN's casting IDs columns into integers
    if is_train:
        broken_ind = 364111
        df = data_raw.drop(data_raw.index[broken_ind])
    else:
        df = test_raw

    to_drop = ['X2', 'X3', 'X10', 'X16', 'X18']
    print("Dropping {}...".format(' '.join(to_drop)))
    df = df.drop(to_drop, axis = 1)

    """
    # Cast loan ID and borrower ID into integers
    # (X2, X3)
    to_integer = ['X2', 'X3']
    print("Casting {} into integer...".format(' '.join(to_integer)))
    df.loc[:, to_integer] = df.loc[:, to_integer].apply(lambda x: x.astype(int))
    if test_only: return df
    """

    # Cast a loan's rate and an applicant's revolving utilization rate into decimal representation
    # (X1, X30)
    to_decimal = ['X1', 'X30']
    print("Casting {} into decimals...".format(' '.join(to_decimal)))
    if is_train:
        df.loc[:, to_decimal] = df.loc[:, to_decimal].apply(lambda x: x.str.replace('%', '').astype(float) * .01)
    else:
        df.loc[:, ['X1']] = 0
        df.loc[:, ['X30']] = df.loc[:, ['X30']].apply(lambda x: x.str.replace('%', '').astype(float) * .01)

    # Cast a loan's amounts funded into decimal representation
    to_decimal = ['X4', 'X5', 'X6']
    print("Casting {} into decimals...".format(' '.join(to_decimal)))
    df.loc[:, to_decimal] = df.loc[:, to_decimal].apply(lambda x: x.str.replace(r'\D+', '').astype('float'))

    # Cast number of months (X7) into categorical
    to_category = ['X7']
    print("Casting {} into category...".format(' '.join(to_category)))
    df.loc[:, to_category] = df.loc[:, to_category].apply(lambda x: x.str.replace(r'\D+', '').astype('category'))

    # Format a categorical variable the number of years employed (X11)
    # Cast the number of years a categorical var
    print("Casting X11 into a special categorical...")
    df.loc[:, 'X11'] = df.loc[:, 'X11'].str.replace(r'(years)|(year)', '') \
        .str.replace(r'(\<\s+1)', '0') \
        .str.replace(r'(\d{2}\+)', '11') \
        .str.replace('n/a', '-1') \
        .astype('category')

    # Set borrower's income (X13) to 0 if missing
    print("Set X13 to 0 if missing...")
    df.loc[:, 'X13'] = df.loc[:, 'X13'].apply(lambda x: 0 if pd.isnull(x) else x)

    # Cast into categories payments, loan grade and sub-grade, number of years employed, home ownership, income verification, loan category, zip, state, and category
    # (X8, X9, X14, X17, X19, X20, X32)
    to_category = ['X8', 'X9', 'X12', 'X14', 'X17', 'X19', 'X20', 'X32']
    print("Casting {} into categorical...".format(' '.join(to_category)))
    df.loc[:, to_category] = df.loc[:, to_category].apply(lambda x: x.astype('category'))

    # Set home ownership to other if nan
    print("Set X12 to UNKNOWN if missing...")
    x12 = pd.Series(df['X12']).cat.add_categories('UNKNOWN')
    df.loc[:, 'X12'] = x12.fillna('UNKNOWN').astype('category')

    # Impute NaNs in load grade (X8)
    print("Set X8 to Z if missing...")
    x8 = pd.Series(df['X8']).cat.add_categories('Z')
    df.loc[:, 'X8'] = x8.fillna('Z').astype('category')

    # Impute NaNs in load sub-grade (X9)
    print("Set X9 to Z99 if missing...")
    x9 = pd.Series(df['X9']).cat.add_categories('Z99')
    df.loc[:, 'X9'] = x9.fillna('Z99').astype('category')

    if impute:
        # Impute NaNs in X25
        print("Interpolate X25 if missing...")
        x25 = df.loc[pd.isnull(df['X25']).values, 'X25'] = np.nan
        x25 = pd.Series(df['X25']).interpolate(method='nearest')
        df.loc[:, 'X25'] = x25

        # Impute NaNs in X26
        print("Interpolate X26 if missing...")
        x26 = pd.Series(df['X26']).interpolate(method='nearest')
        df.loc[:, 'X26'] = x26

        # Impute NaNs in X30
        print("Interpolate X30 if missing...")
        max_ind = df.loc[df['X30'] > 1, 'X30'].index
        # df = df.drop(df.index[max_ind])
        df.loc[max_ind, 'X30'] = np.nan
        x30 = pd.Series(df['X30']).interpolate(method='nearest')
        df.loc[:, 'X30'] = x30


    # Cast into integers 30+ delinquencies, inqueries, months since last delinquency, months since the last public record,
    # credit lines, derogatory public records, total credit lines
    # X22, X24, X25, X26, X27, X28, X31
    to_integer = ['X22', 'X27', 'X28', 'X31']
    print("Casting {} into integer...".format(' '.join(to_integer)))
    df.loc[:, to_integer] = df.loc[:, to_integer].apply(lambda x: x.astype(int))

    """
    Cast date issued and the earliest credit line reported, X15, X23, in order to create a new variable X33,
    which signifies the length of a borrower's account in days.
    """
    to_dates = ['X15', 'X23']
    print("Casting {} into dates...".format(' '.join(to_dates)))
    if is_train:
        df.loc[:, to_dates] = df.loc[:, to_dates].apply(lambda x: pd.to_datetime(x, format='%b-%y'))
    else:
        for c in to_dates:
            if c in ['X15']:
                df.loc[:, c] = df.loc[:, c].apply(lambda x: pd.to_datetime(x, format='%y-%b'))
            elif c in ['X23']:
                df.loc[:, c] = df.loc[:, c].apply(lambda x: year_trans(x))

    df.loc[:, 'X33'] = [int(i.days) for i in (df.X15 - df.X23)]
    to_drop = ['X15', 'X23']
    print("Dropping {}...".format(' '.join(to_drop)))
    df = df.drop(to_drop, axis=1)

    # Drop rows with any empty values
    if reduce:
        to_reduce = []
        print("Reducing data set by excluding all rows with empty columns...")
        df = reduce_nan(df, to_reduce)

    """
    # Drop unnecessary features X15, X23
    if drop:
        to_drop = ['X15', 'X23']
        print("Dropping {}...".format(' '.join(to_drop)))
        df = df.drop(to_drop, axis=1)
    """

    # Encode categorical features into dummy variables
    target_column = df.loc[:, 'X1'].values
    if dummies:
        cats = select_type(df)
        # print("Encoding categorical features {}...".format(' '.join(cats)))
        # Remove all categorical features
        df_other = df.drop(cats, axis=1)
        df_cat = df[cats]
        df_dict = df_cat.T.to_dict().values()
        dv = DictVectorizer(sparse=False)
        df_encoded = dv.fit_transform(df_dict)
        feature_names = dv.get_feature_names()

        # Expand the set with dummy variables created in the previous step
        df = pd.DataFrame(df_encoded, columns=feature_names, index=df_other.index)

        # Merge back left over in the previous step with newly created dummy featues
        for c in df_other.columns:
            df[c] = df_other[c]
        df['X1'] = target_column

    if is_train:
        df = reduce_nan(df)
        target = df.loc[:, 'X1'].values
        train_set = df.drop('X1', axis=1)
        return target, train_set
    else:
        test_set = df.drop('X1', axis=1)
        return test_set


def transform_numeric(df):
    """
    Method for transforming continuous features by removing erroneous data points
    and replacing them with the result of interpolation, if needed.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import describe
    from scipy.stats import boxcox

    # Select numeric and continuous features only
    print("Attempting to transform numeric features...")
    int64 = select_type(df, 'int64')
    float64 = select_type(df, 'float64')
    numeric = [l for l1 in [int64, float64] for l in l1]
    df_num = df.loc[:, numeric]
    df_other = df.loc[:, df.columns.difference(numeric)]

    # Temp dropping due to a very skewed distribution
    # to_drop = ['X24']
    # df_num = df_num.drop(to_drop, axis=1)

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

    # Analysis of skewness and kurtosis, decide whether or not the distribution is skewed and its direction,
    # either left, right, or normal
    stats = dict()
    right_skewed = []
    left_skewed = []
    normal = []
    print("Find skewed distros...")
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

    """
    # Scale symmetric features using StandardScaler
    normal = ['']
    for c in normal:
        s = df_num[c]
        scaler = StandardScaler()
        s = scaler.fit_transform(s.reshape(-1, 1))
        df_num.loc[:, c] = s
    """

    # Transform skewed features using various methods (see below):
    # temp comment: skewed = [l for l1 in [left_skewed, right_skewed] for l in l1]
    quart_root = ['X31']
    sqr_root = ['X6', 'X30']
    # box_cox = ['X4', 'X5', 'X21', 'X27', 'X30']
    box_cox = ['X4', 'X5', 'X21', 'X27']
    nat_log = ['X13', 'X22', 'X24', 'X28', 'X29']
    for c in df_num.columns:
        # s = df_num[c]
        if c in quart_root:
            print("Transforming {} by application of QUART_ROOT".format(c))
            s = df_num[c]
            s_trans = np.power(s+1, .25)
            df_num[c] = s_trans
        elif c in sqr_root:
            print("Transforming {} by application of SQRT".format(c))
            s = df_num[c]
            s_trans = np.sqrt(s)
            df_num[c] = s_trans
        elif c in box_cox:
            print("Transforming {} by application of BOXCOX method".format(c))
            s = df_num[c]
            s_trans = boxcox(s+1)[0]
            df_num[c] = s_trans
        elif c in nat_log:
            print("Transforming {} by application of LN".format(c))
            s = df_num[c]
            s_trans = np.log1p(s)
            df_num[c] = s_trans

    for c in df_other.columns:
        df_num[c] = df_other[c]

    return df_num


def transform_categorical(df):
    """
    Method for transforming categorical features by hot-encoding nominal and labeling ordinal
    """
    import pandas as pd

    # Keep categorical features only
    print("Attempting to transform categorical features...")
    categories = select_type(df, 'category')
    df_cat = df.loc[:, categories]
    df_other = df.loc[:, df.columns.difference(categories)]

    # Encode ordinal
    ordinal = ['X8', 'X9', 'X11', 'X12', 'X14', 'X19']
    for c in ordinal:
        s = df_cat[c]
        if c in ['X8', 'X9', 'X19']:
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

    for c in df_other.columns:
        df_cat[c] = df_other[c]

    return(df_cat)