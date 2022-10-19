import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import json
from tqdm import tqdm
import gc

# !
TYPE_TO_PLOT = '117'

DATA_PATH = 'prices_clean.csv'
TYPES_PATH = 'types_dict.json'
CV_RESULTS_PATH = 'cv_resutls.csv'
MODEL_PATH = 'model.pickle'
N_PREDS = 4
N_FOLDS = 10
VAR_PARAMS = {
    'trend' : 'ct',
    'max_lags' : 2,
    'diff' : True
}

def read_data():
    # Что если нет файла
    try:
        data = pd.read_csv(DATA_PATH, index_col=[0], parse_dates=[0])
        with open(TYPES_PATH) as f:
            types_dict = json.load(f)
            data = data.dropna(axis=1)
    except FileNotFoundError as e:
        print(e)
        return

    # !!!!
    max_date, min_date = data.index.max(), data.index.min()
    data.index = pd.date_range(end=max_date, periods=data.shape[0], freq='7D')
    data = data.sort_index()
    return data

def fit(df_train, max_lags, trend='c', exogs_train=None, diff=False, fp=None):
    if diff:
        df = df_train.diff(1).iloc[1:, :]
    else:
        df = df_train.copy()

    model = sm.tsa.VAR(df, exog=exogs_train).fit(maxlags=max_lags, trend=trend)
    if fp:
        model.save(fp)
        print(f'Model is saved as {fp}')
    return model

def predict(model, n_preds, exogs_test=None):
    forecast = np.array(model.forecast_interval(model.endog[-model.k_ar:], n_preds, exog_future=exogs_test))
    fitted = model.fittedvalues.values
    return fitted, forecast

def restore_diff(df_train, fitted, forecast, max_lags):
    fitted = np.r_[df_train.values[[max_lags], :], fitted].cumsum(axis=0)
    forecast[0] = np.r_[df_train.values[-1:, :], forecast[0]].cumsum(axis=0)[1:]
    #lower
    forecast[1] = np.r_[df_train.values[-1:, :], forecast[1]].cumsum(axis=0)[1:]
    #upper
    forecast[2] = np.r_[df_train.values[-1:, :], forecast[2]].cumsum(axis=0)[1:]
    return fitted, forecast   

def cross_validate(df, cv, max_lags, n_preds, trend='c', exogs=None, diff=False):
    train_errors = []
    test_errors = []
    for train_index, test_index in tqdm(cv.split(df), total=cv.n_splits):
        train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]
        if exogs is None:
            exogs_train, exogs_test = None, None
        else:
            exogs_train, exogs_test = exogs.iloc[train_index, :], exogs.iloc[test_index, :]

        model = fit(train_data, max_lags, trend, diff=diff, exogs_train=exogs_train)
        fitted, forecast  =predict(model, n_preds, exogs_test=exogs_test)
        if diff:
            fitted, forecast = restore_diff(train_data, fitted, forecast, max_lags)
        train_errors = np.r_[
            train_errors,\
            100 * np.mean(np.abs(train_data.values[max_lags:] - fitted) / train_data.values[max_lags:], axis=0)
        ]
        test_errors = np.r_[
            test_errors,\
            100 * np.mean(np.abs(test_data.values - forecast[0]) / test_data.values, axis=0)
        ]
    shape = (cv.n_splits, df.shape[1])
    errors = pd.DataFrame(
        np.c_[
            train_errors.reshape(shape).mean(axis=0),\
            train_errors.reshape(shape).std(axis=0),\
            test_errors.reshape(shape).mean(axis=0),\
            test_errors.reshape(shape).std(axis=0)
        ],\
        columns=['cv_train_mean', 'cv_train_std', 'cv_test_mean', 'cv_test_std'],\
        index=df.columns
    )
    errors.loc['total'] = [
        train_errors.mean(),\
        train_errors.std(),\
        test_errors.mean(),\
        test_errors.std()
        ]
    del train_data, test_data, train_errors, test_errors
    gc.collect()
    return errors

def plot(ts_id, forecast, fitted, data):
    pass
    
def iu():
    pass

def main(refit=False):
    data = read_data()

    if refit:
        model = fit(data, fp=MODEL_PATH, **VAR_PARAMS)
    else:
        try:
            model = sm.load_pickle(MODEL_PATH)
        except FileNotFoundError as e:
            print(e)
            return None, None
    fitted, forecast = predict(model, N_PREDS)
    if VAR_PARAMS['diff']:
        fitted, forecast = restore_diff(data, fitted, forecast, VAR_PARAMS['max_lags'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import json
from tqdm import tqdm
import gc

# !
TYPE_TO_PLOT = '117'

DATA_PATH = 'prices_clean.csv'
TYPES_PATH = 'types_dict.json'
CV_RESULTS_PATH = 'cv_resutls.csv'
MODEL_PATH = 'model.pickle'
N_PREDS = 4
N_FOLDS = 10
VAR_PARAMS = {
    'trend' : 'ct',
    'max_lags' : 2,
    'diff' : True
}

def read_data():
    # Что если нет файла
    try:
        data = pd.read_csv(DATA_PATH, index_col=[0], parse_dates=[0])
        with open(TYPES_PATH) as f:
            types_dict = json.load(f)
            data = data.dropna(axis=1)
    except FileNotFoundError as e:
        print(e)
        return

    # !!!!
    max_date, min_date = data.index.max(), data.index.min()
    data.index = pd.date_range(end=max_date, periods=data.shape[0], freq='7D')
    data = data.sort_index()
    return data

def fit(df_train, max_lags, trend='c', exogs_train=None, diff=False, fp=None):
    if diff:
        df = df_train.diff(1).iloc[1:, :]
    else:
        df = df_train.copy()

    model = sm.tsa.VAR(df, exog=exogs_train).fit(maxlags=max_lags, trend=trend)
    if fp:
        model.save(fp)
        print(f'Model is saved as {fp}')
    return model

def predict(model, n_preds, exogs_test=None):
    forecast = np.array(model.forecast_interval(model.endog[-model.k_ar:], n_preds, exog_future=exogs_test))
    fitted = model.fittedvalues.values
    return fitted, forecast

def restore_diff(df_train, fitted, forecast, max_lags):
    fitted = np.r_[df_train.values[[max_lags], :], fitted].cumsum(axis=0)
    forecast[0] = np.r_[df_train.values[-1:, :], forecast[0]].cumsum(axis=0)[1:]
    #lower
    forecast[1] = np.r_[df_train.values[-1:, :], forecast[1]].cumsum(axis=0)[1:]
    #upper
    forecast[2] = np.r_[df_train.values[-1:, :], forecast[2]].cumsum(axis=0)[1:]
    return fitted, forecast   

def cross_validate(df, cv, max_lags, n_preds, trend='c', exogs=None, diff=False):
    train_errors = []
    test_errors = []
    for train_index, test_index in tqdm(cv.split(df), total=cv.n_splits):
        train_data, test_data = df.iloc[train_index, :], df.iloc[test_index, :]
        if exogs is None:
            exogs_train, exogs_test = None, None
        else:
            exogs_train, exogs_test = exogs.iloc[train_index, :], exogs.iloc[test_index, :]

        model = fit(train_data, max_lags, trend, diff=diff, exogs_train=exogs_train)
        fitted, forecast  =predict(model, n_preds, exogs_test=exogs_test)
        if diff:
            fitted, forecast = restore_diff(train_data, fitted, forecast, max_lags)
        train_errors = np.r_[
            train_errors,\
            100 * np.mean(np.abs(train_data.values[max_lags:] - fitted) / train_data.values[max_lags:], axis=0)
        ]
        test_errors = np.r_[
            test_errors,\
            100 * np.mean(np.abs(test_data.values - forecast[0]) / test_data.values, axis=0)
        ]
    shape = (cv.n_splits, df.shape[1])
    errors = pd.DataFrame(
        np.c_[
            train_errors.reshape(shape).mean(axis=0),\
            train_errors.reshape(shape).std(axis=0),\
            test_errors.reshape(shape).mean(axis=0),\
            test_errors.reshape(shape).std(axis=0)
        ],\
        columns=['cv_train_mean', 'cv_train_std', 'cv_test_mean', 'cv_test_std'],\
        index=df.columns
    )
    errors.loc['total'] = [
        train_errors.mean(),\
        train_errors.std(),\
        test_errors.mean(),\
        test_errors.std()
        ]
    del train_data, test_data, train_errors, test_errors
    gc.collect()
    return errors

def plot(ts_id, forecast, fitted, data):
    pass
    
def iu():
    pass

def main(refit=False):
    data = read_data()

    if refit:
        model = fit(data, fp=MODEL_PATH, **VAR_PARAMS)
    else:
        try:
            model = sm.load_pickle(MODEL_PATH)
        except FileNotFoundError as e:
            print(e)
            return None, None
    fitted, forecast = predict(model, N_PREDS)
    if VAR_PARAMS['diff']:
        fitted, forecast = restore_diff(data, fitted, forecast, VAR_PARAMS['max_lags'])
    return fitted, forecast
if __name__ == '__main__':
    fitted, forecast = main(refit=True)
    print(fitted.shape, forecast.shape)
