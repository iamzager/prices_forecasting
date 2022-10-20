import streamlit as st
import argparse
import pandas as pd
import json

CV_RESULTS_PATH = 'cv_results.csv'
TYPES_PATH = 'types_dict.json'
DATA_PATH = 'prices_clean.csv'
FORECAST_PATH = 'forecast.csv'
LOWER_PATH = 'forecast_lower.csv'
UPPER_PATH = 'forecast_upper.csv'


@st.cache()
def read_data():
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
    return data, types_dict

@st.cache()
def read_cv_results(ts_type):
    try:
        cv_results = pd.read_csv(CV_RESULTS_PATH, index_col=0).round(2) 
    except FileNotFoundError as e:
        print(e)
        return
    return cv_results

@st.cache()
def read_forecast():
    try:
        yhat = pd.read_csv(FORECAST_PATH, index_col=0)
        lower = pd.read_csv(LOWER_PATH, index_col=0)
        upper = pd.read_csv(UPPER_PATH, index_col=0)
    except FileNotFoundError as e:
        print(e)
        return
    return yhat, lower, upper

if __name__ == '__main__':
    st.title('Прогноз средних недельных цен')
    
    parser = argparse.ArgumentParser(description='Run the app')
    parser.add_argument(
        '-t', '--type', type=int
        )
    args = parser.parse_args()
    TYPE_TO_PLOT = str(args.type)
    
    cv_results = read_cv_results(TYPE_TO_PLOT)
    data, types_dict = read_data()

    st.write(f'TYPE: {types_dict[TYPE_TO_PLOT]}')
    st.write(f'Total CV MAPE: {cv_results.loc["total", :]}%')
    
    
