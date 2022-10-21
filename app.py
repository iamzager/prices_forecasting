import streamlit as st
import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt

CV_RESULTS_PATH = 'cv_results.csv'
TYPES_PATH = 'types_dict.json'
DATA_PATH = 'prices_clean.csv'
FORECAST_PATH = 'forecast.csv'
LOWER_PATH = 'forecast_lower.csv'
UPPER_PATH = 'forecast_upper.csv'


@st.cache()
def read_data():
    # st.write('reading data')
    try:
        data = pd.read_csv(DATA_PATH, index_col=[0], parse_dates=[0])
        with open(TYPES_PATH) as f:
            types_dict = json.load(f)
            inversed_types_dict = {value : key for key, value in types_dict.items()}
        data = data.dropna(axis=1)
    except FileNotFoundError as e:
        print(e)
        return    

    # !!!!
    max_date = data.index.max()
    data.index = pd.date_range(end=max_date, periods=data.shape[0], freq='7D')
    data = data.sort_index()
    return data, types_dict, inversed_types_dict

@st.cache()
def read_cv_results():
    # st.write('reading types')
    try:
        cv_results = pd.read_csv(CV_RESULTS_PATH, index_col=0).round(2) 
    except FileNotFoundError as e:
        print(e)
        return
    return cv_results

@st.cache()
def read_forecast():
    # st.write('reading forecast')
    try:
        yhat = pd.read_csv(FORECAST_PATH, index_col=0, parse_dates=[0])
        lower = pd.read_csv(LOWER_PATH, index_col=0, parse_dates=[0])
        upper = pd.read_csv(UPPER_PATH, index_col=0, parse_dates=[0])
    except FileNotFoundError as e:
        print(e)
        return
    return yhat, lower, upper

def plot(yhat, lower, upper, y, name):
    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(y, label='Actual', color='r')
        ax.plot(yhat, '-o', markersize=5, label='Model', color='g')
        ax.plot(lower, "r--", alpha=0.5, label = "Predictive interval", color='g')
        ax.plot(upper, "r--", alpha=0.5, color='g')
        ax.fill_between(x=upper.index, y1=lower, y2=upper, alpha=0.2, color = "grey")  
        ax.legend()
    return fig


if __name__ == '__main__':
    st.title('Прогноз средних недельных цен по данным Росстата')
    
    cv_results = read_cv_results()
    data, types_dict, inversed_types_dict = read_data()
    yhat, lower ,upper = read_forecast()
    
    TYPE_TO_PLOT = st.selectbox(
        '',\
        list(map(types_dict.get, data.columns)),\
        index=0
        )
    
    TYPE_TO_PLOT = inversed_types_dict[TYPE_TO_PLOT]
    
    
    fig = plot(
        yhat[TYPE_TO_PLOT][-100:],
        lower[TYPE_TO_PLOT][-100:],
        upper[TYPE_TO_PLOT][-100:],
        data[TYPE_TO_PLOT][-100:],
        types_dict[TYPE_TO_PLOT],
        )
    col1, col2 = st.columns(2)
    col1.metric("Mean MAPE", f'{cv_results.at["total", "cv_test_mean"]} \u00B1 {cv_results.at["total", "cv_test_std"]}%')
    col2.metric("MAPE", f'{cv_results.at[TYPE_TO_PLOT, "cv_test_mean"]} \u00B1 {cv_results.at[TYPE_TO_PLOT, "cv_test_std"]}%')
    
    #magic    
    fig
    

    
    
