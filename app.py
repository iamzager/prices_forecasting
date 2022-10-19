import streamlit as st
import argparse
import pandas as pd
import json

CV_RESULTS_PATH = 'cv_results.csv'
TYPES_PATH = 'types_dict.json'
DATA_PATH = 'prices_clean.csv'

@st.cache()
def read_cv_results(ts_type):
    try:
        cv_results = pd.read_csv(CV_RESULTS_PATH, index_col=0).round(2) 
    except FileNotFoundError as e:
        print(e)
        return
    return cv_results

@st.cache()
def read_types_dict():
    try:
        with open(TYPES_PATH) as f:
            types_dict = json.load(f)
    except FileNotFoundError as e:
        print(e)
        return
    return types_dict

if __name__ == '__main__':
    st.title('Прогноз средних недельных цен')
    
    parser = argparse.ArgumentParser(description='Run the app')
    parser.add_argument(
        '-t', '--type', type=int
        )
    args = parser.parse_args()
    TYPE_TO_PLOT = str(args.type)
    
    cv_results = read_cv_results(TYPE_TO_PLOT)
    types_dict = read_types_dict()

    st.write(f'TYPE: {types_dict[TYPE_TO_PLOT]}')
    st.write(f'Total CV MAPE: {cv_results.loc["total", :]}%')
    
    
