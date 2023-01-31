# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:07:05 2023

@author: carlo
"""
import pandas as pd
import numpy as np
import re, string, os
# for importing data
import openpyxl, requests
from io import BytesIO
# handling dates
import datetime as dt
from datetime import date, datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# models and metrics
from xgboost import XGBClassifier, XGBRegressor
from prophet import Prophet
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve,roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from calendar import monthrange

# streamlit
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
#warnings.filterwarnings(action='ignore', category=ValueWarning)

def remove_emoji(text):
    '''
    REMOVE EMOJIS FROM STRINGS
    '''
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()


def fix_name(name):
    '''
    Fix names which are duplicated.
    Ex. "John Smith John Smith"
    
    Parameters:
    -----------
    name: str
      
    Returns:
    --------
    fixed name; str
    
    '''
    name_corrections = {'MERCEDESBENZ' : 'MERCEDES-BENZ',
                        'MERCEDES BENZ' : 'MERCEDES-BENZ',
                        'IXSFORALL INC.' : 'Marvin Mendoza (iXSforAll Inc.)',
                        'GEMPESAWMENDOZA' : 'GEMPESAW-MENDOZA',
                        'MIKE ROLAND HELBES' : 'MIKE HELBES'}
    name_list = list()
    # removes emojis and ascii characters (i.e. chinese chars)
    name = remove_emoji(name).encode('ascii', 'ignore').decode()
    # split name by spaces
    for n in name.split(' '):
      if n not in name_list:
      # check each character for punctuations minus '.' and ','
          name_list.append(''.join([ch for ch in n 
                                  if ch not in string.punctuation.replace('.', '')]))
      else:
          continue
    name_ = ' '.join(name_list).strip().upper()
    for corr in name_corrections.keys():
        if re.search(corr, name_):
            name_ = name_corrections[re.search(corr, name_)[0]]
        else:
            continue
    return name_.strip().title()

def cleanup_specs(specs, col):
    '''
    Parameters
    ----------
    specs : string
        String to obtain specs info
    col : string
        Column name to apply function ('width', 'aspect_ratio', 'diameter')

    Returns
    -------
    specs : string
        Corresponding string value in specs to "col"

    '''
    specs_len = len(specs.split('/'))
    if specs_len == 1:
        return specs.split('/')[0]
    else:
        if col == 'width':
            return specs.split('/')[0]
        elif col == 'aspect_ratio':
            ar = specs.split('/')[1]
            if ar == '0' or ar == '':
                return 'R'
            else:
                return ar
        elif col == 'diameter':
            try:
                diameter = re.search('(?<=R)?[0-9]{2}', specs.split('/')[2])[0]
                return diameter
            except:
                return specs.split('/')[2]
        else:
            return specs

def combine_specs(row):
    '''
    Helper function to join corrected specs info

    Parameters
    ----------
    row : dataframe row
        From gogulong dataframe
    Returns
    -------
    string
        joined corrected specs info

    '''       
    if '.' in str(row['aspect_ratio']):
        return '/'.join([str(row['width']), str(float(row['aspect_ratio'])), str(row['diameter'])])
    else:
        return '/'.join([str(row['width']), str(row['aspect_ratio']), str(row['diameter'])])

@st.experimental_memo
def import_data():
    '''
    IMPORT DATA FROM REDASH QUERY FOR GULONG
    '''
    # Transactions data
    gulong_txns_redash = "http://app.redash.licagroup.ph/api/queries/104/results.csv?api_key=YqUI9o2bQn7lQUjlRd9gihjgAhs8ls1EBdYNixaO"
    df_txns_ = pd.read_csv(gulong_txns_redash , parse_dates = ['date'])
    # filter rows: include fulfilled, cancelled (via gulong due to no available stock/prod not available)
    df_txns_ = df_txns_[(df_txns_.status == 'fulfilled') | \
                        ((df_txns_.status == 'cancelled') & \
                         (df_txns_.cancel_type == 'gulong cancellation') & 
                         df_txns_.reason.isin(['no available stock', 
                                               'product not available']))]
    # filter by active brands
    active_brands = ['BRIDGESTONE', 'DUNLOP', 'GOODYEAR', 'MICHELIN', 'BFGOODRICH',
                     'MAXXIS', 'NEXEN', 'NITTO', 'TOYO', 'YOKOHAMA', 'ALLIANCE',
                     'ARIVO', 'DOUBLECOIN', 'TORQUE', 'WANLI', 'COOPER', 'CST',
                     'PRESA']
    
    df_txns_ = df_txns_[df_txns_.make.isin(active_brands)]
    
    # drop duplicates
    # exclude ID since some transactions have different IDs but same information
    df_txns_ = df_txns_.drop_duplicates(subset= ['date', 'name', 'make', 'cost'], keep='first')
    
    # cleanup columns
    #df_txns_.loc[:, 'date'] = df_txns_.loc[:,'date'].dt.date
    df_txns_.loc[:, 'name'] = df_txns_.apply(lambda x: fix_name(x['name']), axis=1)
    df_txns_.loc[:, 'width'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'width'), axis=1)
    df_txns_.loc[:, 'aspect_ratio'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'aspect_ratio'), axis=1)
    df_txns_.loc[:, 'diameter'] = df_txns_.apply(lambda x: cleanup_specs(x['dimensions'], 'diameter'), axis=1)
    df_txns_.loc[:, 'dimensions'] = df_txns_.apply(lambda x: combine_specs(x), axis=1)
    df_txns_.loc[:, 'year-month'] = df_txns_.apply(lambda x: '-'.join([str(x['date'].year), str(x['date'].month).zfill(2), '01']), axis=1)
    # filter based on dimension values
    df_txns_ = df_txns_[df_txns_.width.apply(lambda x: 1 if float(re.search('[\d\.]*', x)[0]) < 400 else 0) == 1]
    df_txns_ = df_txns_[(df_txns_.aspect_ratio == 'R') | \
                        (df_txns_[df_txns_.aspect_ratio != 'R'].aspect_ratio\
                         .apply(lambda x : 1 if float(x) > 10 else 0) == 1)]
    df_txns_ = df_txns_[df_txns_.diameter.apply(lambda x: 1 if x.isnumeric() else 0) == 1]
    return df_txns_

def ratio(a, b):
    '''
    Function for calculating ratios to avoid inf
    '''
    return a/b if b else 0 

@st.experimental_memo
def import_traffic_data():
    
    # Website/Traffic ads
    sheet_id = "17Yb2nYaf_0KHQ1dZBGBJkXDcpdht5aGE"
    sheet_name = 'summary'
    url = "https://docs.google.com/spreadsheets/export?exportFormat=xlsx&id=" + sheet_id
    res = requests.get(url)
    data = BytesIO(res.content)
    xlsx = openpyxl.load_workbook(filename=data)
    df_traffic = pd.read_excel(data, sheet_name = sheet_name)
    
    # daily traffic
    df_traffic = df_traffic.dropna(axis=1)
    df_traffic = df_traffic.rename(columns={'Unnamed: 0': 'date'})
    clicks_cols = ['link_clicks_ga', 'link_clicks_fb']
    impressions_cols = ['impressions_ga', 'impressions_fb']
    df_traffic.loc[:,'year-month'] = df_traffic.apply(lambda x: '-'.join([str(x['date'].year), str(x['date'].month).zfill(2)]), axis=1)
    df_traffic.loc[:, 'clicks_total'] = df_traffic.loc[:,clicks_cols].sum(axis=1)
    df_traffic.loc[:, 'impressions_total'] = df_traffic.loc[:,impressions_cols].sum(axis=1)
    df_traffic.loc[:, 'signups_total'] = df_traffic.loc[:,'signups_ga'] + df_traffic.loc[:, 'signups_backend']
    df_traffic.loc[:, 'ctr_ga'] = df_traffic.apply(lambda x: ratio(x['link_clicks_ga'], x['impressions_ga']), axis=1)
    df_traffic.loc[:, 'ctr_fb'] = df_traffic.apply(lambda x: ratio(x['link_clicks_fb'], x['impressions_fb']), axis=1)
    df_traffic.loc[:, 'ctr_total'] = df_traffic.apply(lambda x: ratio(x['clicks_total'], x['impressions_total']), axis=1)
    df_traffic.loc[:, 'ad_costs_total'] = df_traffic.loc[:, 'ad_costs_ga'] + df_traffic.loc[:, 'ad_costs_fb_total']
    
    purchases_backend_cols = [col for col in df_traffic.columns if 'purchases_backend' in col]
    df_traffic.loc[:, 'purchases_backend_total'] = df_traffic.loc[:,purchases_backend_cols].sum(axis=1)
    df_traffic.loc[:, 'purchases_backend_marketplace'] = df_traffic.loc[:, 'purchases_backend_fb'] + df_traffic.loc[:, 'purchases_backend_shopee'] + df_traffic.loc[:, 'purchases_backend_lazada']
    df_traffic.loc[:, 'purchases_backend_b2b'] = df_traffic.loc[:, 'purchases_backend_b2b'] + df_traffic.loc[:, 'purchases_backend_walk-in']
    df_traffic.drop(labels = ['purchases_backend_shopee', 'purchases_backend_lazada', 
                                 'purchases_backend_fb', 'purchases_backend_walk-in', 
                                 'purchases_backend_nan'], axis=1, inplace=True)
    return df_traffic

def eval_model(train, actual_train, test, actual_test):
    residuals = train - actual_train
    
    print ('Residuals check:')
    print (f'Residuals mean = {np.round(np.mean(residuals), 3)}')
    
    residuals_x = np.array(range(len(residuals))).reshape(-1,1)
    residuals_y = residuals.values.reshape(-1, 1)
    residual_lin_reg = LinearRegression()
    residual_lin_reg.fit(residuals_x, residuals_y)
    R2 = residual_lin_reg.score(residuals_x, residuals_y)
    print (f'Residuals correlation = {np.round(R2, 3)}')
    
    print ('Forecast Errors')
    # MAE:
    MAE = mean_absolute_error(actual_test, test)
    print (f'MAE =  {np.round(MAE, 3)}')
    
    # RMSE
    RMSE = np.sqrt(mean_squared_error(actual_test, test))
    print (f'RMSE = {np.round(RMSE,3)}')
    
def weighted_residuals(series):
    series_len = len(series)
    wts = np.exp(-1*(series_len - 1 - np.array(range(series_len)))/series_len)
    weighted_series = pd.Series(series * wts)
    return weighted_series

def naive_forecast(train):
    '''

    Parameters
    ----------
    train : pandas series

    Returns
    -------
    - forecasted values for train data
    - 1-step out of sample forecast

    '''
    train_fit = train.shift().bfill()
    forecast = train.iloc[-1]
    
    return train_fit, forecast

def mean_forecast(train):
    '''

    Parameters
    ----------
    train : pandas series

    Returns
    -------
    - forecasted values for train data
    - 1-step out of sample forecast

    '''
    train_fit = pd.Series([weighted_residuals(train).mean()] * len(train))
    forecast = train_fit.iloc[-1]
    
    return train_fit, forecast

def poly_forecast(y_train, deg = 3):
    '''

    Parameters
    ----------
    y_train : pandas series
    deg : polynomial degree

    Returns
    -------
    - forecasted values for train data
    - 1-step out of sample forecast

    '''
    
    def poly_pipeline(x_train, y_train, deg):
        polynomial_converter = PolynomialFeatures(degree = d, 
                                                  interaction_only = False, 
                                                  include_bias=False)
        poly_features = polynomial_converter.fit_transform(x_train.array.reshape(-1,1))
        model = LinearRegression(fit_intercept = True)
        model.fit(poly_features, y_train)
        return model, poly_features, polynomial_converter
    
    
    x_train = pd.Series(range(len(y_train)))
    x_test = [x_train.iloc[-1] + 1]
    rmse_ = list()
    for d in range(1, 6):
        model, poly_features, polynomial_converter = poly_pipeline(x_train, y_train, deg = d)
    
        train_fit = pd.Series(model.predict(poly_features))
        rmse_.append(np.sqrt(mean_squared_error(y_train, train_fit)))
    
    # determine optimal degree
    opt_deg = rmse_[rmse_.index(min(rmse_))]
    opt_model, opt_poly_features, polynomial_converter = poly_pipeline(x_train, y_train, deg = opt_deg)
    forecast = opt_model.predict(polynomial_converter.transform([x_test]))[0]
    return train_fit, forecast

def linear_forecast(y_train):
    x_train = pd.Series(range(len(y_train)))
    x_test = [x_train.iloc[-1] + 1]
    lin_reg = LinearRegression(fit_intercept = True)
    lin_reg.fit(x_train.array.reshape(-1,1), y_train.array.reshape(-1,1))
    
    train_fit = lin_reg.predict(x_train.array.reshape(-1,1)).flatten()
    forecast = lin_reg.predict([x_test])[0]
    return train_fit, forecast    

def holt_forecast(y_train):
    '''
    parameters to extract from model:
        smoothing_level
        smoothing_trend
        damping_trend

    Parameters
    ----------
    x_train : pandas series
    y_train : pandas series
    x_test : pandas series
    deg : polynomial degree

    Returns
    -------
    - forecasted values for train data
    - 1-step out of sample forecast

    '''
    model_holt = Holt(y_train, damped_trend = True).fit(optimized = True)
    forecast = model_holt.forecast(1).iloc[0]
    train_fit = model_holt.fittedfcast[:-1]
    return train_fit, forecast

# time series cross-validation
def time_series_cv(data, start_len):
    '''
    Calculates the RMSE for a model under time series cross validation method
    Forecast step = 1
    
    Parameters
    ----------
    data : Pandas Series
    start_len: int
        starting length of training/estimation data
    
    Returns
    -------
    np.mean(RMSE_list) : mean of RMSE for each step
    
    '''
    RMSE_list = list()
    for i in range((len(data) - start_len)):
        start_train, fcast = mean_forecast(data.iloc[:start_len + i])
        RMSE_list.append(np.sqrt(mean_squared_error([data.iloc[start_len + i]], [fcast])))
        
    return np.mean(RMSE_list)

def interval_score(df_pred, alpha):
    '''
    Parameters
    ----------
    df_pred : pandas dataframe
        dataframe of lower and upper prediction interval of forecast
        index is the model used
        first row is true value
    alpha : confidence interval
        0.8, 0.9, etc
        
    Returns
    -------
    Series of interval score for each method
    
    '''
    lower = df_pred.iloc[:, 0]
    upper = df_pred.iloc[:, 1]
    val = df_pred.iloc[0,0]
    interval_range = upper - lower
    lower_penalty = lower.apply(lambda x: (2/alpha) * (x - val) if val < x else 0)
    upper_penalty = upper.apply(lambda x: (2/alpha) * (val - x) if val > x else 0)
    return (interval_range + lower_penalty + upper_penalty)/interval_range

def weighted_average(row, weights):
    '''
    To be used in pandas apply
    weights is a pandas Series of weights
    '''
    return (row * weights).sum()/weights.sum()


def eval_model_forecast(data, fcast_steps = 4, month = None):
    '''
    Parameters
    ----------
    data: Series
        Series of data values
    trian: Series
        Series of train data values
    fcast_steps: int
        Number of forecast steps
    
    Returns
    -------
    df_RMSE : dataframe
        dataframe of RMSE for each forecast step of each method given the train data
    
    '''
    # prediction interval multiplier
    PIM = {'70': 1.04,
           '75': 1.15,
           '80': 1.28,
           '90': 1.64}
    
    mult = '80'
    naive_RMSE, mean_RMSE, poly_RMSE, holt_RMSE = list(), list(), list(), list()
    naive_pred, mean_pred, poly_pred, holt_pred, y_true = list(), list(), list(), list(), list()
    
    for step in range(fcast_steps):
        #y_train = data.iloc[:train_len + step]
        train_indexer = -1*(fcast_steps - step - 1) if step != (fcast_steps - 1) else None
        
        y_train = data.loc[:month].iloc[:-1].iloc[:train_indexer]
        x_train = pd.Series(range(len(y_train)))
        
        #y_test = [data.iloc[train_len + step]]
        y_test = [data.loc[:month].iloc[len(y_train)]]
        x_test = [x_train.iloc[-1] + 1]
        y_true.append(y_test[0])
        y_true.append(y_test[0])
        
        # naive
        naive_train, naive_fcast = naive_forecast(y_train)
        naive_res = weighted_residuals(pd.Series(naive_train.values) - pd.Series(y_train.values))
        naive_RMSE.append(np.sqrt(mean_squared_error(y_test, [naive_fcast])))
        naive_pred.append(int(np.abs(naive_fcast - PIM[mult]*naive_res.std() if PIM[mult]*naive_res.std() < naive_fcast else 0)))
        naive_pred.append(int(np.abs(naive_fcast + PIM[mult]*naive_res.std())))
        
        # mean
        mean_train, mean_fcast = mean_forecast(y_train)
        mean_res = weighted_residuals(pd.Series(mean_train.values) - pd.Series(y_train.values))
        mean_RMSE.append(np.sqrt(mean_squared_error(y_test, [mean_fcast])))
        mean_pred.append(int(np.abs(mean_fcast - PIM[mult]*mean_res.std() if PIM[mult]*mean_res.std() < mean_fcast else 0)))
        mean_pred.append(int(np.abs(mean_fcast + PIM[mult]*mean_res.std())))
        
        # polynomial
        poly_train, poly_fcast = poly_forecast(y_train, deg = 4)
        poly_res = weighted_residuals(pd.Series(poly_train.values) - pd.Series(y_train.values))
        poly_pred.append(int(np.abs(poly_fcast - PIM[mult]*poly_res.std() if PIM[mult]*poly_res.std() < poly_fcast else 0)))
        poly_pred.append(int(np.abs(poly_fcast + PIM[mult]*poly_res.std())))
        poly_RMSE.append(np.sqrt(mean_squared_error(y_test, [poly_fcast])))
        
        # holt
        holt_train, holt_fcast = holt_forecast(y_train)
        holt_res = weighted_residuals(holt_train - pd.Series(y_train.values))
        holt_pred.append(int(np.abs(holt_fcast - PIM[mult]*holt_res.std() if PIM[mult]*holt_res.std() < holt_fcast else 0)))
        holt_pred.append(int(np.abs(holt_fcast + PIM[mult]*holt_res.std())))
        holt_RMSE.append(np.sqrt(mean_squared_error(y_test, [holt_fcast])))
    
    # df_RMSE
    naive_RMSE.append(np.mean(naive_RMSE))
    mean_RMSE.append(np.mean(mean_RMSE))
    poly_RMSE.append(np.mean(poly_RMSE))
    holt_RMSE.append(np.mean(holt_RMSE))
    
    df_RMSE = pd.concat([pd.Series(naive_RMSE), 
                         pd.Series(mean_RMSE), 
                         pd.Series(poly_RMSE),
                         pd.Series(holt_RMSE)], axis = 1).T
    df_RMSE.columns = [f'{d}_RMSE' for d in y_train.iloc[-fcast_steps:].index] + ['avg_RMSE']
    df_RMSE.index = ['naive', 'mean', 'poly', 'holt']
    
    # create df_pred
    df_pred = pd.concat([pd.Series(y_true), 
                         pd.Series(naive_pred), 
                         pd.Series(mean_pred), 
                         pd.Series(poly_pred),
                         pd.Series(holt_pred)], axis=1).T
    pred_cols = [[f'{d}_lower', f'{d}_upper'] for d in y_train.iloc[-fcast_steps:].index]
    df_pred.columns = [item for sublist in pred_cols for item in sublist]
    df_pred.index = ['true', 'naive', 'mean', 'poly', 'holt']
    
    # interval scoring
    weights = pd.Series(list(range(1,fcast_steps+1)))
    df_interval_score = pd.concat([interval_score(df_pred.iloc[:,2*step:2*(step+1)], int(mult)/100) for step in range(fcast_steps)], axis=1)
    df_interval_score['avg_interval_score'] = df_interval_score.apply(lambda x: weighted_average(x, weights), axis=1)
    return df_RMSE, df_pred, df_interval_score

def model_forecast(data, model, month = None):
    '''
    Parameters
    ----------
    data : dataframe
    
    model: str
    
    Returns
    -------
    '''
    
    # prediction interval multiplier
    PIM = {'70': 1.04,
           '75': 1.15,
           '80': 1.28,
           '90': 1.64}
    
    mult = '80'
    
    model_ = {'naive' : naive_forecast,
              'mean': mean_forecast,
              'poly': poly_forecast,
              'holt': holt_forecast}
    
    if month != None:
        if month in data.index:
            y_train = data.loc[:month].iloc[:-1]
        else:
            y_train = data
    else:
        y_train = data
        
    x_train = pd.Series(range(len(y_train)))
    
    train, fcast = model_[model](y_train)
    train = pd.Series(train)
    train.index = y_train.index
    res = weighted_residuals(pd.Series(train).reindex_like(y_train) - y_train.values)
    pred_lower = np.abs(fcast - PIM[mult]*res.std() if PIM[mult]*res.std() < fcast else 0)
    pred_upper = np.abs(fcast + PIM[mult]*res.std() if PIM[mult]*res.std() < fcast else 0)
    
    return pred_lower, pred_upper

def get_daily_avg(data):
    '''
    Convert monthly total data to daily avg

    Parameters
    ----------
    data : dataframe
        df_trans

    Returns
    -------
    monthly_daily_avg_qty : dataframe

    '''
    monthly_total_qty = data
    monthly_days = pd.Series([monthrange(*list(map(int, d.split('-')[:2])))[1] for d in monthly_total_qty.index])
    ## calendar adjustment (removal of dependence on # of days in month)
    monthly_daily_avg_qty = pd.Series(monthly_total_qty.values/monthly_days.values)
    monthly_daily_avg_qty.index = monthly_total_qty.index
    
    return monthly_daily_avg_qty

def go_scat(name, x, y, c = None, dash = 'solid'):
    # add dash lines
    if c is None:
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'lines+markers',
                            marker = dict(size=6),
                            line = dict(dash = dash,
                                        color = c))
    else:
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'lines+markers',
                            marker = dict(size=6),
                            line = dict(dash = dash,
                                        color = c))
                                        
    return go_fig

def color_coding(row, best):
    
    return ['background-color:green'] * len(
        row) if row.index == best else ['background-color:red'] * len(row)

@st.experimental_memo
def main_overall_forecast(data, month = None, fcast_steps = 4):
    '''
    
    Parameters
    ----------
    data : dataframe
        df_trans from import_data
    
    '''
    
    if month is not None:
        if month in data['year-month'].unique():
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month]
    else:
        month_format = data.groupby('year-month')['quantity'].sum()
    
    # evaluate predictions of different models
    df_RMSE, df_pred, df_interval_score = eval_model_forecast(month_format, fcast_steps = fcast_steps, month = month)
    # get model with best interval score
    min_int_score = df_interval_score['avg_interval_score'].iloc[1:].min()
    
    df_interval_score.columns = list(month_format.iloc[-fcast_steps:].index) + ['avg_interval_score']
    # get best model based on interval score (disregarding true)
    best_model = df_interval_score.iloc[1:][df_interval_score['avg_interval_score'].iloc[1:] == min_int_score].index[0]
    fcast = model_forecast(month_format, best_model, month = month)
    
    # main data
    fig_1 = go.Figure(data = go_scat(name = 'monthly_overall',
                                        x = month_format.index,
                                        y = month_format.values,
                                        c = '#36454F',
                                        dash = 'solid'))
    # lower limit
    lower_lim_cols = [col for col in df_pred.columns if 'lower' in col]
    y_lower = df_pred.loc[best_model, lower_lim_cols].tolist() + [fcast[0]]
    x_lower = list(month_format.iloc[-fcast_steps:].index) + [fcast_month]
    fig_1.add_trace(go_scat(name = 'lower_pred',
                                   x = x_lower,
                                   y = y_lower,
                                   c = '#FF0000',
                                   dash = 'solid'))
    
    # upper limit
    upper_lim_cols = [col for col in df_pred.columns if 'upper' in col]
    y_upper = df_pred.loc[best_model, upper_lim_cols].tolist() + [fcast[1]]
    x_upper = list(month_format.iloc[-fcast_steps:].index) + [fcast_month]
    fig_1.add_trace(go_scat(name = 'upper_pred',
                                   x = x_upper,
                                   y = y_upper,
                                   c = '#008000',
                                   dash = 'solid'))
    
    # 6-month
    train_6mo, fcast_6mo = linear_forecast(month_format.iloc[-6:])
    fig_1.add_trace(go_scat(name = '6 mo. linear fit',
                                   x = list(month_format.index[-6:]) + [fcast_month],
                                   y = list(train_6mo) + [fcast_6mo[0]],
                                   c = '#0F52BA',
                                   dash = 'dash'))
    
    # 1-year polynomial fit
    train_1y, fcast_1y = linear_forecast(month_format.iloc[-12:])
    fig_1.add_trace(go_scat(name = '1 yr. linear fit',
                                   x = list(month_format.index[-12:]) + [fcast_month],
                                   y = list(train_1y) + [fcast_1y[0]],
                                   c = '#7393B3',
                                   dash = 'dash'))
    
    fig_1.update_layout(yaxis_range = [0, 1000])
    st.plotly_chart(fig_1, use_container_width = True)
    
    with st.expander('Forecast details'):
        
        st.info('Forecast interval scores')
        # st.dataframe(df_interval_score.fillna(0).style.\
        #              apply(lambda x: color_coding(x, best_model), axis=1))
        # reference: https://stackoverflow.com/questions/73940163/highlighting-specific-rows-in-streamlit-dataframe
        st.dataframe(df_interval_score.fillna(0))
    

@st.experimental_memo
def main_sku_training(data, month = None, fcast_steps = 4):
    '''
    Model training and selection based on one-step forecasting for each SKU
    
    Parameters
    ----------
    data : dataframe
        df_trans from import_data
    index : Index object
        monthly_index
        
    Returns
    -------
    
    sku_demand_limits : dataframe
        dataframe containing info on training RMSE, predictions, interval_score, best_model
    
    '''
    if month is not None:
        if month in data['year-month'].unique():
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month]
    else:
        month_format = data.groupby('year-month')['quantity'].sum()
        
    sku_demand_dict, sku_limits = {}, {}
    # iterate over all SKU models
    for model in data.model.value_counts().index:
        print(model)
        # filter SKU from data and get monthly quantity data
        sku_data = data[data.model == model]
        if month is not None:
            monthly_model_data = sku_data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            monthly_model_data = sku_data.groupby('year-month')['quantity'].sum()
        
        # reindex to fill in month gaps
        monthly_model_data = monthly_model_data.reindex_like(month_format).fillna(0).reset_index()['quantity']
        monthly_model_data.index = month_format.index
        # evaluate predictions of different models
        df_RMSE, df_pred, df_interval_score = eval_model_forecast(monthly_model_data, fcast_steps = fcast_steps, month = month)
        # get model with best interval score
        min_int_score = df_interval_score['avg_interval_score'].iloc[1:].min()
        df_interval_score.columns = list(monthly_model_data.iloc[-fcast_steps:].index) + ['avg_interval_score']
        # get best model based on interval score (disregarding true)
        best_model = df_interval_score.iloc[1:][df_interval_score['avg_interval_score'].iloc[1:] == min_int_score].index[0]
        # get SKU info
        model_info = sku_data[['make', 'dimensions']].iloc[0]
        sku_limits[model] = pd.concat([model_info, df_pred.loc[best_model,:]], axis = 0)
        # store data into dictionary
        sku_demand_dict[model] = {'df_RMSE': df_RMSE,
                                   'df_pred': df_pred,
                                   'df_interval_score': df_interval_score,
                                   'best_model': best_model}
    # construct dataframe
    df_sku_demand = pd.DataFrame.from_dict(sku_limits, orient = 'index')
    
    return df_sku_demand, sku_demand_dict

def overall_sku_forecast(sku, data, sku_dict, month = None):
    if month is not None:
        if month in data['year-month'].unique():
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month]
    else:
        month_format = data.groupby('year-month')['quantity'].sum()
    
    sku_data = data[data.model == sku]
    if month is not None:
        monthly_model_data = sku_data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
    else:
        monthly_model_data = sku_data.groupby('year-month')['quantity'].sum()
    
    # reindex to fill in month gaps
    y_train = monthly_model_data.reindex_like(month_format).fillna(0).reset_index()['quantity']
    y_train.index = month_format.index
    fcast = model_forecast(y_train, sku_dict['best_model'], month = month)
    
    

@st.experimental_memo
def main_sku_forecast(sku, data, sku_dict, month = None):
    if month is not None:
        if month in data['year-month'].unique():
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            month_format = data.groupby('year-month')['quantity'].sum().loc[:month]
    else:
        month_format = data.groupby('year-month')['quantity'].sum()
    
    sku_data = data[data.model == sku]
    if month is not None:
        monthly_model_data = sku_data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
    else:
        monthly_model_data = sku_data.groupby('year-month')['quantity'].sum()
    
    # reindex to fill in month gaps
    y_train = monthly_model_data.reindex_like(month_format).fillna(0).reset_index()['quantity']
    y_train.index = month_format.index
    fcast = model_forecast(y_train, sku_dict['best_model'], month = month)
    
    # main data
    fig_2 = go.Figure(data = go_scat(name = 'monthly_sku',
                                        x = y_train.index,
                                        y = y_train.values,
                                        c = '#36454F',
                                        dash = 'solid'))
    # lower limit
    lower_lim_cols = [col for col in sku_dict['df_pred'].columns if 'lower' in col]
    y_lower = sku_dict['df_pred'].loc[sku_dict['best_model'], lower_lim_cols].tolist() + [fcast[0]]
    x_lower = list(month_format.iloc[-fcast_steps:].index) + [fcast_month]
    fig_2.add_trace(go_scat(name = 'lower_pred',
                                   x = x_lower,
                                   y = y_lower,
                                   c = '#FF0000',
                                   dash = 'solid'))
    
    # upper limit
    upper_lim_cols = [col for col in sku_dict['df_pred'].columns if 'upper' in col]
    y_upper = sku_dict['df_pred'].loc[sku_dict['best_model'], upper_lim_cols].tolist() + [fcast[1]]
    x_upper = list(month_format.iloc[-fcast_steps:].index) + [fcast_month]
    fig_2.add_trace(go_scat(name = 'upper_pred',
                                   x = x_upper,
                                   y = y_upper,
                                   c = '#008000',
                                   dash = 'solid'))
    
    # 6-month
    train_6mo, fcast_6mo = linear_forecast(y_train.iloc[-6:])
    fig_2.add_trace(go_scat(name = '6 mo. linear fit',
                                   x = list(month_format.index[-6:]) + [fcast_month],
                                   y = list(train_6mo) + [fcast_6mo[0]],
                                   c = '#0F52BA',
                                   dash = 'dash'))
    
    # 1-year polynomial fit
    train_1y, fcast_1y = linear_forecast(y_train.iloc[-12:])
    fig_2.add_trace(go_scat(name = '1 yr. linear fit',
                                   x = list(month_format.index[-12:]) + [fcast_month],
                                   y = list(train_1y) + [fcast_1y[0]],
                                   c = '#7393B3',
                                   dash = 'dash'))
    
    #fig_2.update_layout(yaxis_range = [0, 1000])
    st.plotly_chart(fig_2, use_container_width = True)
    
    with st.expander('Forecast details'):
        
        st.info('Forecast interval scores')
        # st.dataframe(df_interval_score.fillna(0).style.\
        #              apply(lambda x: color_coding(x, best_model), axis=1))
        # reference: https://stackoverflow.com/questions/73940163/highlighting-specific-rows-in-streamlit-dataframe
        st.dataframe(sku_dict['df_interval_score'].fillna(0))
    

@st.experimental_memo
def main_forecast(data, model_dict, month = None):
    '''
    One-step forecasting for each SKU
    
    Parameters
    ----------
    data : dataframe
        df_trans from import_data
    index : Index object
        monthly_index
        
    Returns
    -------
    
    sku_forecast : dataframe
        dataframe containing info on predictions
    
    '''
    
    if month is not None:
        month_format = data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
    else:
        month_format = data.groupby('year-month')['quantity'].sum()
        
    sku_forecast = {}
    for model in data.model.value_counts().index:
        print(model)
        # filter SKU from data and get monthly quantity data
        sku_data = data[data.model == model]
        if month is not None:
            monthly_model_data = sku_data.groupby('year-month')['quantity'].sum().loc[:month].iloc[:-1]
        else:
            monthly_model_data = sku_data.groupby('year-month')['quantity'].sum()
        # reindex to fill in month gaps
        monthly_model_data = monthly_model_data.reindex_like(month_format).fillna(0).reset_index()['quantity']
        monthly_model_data.index = month_format.index
        # get best model from training
        best_model = model_dict[model]['best_model']
        # get forecast using best model
        pred_lower, pred_upper = model_forecast(monthly_model_data, best_model, month = month)
        if type(pred_lower) == np.ndarray or type(pred_upper) == np.ndarray:
            print('Wrong prediction data type')
            break
        # get SKU info
        model_info = sku_data[['make', 'dimensions']].iloc[0]
        sku_forecast[model] = pd.concat([model_info, pd.Series([pred_lower, pred_upper])], axis=0)
    
    monthly_sku_forecast = pd.DataFrame.from_dict(sku_forecast, orient = 'index')
    return monthly_sku_forecast

@st.experimental_memo
def get_fcast_month(date_today):
    '''
    Returns the month to be forecasted based on current day
    Format: YYYY-MM-01
    Requires import datetime as dt
    from datetime import timedelta
    
    Parameters
    ----------
    date_today: datetime
    
    '''
    if date_today.day < 15:
        # get first day of current month
        fcast_month = date_today.replace(day=1).strftime('%Y-%m-%d')
    else:
        # get first day of next month
        # advance day to any number more than any month then get first day of that month
        fcast_month = (date_today.replace(day=1) + dt.timedelta(days=32)).replace(day=1).strftime('%Y-%m-%d')
    return fcast_month
    

if __name__ == '__main__':
    st.title('Gulong.ph Demand Forecasting App')
    
    # import data
    df_trans = import_data()
    df_traffic = import_traffic_data()
    
    # construct main monthly dataframe
    monthly_total_qty = df_trans.groupby('year-month')['quantity'].sum()
    monthly_days = pd.Series([monthrange(*list(map(int, d.split('-')[:2])))[1] for d in monthly_total_qty.index])
    ## monthly_index
    monthly_days.index = monthly_total_qty.index
    
    # select forecast month
    date_today = datetime.today().date()
    reco_fcast_month = get_fcast_month(date_today)
    fcast_month_list = np.unique((list(monthly_total_qty.index) + [reco_fcast_month]))[::-1]
    fcast_month = st.selectbox('Select forecast month:',
                               options = fcast_month_list,
                               key = 'fcast_month',
                               index = 0)
    
    fcast_steps = 4
    df_sku_demand, sku_demand_dict = main_sku_training(df_trans, month = fcast_month, fcast_steps = fcast_steps)
    
    # calculate forecasts for all SKU
    for sku in df_sku_demand.index:
        
    
    
    ## calendar adjustment (removal of dependence on # of days in month)
    #monthly_daily_avg_qty = pd.Series(monthly_total_qty.values/monthly_days.values)
    #monthly_daily_avg_qty.index = monthly_total_qty.index

    # show forecast for overall quantity trend
    # toggle setting to show underlying different model forecast results
    # default show only best forecast results
    
    st.header('Monthly Gulong Overall Demand Forecast')
    main_overall_forecast(df_trans, month = fcast_month, fcast_steps = fcast_steps)
    
    st.header('Monthly Gulong SKU Demand Forecast')
    
    # filter by make, dimensions
    col1, col2 = st.columns(2)
    with col1:
        make_select = st.selectbox('MAKE: ',
                                   options = sorted(df_sku_demand.make.unique()),
                                   index = 0)
    with col2:
        make_filtered = df_sku_demand[df_sku_demand.make == make_select]
        dim_select = st.selectbox('DIMENSIONS: ',
                                  options = sorted(make_filtered.dimensions.unique()),
                                  index = 0)
    
    dim_filtered = make_filtered[make_filtered.dimensions == dim_select]
    SKU_select = st.selectbox('Filtered SKUs: ',
                              options = sorted(list(dim_filtered.index)),
                              index = 0)
    
    # summary of SKU with most forecasted increase/decrease
    main_sku_forecast(SKU_select, df_trans, sku_demand_dict[SKU_select], month = fcast_month)

    # add overall_sku_total forecasts to plot
    # add sku_forecast statistics