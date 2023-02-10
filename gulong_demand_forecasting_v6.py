# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:07:05 2023

@author: carlo
"""
import pandas as pd
import numpy as np

# for importing data
from import_gulong_redash import import_gulong_txn, import_gulong_traffic

# handling dates
import datetime as dt
from datetime import datetime
from calendar import monthrange

# plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# models and metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import Holt

# streamlit
import streamlit as st
#from st_aggrid import GridOptionsBuilder, AgGrid

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
#warnings.filterwarnings(action='ignore', category=ValueWarning)

@st.experimental_memo
def import_data():
    '''
    Wrapper function for importing data from redash so that caching can be applied.
    '''
    df_trans = import_gulong_txn()
    df_trans = df_trans.rename(columns = {'cost' : 'sales'})
    df_traffic = import_gulong_traffic()
    return df_trans, df_traffic

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

def naive_forecast(train):
    '''
    Helper function to implement naive_forecast given 1D series data

    Parameters
    ----------
    train : pandas series

    Returns
    -------
    - forecasted values for train data
    - 1-step forecast

    '''
    train_fit = train.shift().bfill()
    forecast = train.iloc[-1]
    
    return train_fit, forecast

def weighted_residuals(series):
    '''
    Calculate exponential weights (larger value for more recent) for input series
    Used by mean_forecast function
    
    Parameters
    ----------
    series: pandas series
    
    Returns
    -------
    weighted_series: pandas series of exponential weights
    '''
    series_len = len(series)
    wts = np.exp(-1*(series_len - 1 - np.array(range(series_len)))/series_len)
    weighted_series = pd.Series(series * wts)
    
    return weighted_series

def mean_forecast(train):
    '''
    Helper function to implement exponentially-weighted mean forecast

    Parameters
    ----------
    train : pandas series

    Returns
    -------
    - forecasted values for train data
    - 1-step forecast

    '''
    train_fit = pd.Series([weighted_residuals(train).mean()] * len(train))
    forecast = train_fit.iloc[-1]
    
    return train_fit, forecast

def poly_forecast(y_train, deg = 3):
    '''
    Helper function to implement polynomial fitting and forecast

    Parameters
    ----------
    y_train : pandas series
    deg : int, polynomial degree (optional)
        if provided, polynomial will use input degree;
        if None, automatically calculate degree with lowest RMSE

    Returns
    -------
    - forecasted values for train data
    - 1-step forecast

    '''
    
    def poly_pipeline(x_train, y_train, deg):
        '''
        Pipeline for transforming series data and training model
        '''
        polynomial_converter = PolynomialFeatures(degree = deg, 
                                                  interaction_only = False, 
                                                  include_bias=False)
        poly_features = polynomial_converter.fit_transform(x_train.array.reshape(-1,1))
        model = LinearRegression(fit_intercept = True)
        model.fit(poly_features, y_train)
        
        return model, poly_features, polynomial_converter
    
    x_train = pd.Series(range(len(y_train)))
    x_test = [x_train.iloc[-1] + 1]
    
    if deg is None:
        rmse_ = list()
        for d in range(1, 6):
            # create polynomial regression requirements
            model, poly_features, polynomial_converter = poly_pipeline(x_train, y_train, deg = d)
            # polynomial predictions for training data
            train_fit = pd.Series(model.predict(poly_features))
            rmse_.append(np.sqrt(mean_squared_error(y_train, train_fit)))
         
        # determine optimal degree
        opt_deg = rmse_[rmse_.index(min(rmse_))]
        
    else:
        opt_deg = deg
        
    model, poly_features, polynomial_converter = poly_pipeline(x_train, y_train, deg = opt_deg)
    train_fit = pd.Series(model.predict(poly_features))
    forecast = model.predict(polynomial_converter.transform([x_test]))[0]
    
    return train_fit, forecast

def linear_forecast(y_train):
    '''
    Helper function to implement linear regression fitting and forecast
    
    Parameters
    ----------
    y_train : pandas series
    
    Returns
    -------
    - forecasted values for train data
    - 1-step forecast
    
    '''
    
    x_train = pd.Series(range(len(y_train)))
    x_test = [x_train.iloc[-1] + 1]
    lin_reg = LinearRegression(fit_intercept = True)
    lin_reg.fit(x_train.array.reshape(-1,1), y_train.array.reshape(-1,1))
    
    train_fit = lin_reg.predict(x_train.array.reshape(-1,1)).flatten()
    forecast = lin_reg.predict([x_test])[0]
    
    return train_fit, forecast    

def holt_forecast(y_train):
    '''
    Helper to implement holt exponential smoothing to forecast
    
    parameters to extract from model:
        smoothing_level
        smoothing_trend
        damping_trend

    Parameters
    ----------
    y_train : pandas series

    Returns
    -------
    - forecasted values for train data
    - 1-step forecast

    '''
    model_holt = Holt(y_train, damped_trend = True).fit(optimized = True)
    forecast = model_holt.forecast(1).iloc[0]
    train_fit = model_holt.fittedfcast[:-1]
    return train_fit, forecast

def interval_score(df_pred, alpha):
    '''
    Calculates interval score of model given upper and lower bands, confidence level alpha
    Interval score = upper - lower + 2/alpha * (lower - x)I(x < lower) + 2/alpha * (x - upper)I(x > upper)
    
    Parameters
    ----------
    df_pred : pandas dataframe
        dataframe of lower and upper prediction interval of forecast
        index is the models used
        first row is true value
    alpha : confidence interval
        0.8, 0.9, etc
        
    Returns
    -------
    interval_score : Series of interval score for each method
    
    '''
    lower = df_pred.iloc[:, 0]
    upper = df_pred.iloc[:, 1]
    val = df_pred.iloc[0,0]
    interval_range = upper - lower
    lower_penalty = lower.apply(lambda x: (2/alpha) * (x - val) if val < x else 0)
    upper_penalty = upper.apply(lambda x: (2/alpha) * (val - x) if val > x else 0)
    int_score = (interval_range + lower_penalty + upper_penalty)/interval_range
    
    return int_score

def weighted_average(row):
    '''
    To be used in pandas apply
    weights is a pandas Series of weights
    '''
    weights = np.array(range(1, len(row) + 1))
    
    return (row * weights).sum()/weights.sum()

def eval_model_forecast(data, fcast_steps = 4, month = None, alpha = 80):
    '''
    Parameters
    ----------
    data: Series
        Series of data values
    trian: Series
        Series of train data values
    fcast_steps: int
        Number of forecast steps
    alpha: int
        confidence coefficient of forecast bounds
    
    Returns
    -------
    df_RMSE : dataframe
        dataframe of RMSE for each forecast step of each method given the train data
    
    '''
    # prediction interval multiplier
    PIM = {70: 1.04,
           75: 1.15,
           80: 1.28,
           90: 1.64}
    
    naive_RMSE, mean_RMSE, poly_RMSE, holt_RMSE = list(), list(), list(), list()
    naive_pred, mean_pred, poly_pred, holt_pred, y_true = list(), list(), list(), list(), list()
    
    for step in range(fcast_steps):
        #y_train = data.iloc[:train_len + step]
        train_indexer = -1*(fcast_steps - step - 1) if step != (fcast_steps - 1) else None
        
        y_train = data.loc[:month].iloc[:-1].iloc[:train_indexer]
        y_test = [data.iloc[len(y_train)]]
        y_true.append(y_test[0])
        y_true.append(y_test[0])
        
        # naive
        naive_train, naive_fcast = naive_forecast(y_train)
        naive_res = weighted_residuals(pd.Series(naive_train.values) - pd.Series(y_train.values))
        naive_RMSE.append(np.sqrt(mean_squared_error(y_test, [naive_fcast])))
        naive_pred.append(int(np.abs(naive_fcast - PIM[alpha]*naive_res.std() if PIM[alpha]*naive_res.std() < naive_fcast else 0)))
        naive_pred.append(int(np.abs(naive_fcast + PIM[alpha]*naive_res.std())))
        
        # mean
        mean_train, mean_fcast = mean_forecast(y_train)
        mean_res = weighted_residuals(pd.Series(mean_train.values) - pd.Series(y_train.values))
        mean_RMSE.append(np.sqrt(mean_squared_error(y_test, [mean_fcast])))
        mean_pred.append(int(np.abs(mean_fcast - PIM[alpha]*mean_res.std() if PIM[alpha]*mean_res.std() < mean_fcast else 0)))
        mean_pred.append(int(np.abs(mean_fcast + PIM[alpha]*mean_res.std())))
        
        # polynomial
        poly_train, poly_fcast = poly_forecast(y_train)
        poly_res = weighted_residuals(pd.Series(poly_train.values) - pd.Series(y_train.values))
        poly_pred.append(int(np.abs(poly_fcast - PIM[alpha]*poly_res.std() if PIM[alpha]*poly_res.std() < poly_fcast else 0)))
        poly_pred.append(int(np.abs(poly_fcast + PIM[alpha]*poly_res.std())))
        poly_RMSE.append(np.sqrt(mean_squared_error(y_test, [poly_fcast])))
        
        # holt
        holt_train, holt_fcast = holt_forecast(y_train)
        holt_res = weighted_residuals(holt_train - pd.Series(y_train.values))
        holt_pred.append(int(np.abs(holt_fcast - PIM[alpha]*holt_res.std() if PIM[alpha]*holt_res.std() < holt_fcast else 0)))
        holt_pred.append(int(np.abs(holt_fcast + PIM[alpha]*holt_res.std())))
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
    df_RMSE.columns = [f'{d}_RMSE' for d in data.iloc[-fcast_steps:].index] + ['avg_RMSE']
    df_RMSE.index = ['naive', 'mean', 'poly', 'holt']
    
    # create df_pred
    df_pred = pd.concat([pd.Series(y_true), 
                         pd.Series(naive_pred), 
                         pd.Series(mean_pred), 
                         pd.Series(poly_pred),
                         pd.Series(holt_pred)], axis=1).T
    pred_cols = [[f'{d}_lower', f'{d}_upper'] for d in data.iloc[-fcast_steps:].index]
    df_pred.columns = [item for sublist in pred_cols for item in sublist]
    df_pred.index = ['true', 'naive', 'mean', 'poly', 'holt']
    
    # calculate interval scoring
    df_interval_score = pd.concat([interval_score(df_pred.iloc[:,2*step:2*(step+1)], 
                                                  int(alpha)/100) for step in range(fcast_steps)], axis=1)
    df_interval_score['avg_interval_score'] = df_interval_score.apply(lambda x: weighted_average(x), axis=1)
    return df_RMSE, df_pred, df_interval_score

@st.experimental_memo
def main_sku_training(data, month = None, fcast_steps = 6):
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
    
    training_placeholder = st.empty()
    sku_demand_dict, sku_limits = {}, {}
    # streamlit show progress bar during training
    with training_placeholder.container():
        my_bar = st.progress(0)
        # iterate over all SKU models
        for model_ndx, model in enumerate(data.model.value_counts().index):
            my_bar.progress(int((model_ndx+1)/(len(data.model.value_counts().index))*100))
            print (model)
            # filter SKU from data and get monthly quantity data
            sku_data = data[data.model == model]
            
            # reindex to fill in month gaps
            monthly_model_data = sku_data.groupby('year-month')['quantity'].sum().reindex_like(month_format).fillna(0).reset_index()['quantity']
            monthly_model_data.index = month_format.index
            
            if month is not None:
                if month in monthly_model_data.index:
                    monthly_model_data = monthly_model_data.loc[:month].iloc[:-1]
                else:
                    monthly_model_data = monthly_model_data.loc[:month]
            else:
                pass
            
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
    
    # empty placeholder contents
    training_placeholder.empty()
        
    # construct dataframe
    df_sku_demand = pd.DataFrame.from_dict(sku_limits, orient = 'index')
    
    return df_sku_demand, sku_demand_dict

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


def model_forecast(data, model, month = None, alpha = 80):
    '''
    Calculate 1-step forecast given training data and best model
    
    Parameters
    ----------
    data : dataframe
        monthly training data
    model: str
        best model as determined from interval scores
    month : str
        forecast month
    alpha : str
        
    
    Returns
    -------
    pred_lower, pred_upper: lower and upper prediction bounds by model
    '''
    
    # prediction interval alphaiplier
    PIM = {70: 1.04,
           75: 1.15,
           80: 1.28,
           90: 1.64}
    
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
    
    train, fcast = model_[model](y_train)
    train = pd.Series(train)
    train.index = y_train.index
    res = weighted_residuals(pd.Series(train).reindex_like(y_train) - y_train.values)
    pred_lower = np.abs(fcast - PIM[alpha]*res.std() if PIM[alpha]*res.std() < fcast else 0)
    pred_upper = np.abs(fcast + PIM[alpha]*res.std() if PIM[alpha]*res.std() < fcast else 0)
    
    return pred_lower, pred_upper

@st.experimental_memo
def overall_sku_forecast(sku, data, sku_dict, month = None):
    '''
    Calculate overall monthly forecast from total SKU forecasts
    
    Parameters
    ----------
    sku : str
        SKU to forecast
    data: dataframe
        df_trans
    sku_dict: dictionary
        sku_demand_dict
    month: str
        forecast month
    
    Returns
    -------
    fcast: tuple
        lower and upper forecast bounds
    
    
    '''
    
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
    return fcast

def color_coding(row, best):
    '''
    Applies background color to row of best model in df_interval_score

    Parameters
    ----------
    row : df_interval_score row
        Need reset_index so best model column is not index
    best : str
        best model as determined from df_interval_score
        obtained from sku_demand_dict

    Returns
    -------
    list
        list of background color for each cell in row
        color depends on value of model

    '''
    
    return ['background-color:green'] * len(
        row) if row['index'] == best else ['background-color:white'] * len(row)

@st.experimental_memo
def main_overall_forecast(data, month = None, fcast_steps = 6):
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
    # leave out columns
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
        st.dataframe(df_interval_score.fillna(0).reset_index())
        # st.dataframe(df_interval_score.fillna(0).reset_index().style.\
        #               apply(lambda x: color_coding(x, best_model), axis=1))
        # reference: https://stackoverflow.com/questions/73940163/highlighting-specific-rows-in-streamlit-dataframe
        

    
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
        # st.dataframe(sku_dict['df_interval_score'].fillna(0).reset_index().style\
        #               .apply(lambda x: color_coding(x, sku_dict['best_model']), axis=1))
        
        # st.dataframe(sku_dict['df_interval_score'].fillna(0).reset_index())
    

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
def target_setting(data, target, month = None, fcast_steps = 6):
    '''
    
    Parameters
    ----------
    data : dataframe
        df_trans from import_data
    
    '''
    
    if month is not None:
        if month in data['year-month'].unique():
            month_format = data.groupby('year-month')[target].sum().loc[:month].iloc[:-1]
        else:
            month_format = data.groupby('year-month')[target].sum().loc[:month]
    else:
        month_format = data.groupby('year-month')[target].sum()
    
    # evaluate predictions of different models
    df_RMSE, df_pred, df_interval_score = eval_model_forecast(month_format, fcast_steps = fcast_steps, month = month)
    # get model with best interval score
    # leave out columns
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
    
    #fig_1.update_layout()
    st.plotly_chart(fig_1, use_container_width = True)
    
    with st.expander('Forecast details'):
        
        st.info('Forecast interval scores')
        # st.dataframe(df_interval_score.fillna(0).reset_index().style.\
        #               apply(lambda x: color_coding(x, best_model), axis=1))

## =========================== main flow ======================================
if __name__ == '__main__':
    st.title('Gulong.ph Demand Forecasting App')
    
    # 1. import data
    df_trans, df_traffic = import_data()

    # 2. select forecast month and training steps
    month_series = pd.Series(pd.date_range(df_trans.date.min().replace(day = 1), 
                                            datetime.today(), freq = 'MS'))\
                                            .dt.strftime('%Y-%m-%d').tolist()
    monthly_days = pd.Series([monthrange(*list(map(int, d.split('-')[:2])))[1] for d in list(month_series)])
    ## monthly_index
    monthly_days.index = month_series
    
    fcast_tab, overview_tab, target_tab = st.tabs(['Forecast', 'Transactions Overview',
                                                   'Target Setting'])
    
    with fcast_tab:
        date_today = datetime.today().date()
        reco_fcast_month = get_fcast_month(date_today)
        fcast_month_list = np.unique((list(month_series) + [reco_fcast_month]))[::-1]
        
        month_col, step_col = st.columns(2)
        with month_col:
            # select month to forecast
            fcast_month = st.selectbox('Select forecast month:',
                                       options = fcast_month_list,
                                       key = 'fcast_month',
                                       index = 0)
        
        with step_col:
            # select number of previous months for training
            fcast_steps = st.selectbox('# of previous months used for training:',
                                       options = range(2, 13),
                                       index = 4)
        
        # 3. Obtain interval scores and best model of each SKU
        df_sku_demand, sku_demand_dict = main_sku_training(df_trans, 
                                                           month = fcast_month, 
                                                           fcast_steps = fcast_steps)
        
        # calculate forecasts for all SKU
        # sku_fcast_lower_, sku_fcast_upper_ = list(), list()
        # for sku in df_sku_demand.index:
        #     sku_fcast_lower, sku_fcast_upper = overall_sku_forecast(sku, df_trans, 
        #                                                             sku_demand_dict[sku],
        #                                                             month = fcast_month)
        #     sku_fcast_lower_.append(sku_fcast_lower)
        #     sku_fcast_upper_.append(sku_fcast_upper)
        
        
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
            # ARIVO
            make_select = st.selectbox('MAKE: ',
                                       options = sorted(df_sku_demand.make.unique()),
                                       index = 1)
        with col2:
            # default
            make_filtered = df_sku_demand[df_sku_demand.make == make_select]
            dim_select = st.selectbox('DIMENSIONS: ',
                                      options = sorted(make_filtered.dimensions.unique()),
                                      index = 8)
        
        dim_filtered = make_filtered[make_filtered.dimensions == dim_select]
        SKU_select = st.selectbox('Filtered SKUs: ',
                                  options = sorted(list(dim_filtered.index)),
                                  index = 0)
        
        
        main_sku_forecast(SKU_select, df_trans, sku_demand_dict[SKU_select], month = fcast_month)
        
        # summary of SKU with most forecasted increase/decrease
        # add overall_sku_total forecasts to plot
        # add sku_forecast statistics
    with overview_tab:
        st.header('Brand Market Share %')
        brand_market_share = (df_trans.groupby(['year-month', 'make'])['quantity'].sum()*100\
                              /df_trans.groupby(['year-month'])['quantity'].sum()).reset_index()
        brand_market_share.columns = ['year-month', 'make', 'percentage']
        brand_market_share.loc[:,'quantity'] = df_trans.groupby(['year-month', 'make'])['quantity'].sum().reset_index()['quantity']
        
        fig_make = px.bar(brand_market_share, 
                     x = 'year-month', 
                     y = 'quantity',
                     color = 'make',
                     text = brand_market_share['percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
        st.plotly_chart(fig_make)
        
        st.header('Dimensions Market Share %')
        dims_market_share = (df_trans.groupby(['year-month', 'dimensions'])['quantity'].sum()*100\
                              /df_trans.groupby(['year-month'])['quantity'].sum()).reset_index()
        dims_market_share.columns = ['year-month', 'dimensions', 'percentage']
        dims_market_share.loc[:,'quantity'] = df_trans.groupby(['year-month', 'dimensions'])['quantity'].sum().reset_index()['quantity']
        
        fig_dims = px.bar(dims_market_share, 
                      x = 'year-month', 
                      y = 'quantity',
                      color = 'dimensions',
                      text = dims_market_share['percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
        st.plotly_chart(fig_dims)
        
        st.header('Models by Gross Income')
        models_GI_option = st.selectbox('Month to View',
                     options = ['All-time'] + list(df_trans['year-month'].unique()),
                     index = 0)
        
        if models_GI_option == 'All-time':
            models_gross_income = df_trans.groupby('model')[['sales', 'quantity']]\
                .sum().sort_values('sales', ascending = False)
        
        else:
            models_gross_income = df_trans[df_trans['year-month'] == models_GI_option]\
                .groupby('model')[['sales', 'quantity']].sum()\
                .sort_values('sales', ascending = False)
        
        fig_gross_income = px.bar(models_gross_income.reset_index().head(20), y = 'model', x = 'sales')
        st.plotly_chart(fig_gross_income)
        
        # demand fluctuation with price
        st.header('SKU Demand Fluctuation with price')
        sku_ = st.selectbox('Select SKU',
                            options = df_trans.model.value_counts().index,
                            index = 0)
        
        sku_demand_fluct = df_trans[df_trans.model == sku_][['date', 'quantity', 'price']]
        
        fig_demand_fluct = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_demand_fluct.add_trace(go.Scatter(x = sku_demand_fluct['date'],
                                 y = sku_demand_fluct['price'].diff().bfill(0),
                                 name = 'Price Difference',
                                 mode = 'lines+markers'),
                                 secondary_y = False)
        fig_demand_fluct.add_trace(go.Bar(x = sku_demand_fluct['date'],
                                          y = sku_demand_fluct['quantity'],
                                          name = 'Demand'),
                                   secondary_y = True)
        
        st.plotly_chart(fig_demand_fluct)
    
    with target_tab:
        target_select = st.selectbox('Target to forecast',
                                     options = ['sales', 'sessions'],
                                     index = 0)
        if target_select in df_trans.columns:
            target_data = df_trans
        elif target_select in df_traffic.columns:
            target_data = df_traffic
        else:
            raise Exception('Target is not found.')
            
        target_setting(target_data, target_select, month = fcast_month, fcast_steps = 6)
        # check regular purchases
        # check optimal price
        # check price elasticity