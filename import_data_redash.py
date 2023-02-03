# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:10:51 2023

@author: carlo
"""

import pandas as pd
import re, string, requests
import streamlit as st
from io import BytesIO

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
def import_txn_data():
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
    #xlsx = openpyxl.load_workbook(filename=data)
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