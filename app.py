from enum import unique
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import sys

st.set_page_config(layout="wide")


def _event_log(event_log):
    """
    Function to read event log and return a dataframe
    """
    # Read event log
    df = pd.read_csv(event_log, sep='\t', header=None)
    df.columns = ['timestamp', 'event_type', 'event_data']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def check(df, val):
    a = df.index[df['event'].str.contains(val)]
    if a.empty:
        return 'not found'
    elif len(a) > 1:
        return a.tolist()
    else:
        #only one value - return scalar  
        return a.item()


app = st.sidebar

app.title('DAT - Data Analysis Tool')

uploaded_file = app.file_uploader('Upload a VR log file', type=['dat'])
app.button('run')

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, header=None, names=['raw'])
    dataframe[['date','time', 'event']] =  dataframe['raw'].str.split(' ', 2, expand=True)
    st.dataframe(dataframe)

    mat = dataframe.replace(to_replace='None', value=np.nan).dropna()
    mat.drop(['raw'], axis=1, inplace=True)
    st.dataframe(mat)

    # Meal empty array
    meal = []


    """
    ## Placed and dropped items in the scene
    """
    plate = []
    for i in range(len(mat)):
        try:
            if "Placed" in mat['event'][i] and "Dropped" in mat['event'][i+1] and (float(mat['time'][i][-4:]) - float(mat['time'][i+1][-4:])) < 2.0000:
                plate.append(mat['event'][i+1].split()[1])
        except Exception as e:
            print(e)
                
    st.write(plate)

    """
    ## Drink items
    """
    drinks = {}
    for i in range(len(dataframe)):
        if "%" in dataframe['raw'][i]:
                drinks[dataframe['raw'][i].split('_')[-1]] = dataframe['raw'][i]

    # drinks = [dataframe['raw'][i] for i in range(len(dataframe)) if "%" in dataframe['raw'][i]]
    
    st.write(drinks)

    """
    ## Event log of an item
    """
    picked = []
    for i in range(len(mat)):
        if "Picked" in mat['event'][i]:
            picked.append(mat['event'][i].split())
    
    uniquePicked = [i for n, i in enumerate(picked) if i not in picked[:n]]

    item = st.selectbox("", uniquePicked)

    item_log = [i for i in mat['event'] if item[2] in i]

    # if st.checkbox("Show event log"):
    #     st.write(item_log)

    if st.checkbox("show item event log"):
        events = {}
        item_event_index = check(mat, item[2])
        for i in item_event_index:
            events[mat['time'][i]] = mat['event'][i]
        st.write(events)

    """
    ## Picked and Placed back items
    """
    picked_back = {}

    for i in uniquePicked:
        for j in range(len(mat)):
            if i[2] in mat['event'][j]:
                if "Dropped" in mat['event'][j] and "onto" not in mat['event'][j]:
                    picked_back[mat['time'][j]] = i[2]
    

    st.write(picked_back)
    


