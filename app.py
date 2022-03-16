#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  14 19:07:31 2022
@author: lucky verma
"""

from enum import unique
import time
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import sys
import os
import cv2
import pickle
import tempfile

st.set_page_config(
    page_title="FoodVR",
     page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded", )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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

apps = ['Dat file Analyzer', 'Video Analyzer']

add_selectbox = st.sidebar.selectbox(
    "Select an App",
    apps
)

if add_selectbox == 'Dat file Analyzer':

    st.title('DAT - Data Analysis Tool')

    uploaded_file = st.file_uploader('Upload a VR log file', type=['dat'])
    st.button('run')

    col1, col2 = st.columns(2)


    if uploaded_file is not None:
        
        # Meal empty array
        meal = []

        dataframe = pd.read_csv(uploaded_file, header=None, names=['raw'])
        with col1:
            st.markdown("##### Raw Log")
            st.dataframe(dataframe)
            dataframe[['date','time', 'event']] =  dataframe['raw'].str.split(' ', 2, expand=True)

            mat = dataframe.replace(to_replace='None', value=np.nan).dropna()
            mat.drop(['raw'], axis=1, inplace=True)
        
            # with st.expander("See explanation"):
            st.markdown("##### This table shows events in the formatted manner")
            st.dataframe(mat)

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
                        drinks[dataframe['raw'][i].split('_')[-1]] = dataframe['raw'][i].split('_', 1)[1][:-4]

            # drinks = [dataframe['raw'][i] for i in range(len(dataframe)) if "%" in dataframe['raw'][i]]
            
            st.write(drinks)

        with col2:
            """
            ## Event log of an item
            """
            picked = []
            for i in range(len(mat)):
                if "Picked" in mat['event'][i]:
                    picked.append(mat['event'][i].split())
            
            uniquePicked = [i for n, i in enumerate(picked) if i not in picked[:n]]

            item = st.selectbox("Select an Item from the dropdown", uniquePicked)

            item_log = [i for i in mat['event'] if item[2] in i]

            # if st.checkbox("Show event log"):
            #     st.write(item_log)

            # if st.checkbox("show item event log"):
            events = {}
            item_event_index = check(mat, item[2])
            for i in item_event_index:
                events[mat['time'][i]] = mat['event'][i]
            st.write(events)

            """
            ## Picked and Placed items
            """
            picked_placed = {}

            for i in uniquePicked:
                for j in range(len(mat)):
                    if i[2] in mat['event'][j]:
                        if "Dropped" in mat['event'][j]:
                            for k in mat["event"].tolist():
                                if f"{i[2]} onto" in k:
                                    picked_placed[mat['time'][j]] = i[2]
            
            st.write(picked_placed)

            """
            ## Placed back items
            """
            placed_back = {}

            temp_picked_index = []
            for i in uniquePicked:
                for j in range(len(mat)):
                    if i[2] in mat['event'][j]:
                        if "Dropped" in mat['event'][j]:
                            for k in mat["event"].tolist():
                                if f"{i[2]} onto" in k:
                                    temp_picked_index.append(i)

            def list_diff(a, b):
                return [x for x in a if x not in b]

            temp = list_diff(uniquePicked, temp_picked_index)

            for i in temp:
                for j in range(len(mat)):
                    if i[2] in mat['event'][j]:
                        if "Dropped" in mat['event'][j]:
                                placed_back[mat['time'][j]] = i[2]
            
            st.write(placed_back)
            


elif add_selectbox == 'Video Analyzer':
    
    st.title('Video - Data Analysis Tool')

    uploaded_video_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mkv'])

    # col1, col2 = st.columns(2)

    if uploaded_video_file is not None:
        
        def draw_flow(img, flow, step=16):

            h, w = img.shape[:2]
            y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
            fx, fy = flow[y,x].T

            lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

            for (x1, y1), (_x2, _y2) in lines:
                cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

            return img_bgr


        def draw_hsv(flow):

            h, w = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]

            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx*fx+fy*fy)

            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[...,0] = ang*(180/np.pi/2)
            hsv[...,1] = 255
            hsv[...,2] = np.minimum(v*4, 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            return bgr


        def rescale_frame(frame, percent=75):
            width = int(frame.shape[1] * percent/ 100)
            height = int(frame.shape[0] * percent/ 100)
            dim = (width, height)
            return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

        # Temporary file to store the uploaded video
        
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video_file.read())
        cap = cv2.VideoCapture(tfile.name) # tempfile.name added to the path
        # cap.set(cv2.CAP_PROP_FPS, 30)
        cap.isOpened()

        try:
            suc, prev = cap.read()
            prev = rescale_frame(prev, percent=15)
            prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        except:
            pass

        rawOuput = []

        # my_bar = st.progress(0)

        if st.button('run'):
            try:
                while True:

                    suc, img = cap.read()
                    img = rescale_frame(img, percent=15)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # start time to calculate FPS
                    start = time.time()


                    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    rawOuput.append([mag, ang])
                    prevgray = gray


                    # End time
                    end = time.time()
                    # calculate the FPS for current frame detection
                    fps = 1 / (end-start)

                    print(f"{fps:.2f} FPS")

                    # cv2.imshow('flow', draw_flow(gray, flow))
                    # cv2.imshow('flow HSV', draw_hsv(flow))


                    key = cv2.waitKey(5)
                    if key == ord('q'):
                        break

            except:
                pass

            # with open('denseOutput.pkl', 'wb') as f:
            #     pickle.dump(rawOuput, f)

            cap.release()
            cv2.destroyAllWindows()

            # st.write(rawOuput)
            df = pd.DataFrame(rawOuput, columns=['mag', 'ang'])

            df['mean'] = df['mag'].apply(lambda x: np.mean(x))
            df['std'] = df['mag'].apply(lambda x: np.std(x))

            meanList = df['mean'].tolist()
            stdList = df['std'].tolist()

            mean_per_second = np.add.reduceat(meanList, np.arange(0, len(meanList), 30))
            std_per_second = np.add.reduceat(stdList, np.arange(0, len(stdList), 30))

            st.area_chart(pd.DataFrame(std_per_second).fillna(method="ffill"))
            st.area_chart(pd.DataFrame(mean_per_second).fillna(method="ffill"))

    
    st.header('TODO:')
    st.write('- Decraese circles/lines')
    st.write('- Make it more faster')
    st.write('- try croping the video')
    st.write('- try removing the soft cap')