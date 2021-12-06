# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pathlib
import random
from functools import reduce
from collections import defaultdict

import pandas as pd
import geopandas as gpd # 설치가 조금 힘듭니다. 어려우시면 https://blog.naver.com/PostView.nhn?blogId=kokoyou7620&logNo=222175705733 참고하시기 바랍니다.
import folium
import shapely 
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sklearn.cluster
import tensorflow as tf  # 설치 따로 필요합니다. https://chancoding.tistory.com/5 참고 하시면 편해요.

#from geoband import API         이건 설치 필요 없습니다.

import pydeck as pdk                  # 설치 따로 필요합니다.
import os

import pandas as pd


import cufflinks as cf                 # 설치 따로 필요합니다.   
cf.go_offline(connected=True)
cf.set_config_file(theme='polar')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'

import numpy as np
from shapely.geometry import Polygon, Point
from numpy import random

import geojson                       # 밑에 Line 84에 추가하여야 하지만 바로 import 안되서 설치 필요

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  # 설치 따로 필요합니다.

# +
import sys
'geopandas' in sys.modules

import os
path = os.getcwd()
path
# -

df_10 = gpd.read_file("MOCT_LINK.json")
df_10.head()

df_12 = pd.read_csv("./혼잡빈도시간강도_total.csv", encoding='cp949')
df_12.head()

# +
# df_12_ = df_12.groupby(['LINK ID', 'ITS LINK ID', '도로등급', '연장(km)', '도로명', '권역'], as_index=False).mean()
# df_12.to_csv('./혼잡빈도시간강도_total.csv', encoding='cp949')

# +
#Pydeck 사용을 위한 함수 정의
import geopandas as gpd 
import shapely # Shapely 형태의 데이터를 받아 내부 좌표들을 List안에 반환합니다. 

def line_string_to_coordinates(line_string): 
    if isinstance(line_string, shapely.geometry.linestring.LineString): 
        lon, lat = line_string.xy 
        return [[x, y] for x, y in zip(lon, lat)] 
    elif isinstance(line_string, shapely.geometry.multilinestring.MultiLineString): 
        ret = [] 
        for i in range(len(line_string)): 
            lon, lat = line_
            string[i].xy 
            for x, y in zip(lon, lat): 
                ret.append([x, y])
        return ret 

def multipolygon_to_coordinates(x): 
    lon, lat = x[0].exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 

def polygon_to_coordinates(x): 
    lon, lat = x.exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 


# -

# ---------------------------------
# ## 1. 2017~2018년 혼잡빈도강도, 혼잡시간강도 분석
#
#
# **목적: 혼잡빈도강도와 혼잡시간빈도를 분석하여 차량이 많은 위치 파악**
#
# **분석 데이터 종류**
# - df_10: MOCT_LINK.json
# - df_12: 혼잡빈도시간강도_total.csv
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

isin_=df_10['ROAD_NAME'].isin(['동해대로', '청학로', '번영로', '중앙로', '장안로', '설악금강대교로', '미시령로', '청대로',
       '온천로', '관광로', '수복로', '중앙시장로', '교동로', '청초호반로', '엑스포로', '조양로', '법대로',
       '설악산로', '동해고속도로(삼척속초)', '장재터마을길', '진부령로', '간성로', '수성로'])
df_10 = df_10.loc[isin_]
df_10

df = df_10
df['coordinate'] = df['geometry'].apply(line_string_to_coordinates) 
df

df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df.head()

df.loc[df['LANES'] == 1, 'WIDTH'] = '2'
df.loc[df['LANES'] == 2, 'WIDTH'] = '3'
df.loc[df['LANES'] == 3, 'WIDTH'] = '4'
df.loc[df['LANES'] == 4, 'WIDTH'] = '4'
df.loc[df['LANES'] == 5, 'WIDTH'] = '5'
df.loc[df['LANES'] == 6, 'WIDTH'] = '5'
df.loc[df['LANES'] == 7, 'WIDTH'] = '5'

df['정규화도로폭'] = df['WIDTH'].apply(int) / df['WIDTH'].apply(int).max()
df['정규화도로폭'] 

df['정규화도로폭'] .unique()

df_12

# +
# 혼합빈도강도 양방향 총 합
data = []

for i in df.LINK_ID:
    data.append([i,sum(df_12[df_12['ITS LINK ID'].apply(str).str.contains(i)].혼잡빈도강도)])
    
data = pd.DataFrame(data).fillna(0)
data.columns = ["LINK_ID", "혼잡빈도강도합"]
result = pd.merge(df, data, on = 'LINK_ID' )

# 혼잡빈도강도 합이 가장 높은 도로
result.iloc[result["혼잡빈도강도합"].sort_values(ascending=False).index].reindex().head()

# +
# result.loc[result['혼잡빈도강도합'] == 0, '혼잡빈도강도합'] = '1'
# -

result['혼잡빈도강도합'].unique()

# +
layer = pdk.Layer('PathLayer', 
                  result, 
                  get_path='coordinate', 
                  get_width='혼잡빈도강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 

center = [128.5918, 38.20701] 

view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 

r.to_html('./혼잡빈도강도_테스트.html')

# +
# 혼합시간강도 양방향 총 합
data_ = []

for i in df.LINK_ID:
    data_.append([i,sum(df_12[df_12['ITS LINK ID'].apply(str).str.contains(i)].혼잡시간강도)])
    
data_ = pd.DataFrame(data_).fillna(0)
data_.columns = ["LINK_ID", "혼잡시간강도합"]
result_ = pd.merge(df, data_, on = 'LINK_ID' )

# 혼잡빈도강도 합이 가장 높은 도로
result_.iloc[result_["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

result_['혼잡시간강도합'].unique()

result_.loc[result_['혼잡시간강도합'] == 0, '혼잡시간강도합'] = '1'

# +
layer = pdk.Layer('PathLayer', 
                  result_, 
                  get_path='coordinate', 
                  get_width='혼잡시간강도합/2', 
                  get_color='[255, 255 * 정규화도로폭 , 120]', 
                  pickable=True, auto_highlight=True 
                 ) 

center = [128.5918, 38.20701] 

view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 

r.to_html('./혼잡시간강도_최종.html')
# -


