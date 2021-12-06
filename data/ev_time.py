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

import sys
'geopandas' in sys.modules

import os
path = os.getcwd()
path

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
            lon, lat = line_string[i].xy
            for x, y in zip(lon, lat):
                ret.append([x, y])
        return ret

def multipolygon_to_coordinates(x):
    lon, lat = x[0].exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]

def polygon_to_coordinates(x):
    lon, lat = x.exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]

df_11 = pd.read_csv('11.평일_일별_시간대별__추정교통량(대체데이터).csv')
df_10= gpd.read_file("./강원도속초시,고성군_상세도로망.json")
# df_11= pd.read_csv("11.광양시_평일_일별_시간대별_추정교통량.csv")

df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates)
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다.
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

df.head()

# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정
# 승용차만 고려
df_11_time11 = df_11[df_11['시간적범위']==11]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time11[df_11_time11['link_id'].apply(str).str.contains(i)]['승용차'])])

df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time11=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time11.iloc[df_10_11_time11["교통량"].sort_values(ascending=False).index].reindex().head()

layer = pdk.Layer( 'PathLayer',
                  df_10_11_time11,
                  get_path='coordinate',
                  get_width='교통량/2',
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
r.to_html()
