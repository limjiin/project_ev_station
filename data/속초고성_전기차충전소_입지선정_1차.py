# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3
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


# -

# ---------------------------------
# # I. EDA
#
# ## 1. 인구현황 분석
#
#
# **목적: 격자 별 인구 현황을 확인**
#
# **분석 데이터 종류**
# - df_08: 08.속초-고성_격자별인구현황.json
#
# **분석 설명**
# - 초록색일 수록 인구가 많이 있으며, 검은색일 수록 인구가 적으며, 색이 칠해지지 않은 곳은 인구 현황 값이 0 이다.
# - 인구현황데이터는 현재 100X100 grid로 나누어져 있으나 추후 분석을 위해 grid의 중심에 해당하는 Point 값 (Central point)을 계산해 주었고, 각각에 고유한 grid id를 부여했다.
# - 따라서 인구 현황을 100X100 point로 설명 할 수 있는 결과를 도출하였다. 
#

df_08= gpd.read_file("08.속초-고성_격자별인구현황.json") # geojson -> json
df_08

# +
df_08['val'] = df_08['val'].fillna(0)

df_08['정규화인구'] = df_08['val'] / df_08['val'].max()

df_08.head()
# -

df_08['coordinates'] = df_08['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type

# +
# 100X100 grid에서 central point 찾기 (multipolygon -> polygon 
# : cent 코드 [[i[0].centroid.coords[0][0],i[0].centroid.coords[0][1]]] -> [[i.centroid.coords[0][0],i.centroid.coords[0][1]]] )
df_08_list = []
df_08_list2 = []
for i in df_08['geometry']:
    cent = [[i.centroid.coords[0][0],i.centroid.coords[0][1]]]
    df_08_list.append(cent)
    df_08_list2.append(Point(cent[0]))
df_08['coord_cent'] = 0
df_08['geo_cent'] = 0
df_08['coord_cent']= pd.DataFrame(df_08_list) # pydeck을 위한 coordinate type
df_08['geo_cent'] = df_08_list2 # geopandas를 위한 geometry type

# 쉬운 분석을 위한 임의의 grid id 부여
df_08['grid_id']=0
idx = []
for i in range(len(df_08)):
    idx.append(str(i).zfill(5))
df_08['grid_id'] = pd.DataFrame(idx)

# 인구 현황이 가장 높은 위치
df_08.iloc[df_08["val"].sort_values(ascending=False).index].reindex().head()

# +
# Make layer
# 사람이 있는 그리드만 추출
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[900, 255*정규화인구, 0, 정규화인구*10000 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
)

# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)

r.to_html()
# -

# ---------------------------------
# ## 2. 자동차 등록 대수 분석
#
#
# **목적: 어느 지역이 가장 많은 자동차가 등록 되어있는지 분석**
#
# **분석 데이터 종류**
# - df_03: 광양시_자동차등록현황_격자(100X100).geojson
#
# **분석 설명**
# - 분석 시간 단축을 위해 차량이 1대 이상 등록된 곳만 필터링 하였다. 
# - 초록색에 가까울 수록 차량이 많이 등록 되었다는 것을 의미하며, 검은색은 초록색에 비해 적게 등록되어있음을 의미한다. 그리드가 없는 곳은 차량이 1대 또는 등록이 되지 않은 곳이다. 

df_03= pd.read_csv("03.속초시-고성군_자동차등록현황.csv", encoding='utf-8')
df_20= gpd.read_file("20.행정경계(읍면동).geojson")

df_03.rename(columns={'총합':'2021'}, inplace=True)
df_03

df_03.rename(columns={'읍면동':'EMD_NM'}, inplace=True)
df_03

df_03=df_03.iloc[df_03[["EMD_NM", "2021"]].mean(axis=1).sort_values(ascending=False).index].reindex()
df_03.columns

df_03 = pd.merge(df_03, df_20, on = "EMD_NM")
df_03

# ### 분석 결과
# - 중마동과 금호동에 자동차 등록이 많이 됨
# - 자동차 등록 위치는 세대 위치와 비슷하므로 자동차 등록대수가 많은 곳에 완속충전소를 세우는 것이 효과적일것으로 판단
# - 중마동과 금호동에 완속충전소를 설치하는 것이 효과적일 것으로 기대
#
#
# -------------------------------------

# ---------------------------------
# ## 3. 전기자동차 등록 대수 분석
#
#
# **목적: 어느 지역이 가장 많은 전기 자동차가 등록 되어있는지 분석**
#
# **분석 데이터 종류**
# - df_06: 06.전기차보급현황(연도별,읍면동별).csv
# - df_20: 20.행정경계(읍면동).geojson
#
# **분석 설명**
# - 지역별/ 연도별 전기차 보금 현황 확인을 위한 전처리를 수행하였다.
# - 지역 geometry 정보를 함께 통합하였다.
#

df_06 = pd.read_csv('./03.속초시-고성군_전기자동차등록현황.csv')
df_06 

# df_06.drop(columns={'기준년도'}, inplace=True)
df_06.rename(columns={'전기차 총합':'2021'}, inplace=True)
df_06.rename(columns={'읍면동':'EMD_NM'}, inplace=True)
df_06.head()

# df_20 = gpd.read_file('./20.행정경계(읍면동).geojson')
df_20

# 전기차 등록 대수 점 수 부여
#년도 별, 행정구역 별, 전기차 보급 추세
list_EV_dist = df_06

list_EV_dist=list_EV_dist.iloc[list_EV_dist[["EMD_NM", "2021"]].mean(axis=1).sort_values(ascending=False).index].reindex()

list_EV_dist.columns

# +
# 2020년 기준으로 가장 많은 비율을 차지하는 광양읍에 전체적으로 점수를 크게 부여할 것
df_EV_ADM = pd.merge(list_EV_dist, df_20, on = "EMD_NM")

#list_EV_dist[["행정구역", "2017","2019","2020"]].mean(axis=1)
df_EV_ADM
# -

# ---------------------------------
# ## 4. 교통량 분석
#
#
# **목적: 주거시간과 업무시간의 교통량 분석**
#
# **분석 데이터 종류**
# - df_10: 10.강원도속초시,고성군_상세도로망.geojson
# - df_11: 11.평일_일별_시간대별__추정교통량(대체데이터).csv"
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

df_11 = pd.read_csv('11.평일_일별_시간대별__추정교통량(대체데이터).csv')
df_10 = gpd.read_file("./10.강원도속초시,고성군_상세도로망.json")
# df_11= pd.read_csv("11.광양시_평일_일별_시간대별_추정교통량.csv")

df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

df.head()

# +
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

# +
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
# -

# ### 분석 결과
#
# - 7시에 교통량이 많은 곳은 주거지역으로 간주하였으며, 중마동, 마동은 주거지역일 것으로 기대

# +
# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정
# 승용차만 고려
df_11_time17 = df_11[df_11['시간적범위']==17]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time17[df_11_time17['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time17=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time17.iloc[df_10_11_time17["교통량"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_11_time17, 
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

# ### 분석 결과
# - 15시에 교통량이 많은 곳은 상업지역 혹은 차량이 가장 많이 지나는 중심 도로 간주하였으며,  광양읍은 업무 중심 지역일 것으로 기대
# - 급속 충전소는 업무지역 설치가 효과적이라 판단
#
# -------------------------------------

# ---------------------------------
# ## 5. 혼잡빈도강도, 혼잡시간강도 분석
#
#
# **목적: 혼잡빈도강도와 혼잡시간빈도를 분석하여 차량이 많은 위치 파악**
#
# **분석 데이터 종류**
# - df_10: 10.강원도속초시,고성군_상세도로망.json
# - df_12: 12.평일_혼잡빈도강도_강원도 속초시, 고성군.csv"
# - df_13: 13.평일_혼잡시간강도_강원도 속초시, 고성군.csv"
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

# ## 데이터 읽기

df_12= pd.read_csv("./12.평일_혼잡빈도강도_강원도 속초시, 고성군.csv")
df_13= pd.read_csv("./13.평일_혼잡시간강도_강원도 속초시, 고성군.csv")

df_12.columns

df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

# ---------------------------------
# ## 1. 2017~2018년 혼잡빈도강도, 혼잡시간강도 분석
#
#
# **목적: 혼잡빈도강도와 혼잡시간빈도를 분석하여 차량이 많은 위치 파악**
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 붉은 색일수록 혼잡빈도시간강도가 높은 것이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

# +
# 혼합빈도강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_12[df_12['link_id'].apply(str).str.contains(i)].혼잡빈도강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡빈도강도합"]
df_10_12=pd.merge(df, df_10_,on = 'link_id' )

# 혼잡빈도강도 합이 가장 높은 도로
df_10_12.iloc[df_10_12["혼잡빈도강도합"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_12, 
                  get_path='coordinate', 
                  get_width='혼잡빈도강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#             mapbox_key = 'sk.eyJ1IjoieW9vYnl1bmdjaHVsIiwiYSI6ImNrd245YnMwZzFiMnEycHBkc2gzbzkzd3AifQ.sc9Gmo56AsAHzJ2B3wCkXg') 
r.to_html()

# +
# 혼합시간강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_13[df_13['link_id'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡시간강도합"]
df_10_13 = pd.merge(df, df_10_,on = 'link_id' )
# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_13, 
                  get_path='coordinate', 
                  get_width='혼잡시간강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 
r.to_html()

# ---------------------------------
# ## 6. 급속충전소 설치가능 장소 필터링
#
#
# **목적: 급속 충전소의 경우 사유지는 제외 해야 하므로 설치 가능 장소 필터링 필요**
#
# **분석 데이터 종류**
# - df_14: 14.소유지정보.geojson
#
# **분석 설명**
# - 사유지를 포함한 임야, 염전, 도로, 철도 용지, 제방, 하천과 같이 설치가 부적절 한 곳을 필터링 한 multipolygone을 시각화하였다.
# - 앞서 도출한 인구현황 100X100 Point 데이터셋에서 설치가능한 장소에 해당하는 point를 추출하였다.

df_14= gpd.read_file("./14.소유지정보.geojson") # geojson -> json

df_14.columns

len(df_14)

df_14.isna().sum()

df_14.head()

df_14[df_14['지목코드'] == '18'].head()

# +
df_14_=df_14[df_14['소유구분코드'].isin(['02','04'])] # 소유구분코드: 국유지, 시/군

df_14_possible=df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['05','07','14','15','16','17',
                             '18', '19', '20','27' ])==False)] # 임야, 염전, 도로, 철도 용지, 제방, 하천, 묘지 제외 

# 07 없음

# geometry to coordinates (multipolygon_to_coordinates -> polygon_to_coordinates)
df_14_possible['coordinates'] = df_14_possible['geometry'].apply(polygon_to_coordinates) 

# 설치가능한 모든 polygone을 multipolygone으로 묶음
from shapely.ops import cascaded_union
boundary = gpd.GeoSeries(cascaded_union(df_14_possible['geometry'].buffer(0.001)))

from geojson import Feature, FeatureCollection, dump
MULTIPOLYGON = boundary[0]

features = []
features.append(Feature(geometry=MULTIPOLYGON, properties={"col": "privat"}))
feature_collection = FeatureCollection(features)
with open('geo_possible.geojson', 'w') as f:
    dump(feature_collection, f)

geo_possible= gpd.read_file("geo_possible.geojson")
# -

# 브로드캐스팅을 이용한 요소합 (평행이동)
# 요소합 진행 후, 마지막 데이터를 list로 형변환
v = np.array([-0.0022, 0.0027])
for i in range(len(df_14_possible["coordinates"])):
    for j in range(len(df_14_possible["coordinates"].iloc[i])):
                   df_14_possible["coordinates"].iloc[i][j] = list(df_14_possible["coordinates"].iloc[i][j] + v)
df_14_possible["coordinates"]

# +
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_14_possible, # 시각화에 쓰일 데이터프레임
                  #df_result_fin[df_result_fin['val']!=0],
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.5918, 38.20701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 


# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
            ) 

    
r.to_html()
# -

# # II. 입지선정지수 개발
#
# ## 1. 지역특성 요소 추출
#
# ### 100X100 Point 중 설치 가능 한 Point 필터링 
# - 100X100 중 설치 가능한 multipolygone에 있는 point를 필터링하는 시간이 굉장히 오래 소요됨(약 1시간)
# - 따라서 이를 모두 계산한 'within_points.csv'를 따로 저장하여 첨부함
# - df_result로 최종 분석 할 데이터셋을 만듦

# +
# 최종 분석 데이터 정제하기

# 개발 가능 한 grid point 찾기
shapely.speedups.enable()
df_result = df_08[['grid_id','val','geometry','coordinates','coord_cent','geo_cent']]
df_result['val'] = df_result['val'].fillna(0)

#굉장히 오래걸림
point_cent= gpd.GeoDataFrame(df_result[['grid_id','geo_cent']],geometry = 'geo_cent')
within_points=point_cent.buffer(0.00000001).within(geo_possible.loc[0,'geometry'])
pd.DataFrame(within_points).to_csv("within_points.csv", index = False)

within_points=pd.read_csv("within_points.csv")
df_result['개발가능'] = 0
df_result['개발가능'][within_points['0']==True] = 1
df_result[df_result['개발가능']==1]
# -

within_points=pd.read_csv("within_points.csv")
df_result['개발가능'] = 0
df_result['개발가능'][within_points['0']==True] = 1
df_result[df_result['개발가능']==1]

df_result['개발가능'].value_counts()

# ### 분석 결과
# - 기본 상식선에서 개발 가능한 토지를 구분하였으나 토지관련 전문가를 통해 더 자세하게 토지를 추가/제거 한다면 더 현실적인 결과가 나올 것이다.
#
# -------------------------------------

# ## 3. 100X100 Point에 자동차등록현황 부여
#
# **목적: 전기자동차 등록대 수 만큼 값을 부여**
#
# **분석 데이터 종류**
# - df_03: 03.자동차보급현황(연도별,읍면동별).csv
# - df_20: 20.행정경계(읍면동).geojson
#
# **분석 설명**
# - 전기자동차를 point에 대입함
#
# **분석 설명**
# - 각 지역에 해당하는 Point를 모두 추출, 전기자동차 보급 현황을 부여하였다. (2020년 등록 대수 만 사용)

# +
ADM_points_ = []
for i in tqdm(range(len(df_03))):
    ADM_points_.append([df_03.loc[i,'EMD_NM'],
                       df_03.loc[i,'2021'],
                        point_cent.buffer(0.00000001).within(df_03.loc[i,'geometry'])])
df_result['자동차등록'] = 0

for i in range(len(ADM_points_)):
    df_result['자동차등록'][ADM_points_[i][2]] = ADM_points_[i][1]
df_result
# -

# ## 4. 100X100 Point에 전기 자동차등록현황 부여
#
# **목적: 전기자동차 등록대 수 만큼 값을 부여**
#
# **분석 데이터 종류**
# - df_06: 06.전기차보급현황(연도별,읍면동별).csv
# - df_20: 20.행정경계(읍면동).geojson
#
# **분석 설명**
# - 전기자동차를 point에 대입함
#
# **분석 설명**
# - 각 지역에 해당하는 Point를 모두 추출, 전기자동차 보급 현황을 부여하였다. (2020년 등록 대수 만 사용)

# +
ADM_points = []
for i in tqdm(range(len(df_EV_ADM))):
    ADM_points.append([df_EV_ADM.loc[i,'EMD_NM'],
                       df_EV_ADM.loc[i,'2021'],
                        point_cent.buffer(0.00000001).within(df_EV_ADM.loc[i,'geometry'])])
df_result['전기자동차등록'] = 0

for i in range(len(ADM_points)):
    df_result['전기자동차등록'][ADM_points[i][2]] = ADM_points[i][1]
df_result
# -

# -----------------------------------------------------------------------------------------------------------
#
# ## 2. 100X100 Point에 교통량, 혼잡빈도강도, 혼잡시간강도 관련 요소 부여
#
# 목적: grid 마다 교통량 관련 요소 부여
#
# 분석 데이터 종류
# - df_11: 평일_일별_시간대별_추정교통량.csv
# - df_12: 평일_전일_혼잡빈도강도.csv
# - df_13: 평일_전일_혼잡시간강도.csv
#
# 분석 설명
# - 각 100X100 Point 마다 07시 교통량, 15시 교통량, 혼잡빈도강도합, 혼잡시간강도합을 부여
# - 각 요소바다 부여하는데 시간이 다소 소요됨 (약 10분)

# +
# grid 마다 11시 교통량 부여
df_10_11_time11_grid = []
df_superset = df_10_11_time11[df_10_11_time11['교통량']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_11_time11_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_11_time11_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_11_time11_grid)):
    id_idx = df_10_11_time11_grid[i][0]
    grids = df_10_11_time11_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['교통량'])])
    except:
        pass

#2시 승용차 혼잡 빈도 관련 정보
try:
    del df_result['교통량_11']
except:
    pass

grid_=pd.DataFrame(grid_list)
grid_.columns = ["grid_id","교통량_11"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_, on = 'grid_id')
df_result[df_result['교통량_11']>0]

# +
# grid 마다 17시 교통량 부여
df_10_11_time17_grid = []
df_superset = df_10_11_time17[df_10_11_time17['교통량']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_11_time17_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_11_time17_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_11_time17_grid)):
    id_idx = df_10_11_time17_grid[i][0]
    grids = df_10_11_time17_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['교통량'])])
    except:
        pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_17"]
grid_혼잡빈도[grid_혼잡빈도['교통량_17']>0]


#15시 승용차 혼잡 빈도 관련 정보
try:
    del df_result['교통량_17']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_17"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['교통량_17']>0]
# -

df_10_12['혼잡빈도강도합']=df_10_12['혼잡빈도강도합'].astype(int)

df_10_12.info()

# grid 마다 혼잡빈도강도 부여
df_10_grid = []
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]
df_superset

df_superset['road_name'].unique()

# +
# grid 마다 혼잡빈도강도 부여
df_10_grid = []
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]
df_superset

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_grid)):
    id_idx = df_10_grid[i][0]
    grids = df_10_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['혼잡빈도강도합'])])
    except:
        pass

#혼잡빈도강도 관련 정보
try:
    del df_result['혼잡빈도강도합']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","혼잡빈도강도합"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['혼잡빈도강도합']>0]
# -

df_10_13['혼잡시간강도합'] = df_10_13['혼잡시간강도합'].astype(int)

# +
# grid 마다 혼잡시간강도합 부여
df_10_grid = []
df_superset = df_10_13[df_10_13['혼잡시간강도합']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_grid)):
    id_idx = df_10_grid[i][0]
    grids = df_10_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['혼잡시간강도합'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['혼잡시간강도합']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","혼잡시간강도합"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['혼잡시간강도합']>0]
# -

# ---------------------------------
# ## 5. 기존 충전소 위치 분석
#
#
# **목적: 기존 충전소가 있는 위치를 분석, 기존 충전소가 커버가능한 범위는 제외하고 분석**
#
# **분석 데이터 종류**
# - df_01: 01.고성군_속초시_충전기설치현황.csv
#
# **분석 설명**
# - 급속 충전소 (Fast-charing Station, FS) 와 완속 충전소(Slow-charging Station, SS) 의 위치를 확인하였다.
#     - 급속: 파란색
#     - 완속: 초록색
# - 급속 충전소와 완속 충전소 주위 500m를 cover가능하다고 가정하였다.
# - 기존 충전소가 cover 가능한 point를 구분하였다.

df_01 = pd.read_csv('./01.고성군_속초시_충전기설치현황.csv')
df_01.head()

df_01.info()

# 기존 완속/ 급속 충전소가 커버하는 위치 제거
df_01_geo = []
for i in range(len(df_01)):
    df_01_geo.append([df_01.loc[i,'충전소명'],Point(df_01.loc[i,'lon'],df_01.loc[i,'lat']).buffer(0.003)])
#df_01[df_01['급속/완속']=='완속']
df_01_geo = pd.DataFrame(df_01_geo)
df_01_geo.columns = ["충전소명", "geometry"]
df_01_geo = pd.merge(df_01, df_01_geo, on = '충전소명')
df_01_geo['coordinates'] = df_01_geo['geometry'].apply(polygon_to_coordinates) 
df_01_geo = pd.DataFrame(df_01_geo)

df_01_geo

# +
#point_cent.buffer(0.00000001).within(df_EV_ADM.loc[i,'geometry'])])


center = [127.696280, 34.940640] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
layer1 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='급속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[50, 50, 200,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='완속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[100, 200, 100,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

scatt1 = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=50,
    get_fill_color='[50, 50, 200]',
    pickable=True)

scatt2 = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='완속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=50,
    get_fill_color='[100, 200, 100]',
    pickable=True)


r = pdk.Deck(layers=[layer1,scatt1, layer2,scatt2], initial_view_state=view_state)
            # mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 
   
r.to_html()

# +
#Fast-charging Station

#시간이 많이 걸립니다.
df_01_FS = df_01_geo[df_01_geo['급속/완속']=='급속']
FS_points = []
for i in tqdm(range(len(df_01_FS))):
    try:
        FS_points.append(point_cent.buffer(0.00000001).within(df_01_FS.loc[i,'geometry']))
    except: 
        pass
df_result['FS_station'] = 0
for i in range(len(FS_points)):
    df_result['FS_station'][FS_points[i]] = 1

#Slow-charging Station
df_01_SS = df_01_geo[df_01_geo['급속/완속']=='완속']    
SS_points = [] 
for i in tqdm(range(len(df_01_geo))):
    try:
        SS_points.append(point_cent.buffer(0.00000001).within(df_01_SS.loc[i,'geometry']))
    except:
        pass

df_result['SS_station'] = 0
for i in range(len(SS_points)):
    df_result['SS_station'][SS_points[i]] = 1

df_result.head()
# -

# ### 분석 결과
# - 전기자동차 충전소가 cover가능한 거리는 임의로 정한 것으로 바뀔 필요가 있다.
# - 현재는 전기자동차가 많이 등록되어 있지 않아 거리로 임의로 정하였지만 대수가 많아 진다면 전기자동차 등록 대수에 따라 cover 가능한 거리가 바뀌어야 할 것이다.
# - 즉, 전기자동차 등록이 많은 곳은 cover 가능한 거리를 줄여 더 많은 곳에 충전소를 설치해야 한다.
# -------------------------------------
#
#

# ---------------------------------
# ## 6. 전기자동차 충전소 위치선정에 대한 영향 요소 분석 및 상관관계 분석
#
# 본 지원자는 전기자동차 충전소 위치 선정을 최적화 문제로 풀 것이다. 이를 위해 목적함수가 필요하며 다음과 같은 기준으로 식을 세웠다. 
#
# **가정**
# 1. 전기자동차 충전소 위치는 인구현황, 교통량(11시, 17시), 혼잡빈도강도, 혼잡시간강도 만 고려하여 위치를 선정한다.
# 2. 기존 설치된 전기자동차 충전소는 위 고려사항을 충분히 고려하여 만들어진 곳이다.
# 3. 전기자동차 충전소는 전방 약 500m를 커버할 수 있다. 
#
# **분석 방법**
#
# - 고려되는 모든 변수들은 정규화 하였다.
# - 선형회귀분석을 이용해 현재 제공받은 데이터로부터 전기자동차 충전소 위치에 영향을 주는 요소의 관계를 분석하였다.(Linear Regrssion)
#
# - 이때 급속 충전소와 완속 충전소 각각을 따로 분석하였다.  
#
# - **분석 Input**: (정규화된) 인구현황, 11시 교통량, 17시 교통량, 혼잡빈도강도, 혼잡시간강도
#
# - **분석 Output**: 고려되는 요소들과 각 충전소 사이의 상관계수 
#
#

df_result.head()

df_result['교통량_11'].unique()

df_result['정규화_인구'] = df_result['val'] / df_result['val'].max()
df_result['정규화_교통량_11'] = df_result['교통량_11'] / df_result['교통량_11'].max()
df_result['정규화_교통량_17'] = df_result['교통량_17'] / df_result['교통량_17'].max()
df_result['정규화_혼잡빈도강도합'] = df_result['혼잡빈도강도합'] / df_result['혼잡빈도강도합'].max()
df_result['정규화_혼잡시간강도합'] = df_result['혼잡시간강도합'] / df_result['혼잡시간강도합'].max()
df_result['정규화_자동차등록'] = df_result['자동차등록'] / df_result['자동차등록'].max()
df_result['정규화_전기자동차등록'] = df_result['전기자동차등록'] / df_result['전기자동차등록'].max()

# +
# 급속/ 완속 관련 objective function 만들기
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

df_LR = df_result
X = df_LR[["정규화_인구", "정규화_교통량_11","정규화_교통량_17", "정규화_혼잡빈도강도합","정규화_혼잡시간강도합", "정규화_자동차등록","정규화_전기자동차등록"]]
y = df_LR["FS_station"]
regr = linear_model.LinearRegression()
regr.fit(X, y)
FS_coeff = regr.coef_
print('급속충전소 Intercept: ', regr.intercept_)
print('급속충전소 Coefficients: \n', FS_coeff)

# ## +-0.3 기준
## 양 : 정규화 인구, 혼잡시간강도
## 음 : 교통량 17, 혼잡빈도강도

# +
df_LR = df_result
X = df_LR[["정규화_인구", "정규화_교통량_11","정규화_교통량_17", "정규화_혼잡빈도강도합","정규화_혼잡시간강도합", "정규화_자동차등록", "정규화_전기자동차등록"]]
y = df_LR["SS_station"]
regr = linear_model.LinearRegression()
regr.fit(X, y)
SS_coeff = regr.coef_
print('완속충전소 Intercept: ', regr.intercept_)
print('완속충전소 Coefficients: \n', SS_coeff)

# ## +-0.3 기준
## 양 : 정규화 인구, 혼잡시간강도
# -

df_result['w_FS'] = 0 
df_result['w_FS'] = (FS_coeff[0]*df_result['정규화_인구']+
                     FS_coeff[1]*df_result['정규화_교통량_11']+
                     FS_coeff[2]*df_result['정규화_교통량_17']+
                     FS_coeff[3]*df_result['정규화_혼잡빈도강도합']+
                     FS_coeff[3]*df_result['정규화_혼잡시간강도합']+
                     FS_coeff[3]*df_result['정규화_자동차등록']+
                     FS_coeff[3]*df_result['정규화_전기자동차등록']
                    )

df_result['w_SS'] = 0 
df_result['w_SS'] = (SS_coeff[0]*df_result['정규화_인구']+
                     SS_coeff[1]*df_result['정규화_교통량_11']+
                     SS_coeff[2]*df_result['정규화_교통량_17']+
                     SS_coeff[3]*df_result['정규화_혼잡빈도강도합']+
                     SS_coeff[3]*df_result['정규화_혼잡시간강도합']+
                     SS_coeff[3]*df_result['정규화_자동차등록']+
                     SS_coeff[3]*df_result['정규화_전기자동차등록']
                    )

try:    
    df_result[['grid_id','geometry',
               '정규화_인구','정규화_교통량_11','정규화_교통량_17',
               '정규화_혼잡빈도강도합', '정규화_혼잡시간강도합', '정규화_자동차등록', '정규화_전기자동차등록',
               'w_FS','w_SS','개발가능','FS_station','SS_station']].to_file("df_result.geojson", driver="GeoJSON")
except:
    pass


# ### 분석 결과
# - 급속충전소와 완속충전소는 고려되는 요소들의 영향이 차이가 있다.
# - 급속충전소 경우, 정규화_혼잡시간강도합에 가장 많은 영향을 받았고, 완속충전소 경우, 정규화_인구에 가장 많은 영향을 받았다. 
# - 이 결과는 차량이 많이 다는 곳에 급속 충전소를 설치하고, 주거 공간이 많이 있는 곳에 완속 충전소를 설치했다고 해석할 수 있다.
# - 완속 충전소의 경우 급속충전소가 가장 크게 영향을 받은 정규화_혼잡시간강도합의 변수값이 양수로 분석되었으며, 이는 혼잡한 곳에도 완속충전소가 설치된 것을 의미한다. 
# - 위 분석 결과는 상식적인 부분과 잘 맞는다.
# - 따라서 위 결과를 이용하여 추후 최적화 모델의 목적함수 변수 값으로 사용하기 위해 w_fs, w_ss로 가중치값을 계산하였다. 
# -------------------------------------
#
#

# # III. 최적화 문제 정의 및 해결
#
# ## 전기자동차 충전소 위치 선정 최적화 모델 정의 및 결과
#
# ### 최적화 모델 : Maximal Covering Location Problem (MCLP)
#
# - 정의: MCLP는 최대지역커버문제로, 설비가 커버하는 수요 (covered demand)의 합을 최대화 하면서 주어진 K개의 설비를 세울 위치를 선정하는 문제 
# - 가정
#
#     - 설비의 위치가 수요 발생 지점으로부터 일정 거리 Residual 이내에 수요를 커버함. 
#     - 이때 거리 Residual은 커버리지 거리(covered distance) 라고 함.
#     - 커버되지 못한 수여는 서비스를 받지 못하는 수요가 아니라 서비스를 받긴 하지만 서비스 받는 설비로 부터의 거리가 커버리지 밖에 있어 만족할 만한 서비스 수준을 제공받지 못하는 수요를 의미
#     
#     
# - Mathematical statement
#
#
#     - i : 수요 포인트 index
#     - j : 설비 후보지역 index
#     - I : 수요 포인트 집합
#     - J : 설비 후보지역 집합
#     - K : 총 설치해야 하는 설비 개수
#     - x : 설비 후보 지역 중 위치 j에 설비가 설치되면 1, 그렇지 않으면 0
#     - y : 적어도 하나의 설비로 그 포인트가 커버가 되면 1, 그렇지 않으면 0
#     
#     
# - Formulation
#
# $$
# \begin{align*}
# &\text{maximize} \sum_{i\in I} w_i y_i ...(1) \\
# \text{s.t.} \quad & y_i \le \sum_{j\in N_i}x_j \qquad for \quad all \quad i\in I ... (2)\\
# &\sum_{j\in J}x_j = K ... (3)\\
# &x_j, y_i \in \{0,1\} \qquad for \quad all \quad i\in I,j\in J 
# \end{align*}
# $$
#     
#
#     -(1) : 목적함수, 가중치 w인 수요 포인트를 최대한 많이 커버하게 해라
#     -(2) : 수요포인트 i는 설비 후보 지역이 커버하는 거리안에서 적어도 하나 이상의 설비로 부터 커버가 된다. 
#     -(3) : 총 설치할 설비는 K개 이다.
#

# +
def generate_candidate_sites(points,M=100):
    '''
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def generate_candidate_sites(df_result_fin,M=100):
    from shapely.geometry import Polygon, Point
    sites = []
    idx=np.random.choice(np.array(range(0,len(df_result_fin))), M)
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.iloc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.iloc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def generate_candidate_sites(df_result_fin,Weight,M=100):
    sites = []
    idx = df_result_fin.sort_values(by = Weight, ascending = False).iloc[1:M].index
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.loc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.loc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

from scipy.spatial import distance_matrix
def mclp(points,K,radius,M,df_result_fin,w,Weight):

    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    import time
    start = time.time()
    sites = generate_candidate_sites(df_result_fin,Weight,M)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points,sites)
    mask1 = D<=radius
    D[mask1]=1
    D[~mask1]=0

    from mip import Model, xsum, maximize, BINARY

    # Build model
    m = Model("mclp")
    # Add variables

    x = [m.add_var(name = "x%d" % j, var_type = BINARY) for j in range(J)]
    y = [m.add_var(name = "y%d" % i, var_type = BINARY) for i in range(I)]


    m.objective = maximize(xsum(w[i]*y[i] for i in range (I)))

    m += xsum(x[j] for j in range(J)) == K

    for i in range(I):
        m += xsum(x[j] for j in np.where(D[i]==1)[0]) >= y[i]

    m.optimize()
    
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objective_value)

    solution = []
    for i in range(J):
        if x[i].x ==1:
            solution.append(int(x[i].name[1:]))
    opt_sites = sites[solution]
            
    return opt_sites,m.objective_value

# +
import pathlib
import random
from functools import reduce
from collections import defaultdict

import pandas as pd
import geopandas as gpd
import folium
import shapely
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#import xgboost
import sklearn.cluster
import tensorflow as tf

#from geoband import API

import pydeck as pdk
import os

import pandas as pd

import cufflinks as cf 
cf.go_offline(connected=True)
cf.set_config_file(theme='polar')
#import deckgljupyter.Layer as deckgl

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'

import numpy as np
from shapely.geometry import Polygon, Point
from numpy import random

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  
# -

df_result

# df_test=gpd.read_file('df_result.geojson')
df_test = df_result

# +
# 100X100 grid에서 central point 찾기
df_list = []
df_list2 = []

for i in df_test['geometry']:
    cent = [[i.centroid.coords[0][0], i.centroid.coords[0][1]]]
    df_list.append(cent)
    df_list2.append(Point(cent[0]))
# -

df_test['coord_cent'] = 0
df_test['geo_cent'] = 0
df_test['coord_cent']= pd.DataFrame(df_list) # pydeck을 위한 coordinate type
df_test['geo_cent'] = df_list2 # geopandas를 위한 geometry type
df_test

# +
df_result_fin = df_test[(df_test['개발가능']==1)
                          &(df_test['FS_station']!=1)]
df_result_fin

points = []
for i in df_result_fin['coord_cent'] :
    points.append(i)

w= []
for i in df_result_fin['w_FS'] :
    w.append(i)

radius = radius = (1/88.74/1000)*500   
K = 50
M = 5000

opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_FS')

df_opt_FS= pd.DataFrame(opt_sites_org)
df_opt_FS.columns = ['lon', 'lat']
df_opt_FS

# +
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_14_possible, # 시각화에 쓰일 데이터프레임
                  #df_result_fin[df_result_fin['val']!=0],
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer2 = pdk.Layer( 'PathLayer', 
                  df_10_12, 
                  get_path='coordinate', 
                  get_width='혼잡빈도강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120,140]', 
                  pickable=True, auto_highlight=True 
                 ) 

# Set the viewport location 
center = [128.574141, 38.201027] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[50, 50, 200]',
    pickable=True)

opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[255, 255, 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[layer2,layer, scatt,opt], initial_view_state=view_state)
            # mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 

r.to_html('개발가능_전기차충전소_급속.html')
# -

# - 노란색 : 최적화된 입지 선정
# - 파란색 : 기존 위치 선정

# +
# Set the viewport location 
center = [128.574141, 38.201027] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[50, 50, 200]',
    pickable=True)

opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[255, 255, 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[scatt,opt], initial_view_state=view_state)
            # mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 

r.to_html('개발가능_전기차충전소_급속_test.html')

# +
# 완속 충전소
df_result_fin = df_test[(df_test['SS_station']!=1)]
df_result_fin

points = []
for i in df_result_fin['coord_cent'] :
    points.append(i)

w= []
for i in df_result_fin['w_SS'] :
    w.append(i)

radius = (1/88.74/1000)*500    
K = 50
M = 5000

opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_SS')

df_opt_SS= pd.DataFrame(opt_sites_org)
df_opt_SS.columns = ['lon', 'lat']
df_opt_SS

# +
# Make layer 
# 변수명
layer1 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*정규화인구, 0 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer2 = pdk.Layer( 'PathLayer', 
                  df_10_11_time7, 
                  get_path='coordinate', 
                  get_width='교통량/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 

layer3 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_14_possible, # 시각화에 쓰일 데이터프레임
                  #df_result_fin[df_result_fin['val']!=0],
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.574141, 38.201027] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

# 기존 충전소 위치
scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='완속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[100, 200, 100,140]',
    pickable=True)

# 최적화 충전소 위치
opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_SS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[255, 255, 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[layer1,layer2,layer3,scatt,opt], initial_view_state=view_state)
            #mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 
    
r.to_html('개발가능_전기차충전소_완속.html')
# -

# - 노란색 : 최적화된 입지 선정
# - 초록색 : 기존 위치 선정

# +
# Set the viewport location 
center = [128.574141, 38.201027] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

# 기존 충전소 
scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='완속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[100, 200, 100,140]',
    pickable=True)

# 최적화 충전소
opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_SS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[255, 255, 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[scatt,opt], initial_view_state=view_state)
            #mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 
    
r.to_html('개발가능_전기차충전소_완속_test.html')

# +
# Set the viewport location 
center = [128.574141, 38.201027] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

opt1 = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[255, 255, 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

opt2 = pdk.Layer(
    'ScatterplotLayer',
    df_opt_SS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[0, 255, 255]',
    get_line_color = '[100, 100, 100]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[opt1,opt2], initial_view_state=view_state)
            # mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 

r.to_html('개발가능_전기차충전소_급완속_test2.html')
# -

df_opt_FS['충전소구분']='급속'
df_opt_SS['충전소구분']='완속'
pd.concat([df_opt_FS, df_opt_SS]).to_csv("정류장결과.csv", index=False)





