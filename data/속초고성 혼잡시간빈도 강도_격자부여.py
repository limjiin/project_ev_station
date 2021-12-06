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

# ## 데이터 읽기

df_10_12 = pd.read_csv("./100_100_빈도.csv")
df_10_13 = pd.read_csv("./100_100_시간.csv")
df_12 = pd.read_csv("./혼잡빈도시간강도_total.csv", encoding = 'cp949')

# ---------------------------------
# # I. EDA
#
# ## 1. 인구현황 분석
#
#
# **목적: 격자 별 인구 현황을 확인**
#
# **분석 데이터 종류**
# - df_08: 격자별인구현황.csv"
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
# ## 5. 혼잡빈도강도, 혼잡시간강도 분석
#
#
# **목적: 혼잡빈도강도와 혼잡시간빈도를 분석하여 차량이 많은 위치 파악**
#
# **분석 데이터 종류**
# - df_10: 상세도로망
# - df_12: 평일_전일_혼잡빈도강도.csv"
# - df_13: 평일_전일_혼잡시간강도.csv"
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

df_10 = gpd.read_file("MOCT_LINK.json")
df_10.head()

df_12 = pd.read_csv("./혼잡빈도시간강도_total.csv", encoding='cp949')
df_12.head()

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
df

df_12

# +
# 혼합빈도강도 양방향 총 합
df_10_ = []

for i in df.LINK_ID:
    df_10_.append([i,sum(df_12[df_12['ITS LINK ID'].apply(str).str.contains(i)].혼잡빈도강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["LINK_ID", "혼잡빈도강도합"]
df_10_12 = pd.merge(df, df_10_, on = 'LINK_ID' )

# 혼잡빈도강도 합이 가장 높은 도로
df_10_12.iloc[df_10_12["혼잡빈도강도합"].sort_values(ascending=False).index].reindex().head()
# -

df_10_12['혼잡빈도강도합'].unique()

df_10_12.loc[df_10_12['혼잡빈도강도합'] == 0, '혼잡빈도강도합'] = '1'

# +
layer = pdk.Layer('PathLayer', 
                  df_10_12, 
                  get_path='coordinate', 
                  get_width='혼잡빈도강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 

center = [128.5918, 38.20701] 

view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=12
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 

r.to_html()

# +
# 혼합시간강도 양방향 총 합
df_10_ = []

for i in df.LINK_ID:
    df_10_.append([i,sum(df_12[df_12['ITS LINK ID'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["LINK_ID", "혼잡시간강도합"]
df_10_13 = pd.merge(df, df_10_, on = 'LINK_ID' )

# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

df_10_13.loc[df_10_13['혼잡시간강도합'] == 0, '혼잡시간강도합'] = '1'

layer = pdk.Layer( 'PathLayer', 
                  df_10_13, 
                  get_path='coordinate', 
                  get_width='혼잡시간강도합/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.20701] 
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=12
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
# - df_14: 광양시_소유지정보.csv"
#
# **분석 설명**
# - 사유지를 포함한 임야, 염전, 도로, 철도 용지, 제방, 하천과 같이 설치가 부적절 한 곳을 필터링 한 multipolygone을 시각화하였다.
# - 앞서 도출한 인구현황 100X100 Point 데이터셋에서 설치가능한 장소에 해당하는 point를 추출하였다.

df_14= gpd.read_file("./14.소유지정보.geojson") # geojson -> json

df_14.columns

len(df_14)

df_14.isna().sum()

df_14.head()

# +
df_14_=df_14[df_14['소유구분코드'].isin(['02','04'])] # 소유구분코드: 국유지, 시/군

df_14_possible=df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['05','07','14','15','16','17',
                             '18','20','27' ])==False)] # 임야, 염전, 도로, 철도 용지, 제방, 하천 제외 


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

# +
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_result, # 시각화에 쓰일 데이터프레임
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

df_10_12['혼잡빈도강도합']=df_10_12['혼잡빈도강도합'].astype(int)

df_10_12.info()

# grid 마다 혼잡빈도강도 부여
df_10_grid = []
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]
df_superset


df_superset['ROAD_NAME'].unique()

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

df_10_13['혼잡시간강도합']=df_10_13['혼잡시간강도합'].astype(int)

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
# -

result.loc[result['혼잡빈도강도합'] == 0, '혼잡빈도강도합'] = '1'

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


