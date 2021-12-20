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

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  # 설치 따로 필요합니다.

# +
# 머신러닝을 위한 모듈 임포트
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.cluster
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
import lightgbm as lgb
# import xgboost
import pathlib
# from functools import reduce
# from collections import defaultdict
# from IPython.display import display
import os

import cufflinks as cf 
cf.go_offline(connected=True)
cf.set_config_file(theme='polar')
from mip import Model, xsum, maximize, BINARY  
# import jenkspy
# -

sdf_00 = pd.read_csv('./input/00.소상공인시장진흥공단_상가(상권)정보_강원_202106.csv')
sdf_01 = pd.read_csv('./input/01.속초-고성_관광명소데이터(시군구_행정동_100미터격자코드포함).csv', encoding = 'cp949')
sdf_02 = pd.read_csv('./input/02.상권_gid_추가.csv', encoding='utf-8')
# store_gid = pd.read_csv('./store_gid.csv')
sdf_03 = pd.read_csv('./input/03.관광지별_관광소요시간_집계_데이터.csv', encoding='utf-8')
sdf_04 = pd.read_csv('./input/04.속초-고성_년도별읍면동_유동인구.csv')
sdf_05 = gpd.read_file('./input/05.법정경계(읍면동).geojson')
# df01 = pd.read_csv('./df_상권_grid_id부여.csv', index_col=None)
sdf_06 = pd.read_csv('./input/06.연별_기초지자체_네비목적지_검색건수_한국관광데이터랩.csv', encoding='utf-8')
# df01_관광지 = pd.read_csv('./df01_관광지.csv', encoding='cp949')
df06 = pd.read_excel('./input/07.주요관광지점_입장객_한국문화관광연구원.xlsx')
# df = pd.read_csv("df01_관광지_네비_방문객.csv", encoding='utf-8')
# df = pd.read_csv("상권_최종.csv", encoding='utf-8')
# df_merged = pd.read_csv('./df_result.csv', encoding='utf-8')
# df_result = pd.read_csv('./동_유동인구.csv')
df_08= gpd.read_file('./input/08.속초-고성_격자별인구현황.json')
within_points=pd.read_csv('./input/within_points.csv',encoding='utf-8')

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
# ## 1. 유동인구분포
#
#
# **목적**
#
# **분석방법**
# - 격자별 상권 밀집도: 격자에 포함된 상가 수 / 동단위 총 상가 수
# - 격자별 유동인구: 읍면동 유동인구 * 격자별 상권 밀집도 
#
# **분석 데이터 종류**
# - 읍면동 유동인구 현황
# - 속초고성 상권정보
# - 속초고성 100X100격자지도
#
# **분석 설명**
# - 인구수에 따라 색이 다르게 시각화

# ---------------------------------
# ## 2. 상권정보 분석
#
# **목적: 격자 별 상권 현황을 확인**
#
# **분석 데이터 종류**
# - sdf_00 : '01.소상공인시장진흥공단_상가(상권)정보_강원_202106.csv'
#
# **분석 설명**
# - 빨간색일 수록 점포 수가 많이 있으며, 노란색일 수록 점포 수가 적으며, 색이 칠해지지 않은 곳은 점포 수 값이 0 이다.
# - 상권현황데이터는 각 점포의 위치를 나타내므로 (geometry: Point type) 이를 100X100 grid로 나눈 국가지점번호 기준으로 분류하였다.
# - 인구현황(df_08)에서 부여된 고유한 grid id를 국가지점번호 index를 통해 mapping 작업하였다.
# - 따라서 상권 현황을 타 지표와 동일한 100X100 point로 설명 할 수 있는 결과를 도출하였다. 
#

# +
# 강원 상권 정보 현황 (소상공인시장진흥공단)
sdf_00 = pd.read_csv('./input/00.소상공인시장진흥공단_상가(상권)정보_강원_202106.csv')

# 지역 조건 부여
con4 = sdf_00['시군구명'] == '속초시'
con5 = sdf_00['시군구명'] == '고성군'

# 업종 조건 부여
con1 = sdf_00['상권업종대분류명'] == '숙박'
con2 = sdf_00['상권업종대분류명'] == '음식'
con3 = sdf_00['상권업종대분류명'] == '관광/여가/오락'

# 필요한 열만 추출
sdf_00 = sdf_00.loc[(con4 | con5) & (con1 | con2 | con3), ['상가업소번호', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명', '시군구명', '법정동명', '지번주소', '도로명주소','경도', '위도']]

#정렬
sdf_00.sort_values(by=["상권업종대분류명", "상권업종중분류명"], ascending=[True, True], inplace=True)
sdf_00['상권업종대분류명'] = sdf_00['상권업종대분류명'].str.replace('관광/여가/오락','여가/오락')

# 중복된 행 제거
sdf_00 = sdf_00.drop_duplicates(subset=None, 
                          keep='first', 
                          inplace=False, 
                          ignore_index=False)

# 국가지점번호 파일 불러오기
sdf_02 = pd.read_csv('./input/02.상권_gid_추가.csv', encoding='utf-8')
sdf_02 = sdf_02[['상가업소번호', 'gid']]
sdf_02 = sdf_02.drop_duplicates()
sdf_02.set_index('상가업소번호', inplace=True)

# 상권 파일에 gid 추가
sdf_00['gid'] = sdf_00['상가업소번호'].apply(lambda x: sdf_02.loc[x,'gid'])

# 상권 현황이 가장 높은 위치(격자) top 10
sdf_00.groupby(['gid','법정동명'])['상가업소번호'].count().sort_values(ascending=False).head(10)
# -

# 상권 현황이 가장 높은 읍면동 top 10
sdf_00.groupby(['법정동명'])['상가업소번호'].count().sort_values(ascending=False).head(5)

# +
# ScatterplotLayer 시각화
layer = pdk.Layer(
    'ScatterplotLayer',
    sdf_00,
    get_position='[경도, 위도]',
    get_radius=30,
    get_fill_color='[255, 255, 255]',
    pickable=True,
    auto_highlight=True
)

center = [128.48966249, 38.26924467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=10)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.show()

# +
# HeatmapLayer 시각화
layer = pdk.Layer(
    'HeatmapLayer',
    sdf_00,
    get_position='[경도, 위도]'
)

center = [128.48966249, 38.26924467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=10)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.show()

# +
# GridLayer 시각화

layer = pdk.Layer(
    'ScreenGridLayer', # 대용량 데이터의 경우 'GPUGridLayer' , screentype 'ScreenGridLayer'
    sdf_00,
    get_position='[경도, 위도]',
    pickable=True,
    auto_highlight=True
)
layer.cellSizePixels = 10 # screen 사이즈 조정, default 100

center = [128.48966249, 38.26924467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=8)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.to_html('./상권현황.html')

# +
# GridLayer 시각화

layer1 = pdk.Layer(
    'ScreenGridLayer', # 대용량 데이터의 경우 'GPUGridLayer' , screentype 'ScreenGridLayer'
    sdf_00,
    get_position='[경도, 위도]',
    pickable=True,
    auto_highlight=True
)

layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  sdf_05, # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[255*정규화유동인구,150*정규화유동인구, 255*정규화유동인구, 1000*정규화유동인구]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer1.cellSizePixels = 10 # screen 사이즈 조정, default 100

center = [128.48966249, 38.26924467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=8)

r = pdk.Deck(layers=[layer2,layer1], initial_view_state=view_state)
r.to_html('./상권_유동인구layered.html')
# -

# ---------------------------------
# ## 3. 관광정보 분석
#
# **목적: 격자 별 상권 현황을 확인**
#
# **분석 데이터 종류**
# - df_base : '01.소상공인시장진흥공단_상가(상권)정보_강원_202106.csv'
#
# **분석 설명**
# - 빨간색일 수록 관광지가 많고, 노란색일 수록 적으며, 색이 칠해지지 않은 곳은 관광지가 없다.
# - 관광지 데이터는 각 점포의 위치를 나타내므로 (geometry: Point type) 이를 100X100 grid로 나눈 국가지점번호 기준으로 분류하였다.
# - 인구현황(df_08)에서 부여된 고유한 grid id를 국가지점번호 index를 통해 mapping작업하였다.
# - 따라서 관광지 현황을 타 지표와 동일한 100X100 point로 설명 할 수 있는 결과를 도출하였다. 
#
#

# +
# 데이터 불러오기 및 index열 삭제
sdf_01 = pd.read_csv('./input/01.속초-고성_관광명소데이터(시군구_행정동_100미터격자코드포함).csv', encoding = 'cp949')
sdf_01.drop(['Unnamed: 0'], axis = 1, inplace = True)

# 컬럼명 통일
sdf_01.rename(columns={'L2 중분류':'상권업종대분류명'}, inplace=True)
sdf_01.rename(columns={'분류명':'상권업종중분류명'}, inplace=True)
sdf_01['상권업종소분류명']=sdf_01['상권업종중분류명']
sdf_01.rename(columns={'마스터 POI ID':'상가업소번호'}, inplace=True)
sdf_01.rename(columns={'법정읍면동명칭':'법정동명'}, inplace=True)
sdf_01.rename(columns={'X좌표 경도':'경도'}, inplace=True)
sdf_01.rename(columns={'Y좌표 위도':'위도'}, inplace=True)
sdf_01.rename(columns={'시군구명칭':'시군구명'}, inplace=True)
sdf_01.rename(columns={'GRID 격자 코드':'gid'}, inplace=True)

# 컬럼 순서 및 불필요 데이터 삭제
sdf_01 = sdf_01[['상가업소번호', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명', '시군구명','법정동명', '경도', '위도', 'gid']]

# 상건업종중분류 재분류
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('지역호수/저수지','자연관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('대형호수/저수지','자연관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('폭포/계곡','자연관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('휴양림/수목원','휴양관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('유명사찰','역사관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('지역사찰','역사관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('서원/향교/서당','역사관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('토속/특산물/기념품매장','지역축제')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('식물원','일반관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('일반관광지','기타관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('테마공원/대형놀이공원','기타관광지')
sdf_01['상권업종중분류명'] = sdf_01['상권업종중분류명'].str.replace('일반유원지/일반놀이공원','일반관광지')

# 관광지 파일에 gid 추가
# sdf_01['gid'] = sdf_01['상가업소번호'].apply(lambda x: sdf_02.loc[x,'gid'])

# 관광지 현황이 가장 높은 위치(격자) top 10
sdf_01.groupby(['gid','법정동명'])['상가업소번호'].count().sort_values(ascending=False).head(10)
# -

# 관광지 현황이 가장 높은 읍면동 top 10
sdf_01.groupby(['법정동명'])['상가업소번호'].count().sort_values(ascending=False).head(5)

# +
layer = pdk.Layer(
    'ScreenGridLayer', # 대용량 데이터의 경우 'GPUGridLayer' , screentype 'ScreenGridLayer'
    sdf_01,
    get_position='[경도, 위도]',
    pickable=True,
    auto_highlight=True
)
layer.cellSizePixels = 8 # screen 사이즈 조정, default 100

center = [128.48966249, 38.33094467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=9)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.to_html('./관광지현황.html')

# +
# 유동인구 데이터 불러오기
sdf04 = pd.read_csv('./input/04.속초-고성_년도별읍면동_유동인구.csv')

# 주소 unit 별 분할하여 새로운 컬럼 생성
add = sdf04['주소'].str.split(" ", expand=True)
add.columns = ['시도','시군구','읍면동']
sdf04 = pd.concat([sdf04,add], axis=1)

#2015~2018 유동 인구 평균값 구하기 / 컬럼명 통일
sdf04['2015'] = sdf04['2015'].str.replace(',','')
sdf04['2016'] = sdf04['2016'].str.replace(',','')
sdf04['2017'] = sdf04['2017'].str.replace(',','')
sdf04['2018'] = sdf04['2018'].str.replace(',','')
sdf04 = sdf04.astype({'2015':'float','2016':'float','2017':'float','2018':'float'})

sdf04['mean']= sdf04.mean(axis=1,numeric_only = True)
sdf04.rename(columns={'읍면동':'EMD_KOR_NM'}, inplace=True)
sdf04.rename(columns={'시군구':'SIG_KOR_NM'}, inplace=True)
sdf04['SGGEMD'] = sdf04['SIG_KOR_NM']+' '+sdf04['EMD_KOR_NM']

# 읍면동 공간 데이터 불러오기
sdf05 = gpd.read_file('./input/05.법정경계(읍면동).geojson')

# merge 위한 전처리
sdf05['SIG_KOR_NM_x'] = sdf05.loc[sdf05['지역']=='속초','지역'] + '시'
sdf05.fillna('고성군', inplace=True)
sdf05['SGGEMD'] = sdf05['SIG_KOR_NM_x']+' '+sdf05['EMD_NM']

# geojson 파일과 유동인구 파일 merge
sdf05 = pd.merge(sdf05, sdf04, how='outer', on='SGGEMD')
sdf05 = sdf05[['SIG_KOR_NM','EMD_KOR_NM','geometry', 'mean']]
sdf05

# 유동인구 수 정규화
sdf05['정규화유동인구'] = sdf05['mean'] / sdf05['mean'].max()
sdf05.dropna(axis=0, how='any')
sdf05.drop_duplicates(inplace=True)
# -

sdf05

# +
#Pydeck 사용을 위한 함수 정의
import pydeck as pdk
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


# +
#polygon과 multipolygon 혼재 데이터 coordinates화

#polygon과 multipolygon 혼재 데이터 coordinates화

def multipolygon_to_coordinates(series_row): 
    lon, lat = series_row[0].exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]
def polygon_to_coordinates(series_row):
    lon, lat = series_row.exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]
mp_idx = []
p_idx = []
# for i in range(len(floating_geo)):
#     if sdf05['geometry'][i].geom_type == 'MultiPolygon':
#         mp_idx.append(i)
#     if sdf05['geometry'][i].geom_type == 'Polygon':
#         p_idx.append(i)
sdf05['coordinates'] = 0
# for idx1 in p_idx:
#     print(idx1)
#     sdf05['coordinates'].iloc[idx1] = polygon_to_coordinates(sdf05['geometry'][idx1])
# for idx2 in mp_idx:
#         sdf05['coordinates'].iloc[idx2] = multipolygon_to_coordinates(sdf05['geometry'][idx2])
sdf05['coordinates'] = sdf05['geometry'].apply(multipolygon_to_coordinates)

# +
# geometry를 coordinate 형태로 적용
# pydeck 을 위한 coordinate type

# 100X100 grid에서 central point 찾기
sdf05_list1 = []
sdf05_list2 = []
for i in sdf05['geometry']:
    cent = [[i[0].centroid.coords[0][0],i[0].centroid.coords[0][1]]]
    sdf05_list1.append(cent)
    sdf05_list2.append(Point(cent[0]))
sdf05['coord_cent'] = 0
sdf05['geo_cent'] = 0
sdf05['coord_cent']= pd.DataFrame(sdf05_list1) # pydeck을 위한 coordinate type
sdf05['geo_cent'] = sdf05_list2 # geopandas를 위한 geometry type

# 유동인구 현황이 가장 높은 위치
sdf05.iloc[sdf05["mean"].sort_values(ascending=False).index].reindex().head()

# +
layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  sdf05, # 시각화에 쓰일 데이터프레임 
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[255*정규화유동인구,150*정규화유동인구, 255*정규화유동인구, 1000*정규화유동인구]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.48966249, 38.26924467] # 지도 뷰 중심 좌표
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=9
) 

# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state) 

r.to_html('./유동인구현황.html')
# -







# ---------------------------------
# ## 4. 교통량 분석
#
#
# **목적: 주거시간과 업무시간의 교통량 분석**
#
# **분석 데이터 종류**
# - df_10: 10.강원도속초시,고성군_상세도로망.geojson
# - df_11: 11.평일_일별_시간대별_추정교통량(대체데이터).csv"
#
# **분석 방법**
# - 전기자동차는 승용차만을 고려한다고 가정
# - 교통량: 11시, 14시의 승용차 교통량
#
#
# **분석그래프 설명**
# - 붉은 색일수록 혼잡강도가 높은 것이다.
# - 선이 굵을 수록 혼잡강도가 높은 것이며, 얇을 수록 낮은 것이다

df_10 = gpd.read_file('10.강원도속초시,고성군_상세도로망.json')
df_11 = pd.read_csv('11.평일_일별_시간대별__추정교통량.csv')

df = df_10
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()

# +
# 여행객들이 11시에 관광지로 움직일 것으로 가정
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



center = [128.5918, 38.27701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 
r.to_html('교통량_11.html')

# -

# ### 분석 결과
#
# - 출근시간을 피한 오전 11시에 교통량이 많은 곳은 관광객이 많이 가는 곳으로 간주하였다.

# +
# 여행객들이 14시에 관광지로 움직일 것으로 가정
df_11_time14=df_11[df_11['시간적범위']==14]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time14[df_11_time14['link_id'].apply(str).str.contains(i)]['승용차'])])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time14=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time14.iloc[df_10_11_time14["교통량"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_11_time14, 
                  get_path='coordinate', 
                  get_width='교통량/2', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.27701] # 센터 [128.5918, 38.27701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state
            ) 
r.to_html('교통량_14.html')

# ### 분석 결과
# - 출퇴근 시간을 피한 오후 14시에 교통량이 많은 곳은 관광객이 많이 가는 지역일 것으로 기대
# - 급속 충전소는 관광지역 설치가 효과적이라 판단
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
# **분석 방법**
# - 혼잡빈도강도합: 양방향의 혼잡빈도강도 합
# - 정규화도로폭: width변수를 이용하여 정규화
#
#
# **분석그래프 설명**
# - 도로폭이 넓을 수록 노란색이며 좁을 수록 붉은색이다.
# - 선이 굵을 수록 혼잡빈도강도가 높은 것이며, 얇을 수록 낮은 것이다

df_10= gpd.read_file("10.강원도속초시,고성군_상세도로망.json")
df_12= pd.read_csv("12.평일_혼잡빈도강도_강원도 속초시, 고성군.csv")
df_13= pd.read_csv("13.평일_혼잡시간강도_강원도 속초시, 고성군.csv")

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
                  get_width='혼잡빈도강도합', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.27701] # 센터 [128.5918, 38.27701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
#             mapbox_key = 'sk.eyJ1IjoieW9vYnl1bmdjaHVsIiwiYSI6ImNrd245YnMwZzFiMnEycHBkc2gzbzkzd3AifQ.sc9Gmo56AsAHzJ2B3wCkXg') 
r.to_html('혼잡빈도강도.html')

# +
# 혼합시간강도 양방향 총 합
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_13[df_13['link_id'].apply(str).str.contains(i)].혼잡시간강도)])
    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "혼잡시간강도합"]
df_10_13=pd.merge(df, df_10_,on = 'link_id' )
# 혼잡시간강도 합이 가장 높은 도로
df_10_13.iloc[df_10_13["혼잡시간강도합"].sort_values(ascending=False).index].reindex().head()
# -

layer = pdk.Layer( 'PathLayer', 
                  df_10_13, 
                  get_path='coordinate', 
                  get_width='혼잡시간강도합', 
                  get_color='[255, 255 * 정규화도로폭, 120]', 
                  pickable=True, auto_highlight=True 
                 ) 
center = [128.5918, 38.27701] # 센터 [128.5918, 38.27701]
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
# **목적: 급속 충전소의 경우 사유지는 제외 해야 하므로 설치 가능 장소 필터링 필요**
#
# **분석 데이터 종류**
# - df_14: 14.소유지정보.geojson
#
# **분석 방법**
# - 급속충전소 설치 가능 장소 필터링
# - 소유구분 코드: 국유지, 시/군 선택
# - 지목코드: 임야, 염전, 도로, 철도 용지, 제방, 하천, 유지 제외
# - 설치가능장소에 포함되는 기준 point 추출
#
# **분석 설명**
# - 사유지를 포함한 임야, 염전, 도로, 철도 용지, 제방, 하천과 같이 설치가 부적절 한 곳을 필터링 한 multipolygone을 시각화하였다.
# - 앞서 도출한 인구현황 100X100 Point 데이터셋에서 설치가능한 장소에 해당하는 point를 추출하였다.
#

# +
df_14= gpd.read_file("14.소유지정보.geojson") # geojson -> json
# 급속 충전소 설치 가능 장소
df_14_=df_14[df_14['소유구분코드'].isin(['02','04'])] #소유구분코드: 국유지, 시/군
df_14_possible=df_14[df_14['소유구분코드'].isin(['02','04']) 
      & (df_14['지목코드'].isin(['05','07','14','15','16','17',
                             '18','19','20','27' ])==False)] # 임야, 염전, 도로, 철도 용지, 제방, 하천 제외 

# geometry to coordinates
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
                  get_fill_color='[255, 0, 127]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

# Set the viewport location 
center = [128.5918, 38.27701] # 센터 [128.5918, 38.27701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 


# Render 
r = pdk.Deck(layers=[layer], initial_view_state=view_state,
            ) 

    
r.to_html('충전소설치가능위치.html')
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
# 격자별 인구 현황
df_08= gpd.read_file("08.속초-고성_격자별인구현황.json")

# val 열 na 제거
df_08['val'] = df_08['val'].fillna(0)

# geometry를 coordinate 형태로 적용
df_08['coordinates'] = df_08['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type

# 100X100 grid에서 central point 찾기
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
df_08

# +
# %%time
# 최종 분석 데이터 정제하기

# 개발 가능 한 grid point 찾기
shapely.speedups.enable()
df_result = df_08[['gid','grid_id','val','geometry','coordinates','coord_cent','geo_cent']]
df_result['val'] = df_result['val'].fillna(0)
point_cent= gpd.GeoDataFrame(df_result[['grid_id','geo_cent']],geometry = 'geo_cent')

#굉장히 오래걸림
# within_points=point_cent.buffer(0.00000001).within(geo_possible.loc[0,'geometry'])
# pd.DataFrame(within_points).to_csv("within_points.csv", index = False)

within_points=pd.read_csv("within_points.csv")
df_result['개발가능'] = 0
df_result['개발가능'][within_points['0']==True] = 1
df_result[df_result['개발가능']==1]



## 71574개 중 9042개가 개발 가능한 구역
# -

# ### 분석 결과
# - 기본 상식선에서 개발 가능한 토지를 구분하였으나 토지관련 전문가를 통해 더 자세하게 토지를 추가/제거 한다면 더 현실적인 결과가 나올 것이다.
#
# -------------------------------------

# ## 100X100 Point에 관광지 데이터 부여
# - 관광지 geometry

df_21 = gpd.read_file('21.관광지_gemetry.GeoJSON')
df_22 = gpd.read_file('22.상권_gemetry.GeoJSON')

df_21

# +
# %%time
# grid 마다 정규화_관광지_score_stay 부여
df_21_grid = []
df_superset = df_21

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_21_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_21_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_21_grid)):
    id_idx = df_21_grid[i][0]
    grids = df_21_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_21_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_21_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_관광지_score_stay'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_관광지_score_stay']
except:
    pass

grid_21=pd.DataFrame(grid_21_list)
grid_21.columns = ["grid_id","정규화_관광지_score_stay"]

df_result = pd.merge(df_result, grid_21, on = 'grid_id')
df_result



# +
# %%time
# grid 마다 정규화_관광지_네비검색량 부여
df_21_grid = []
df_superset = df_21

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_21_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_21_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_21_grid)):
    id_idx = df_21_grid[i][0]
    grids = df_21_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_21_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_21_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_관광지_네비검색량'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_관광지_네비검색량']
except:
    pass

grid_21=pd.DataFrame(grid_21_list)
grid_21.columns = ["grid_id","정규화_관광지_네비검색량"]

df_result = pd.merge(df_result, grid_21, on = 'grid_id')
df_result


# +
# %%time
# grid 마다 정규화_관광지_방문객수 부여
df_21_grid = []
df_superset = df_21

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_21_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_21_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_21_grid)):
    id_idx = df_21_grid[i][0]
    grids = df_21_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_21_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_21_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_관광지_방문객수'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_관광지_방문객수']
except:
    pass

grid_21=pd.DataFrame(grid_21_list)
grid_21.columns = ["grid_id","정규화_관광지_방문객수"]

df_result = pd.merge(df_result, grid_21, on = 'grid_id')
df_result


# +
# %%time
# grid 마다 관광지_동_유동인구 부여
df_21_grid = []
df_superset = df_21

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_21_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_21_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_21_grid)):
    id_idx = df_21_grid[i][0]
    grids = df_21_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_21_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_21_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['관광지_동_유동인구'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['관광지_동_유동인구']
except:
    pass

grid_21=pd.DataFrame(grid_21_list)
grid_21.columns = ["grid_id","관광지_동_유동인구"]

df_result = pd.merge(df_result, grid_21, on = 'grid_id')
df_result

# -

# ## 100X100 Point에 상권 데이터 부여
# - 상권 geometry

# +
# %%time
# grid 마다 정규화_상권_score_search_point 부여
df_22_grid = []
df_superset = df_22

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_22_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_22_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_22_grid)):
    id_idx = df_22_grid[i][0]
    grids = df_22_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_22_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_22_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_상권_score_search_point'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_상권_score_search_point']
except:
    pass

grid_22=pd.DataFrame(grid_22_list)
grid_22.columns = ["grid_id","정규화_상권_score_search_point"]

df_result = pd.merge(df_result, grid_22, on = 'grid_id')
df_result


# +
# %%time
# grid 마다 정규화_상권_score_view_count 부여
df_22_grid = []
df_superset = df_22

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_22_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_22_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_22_grid)):
    id_idx = df_22_grid[i][0]
    grids = df_22_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_22_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_22_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_상권_score_view_count'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_상권_score_view_count']
except:
    pass

grid_22=pd.DataFrame(grid_22_list)
grid_22.columns = ["grid_id","정규화_상권_score_view_count"]

df_result = pd.merge(df_result, grid_22, on = 'grid_id')
df_result


# +
# %%time
# grid 마다 정규화_상권_score_review_count 부여
df_22_grid = []
df_superset = df_22

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_22_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_22_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_22_grid)):
    id_idx = df_22_grid[i][0]
    grids = df_22_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_22_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_22_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_상권_score_review_count'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_상권_score_review_count']
except:
    pass

grid_22=pd.DataFrame(grid_22_list)
grid_22.columns = ["grid_id","정규화_상권_score_review_count"]

df_result = pd.merge(df_result, grid_22, on = 'grid_id')
df_result

# -

df_22

# +
# %%time
# grid 마다 정규화_상권_score_stay 부여
df_22_grid = []
df_superset = df_22

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_22_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_22_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_22_grid)):
    id_idx = df_22_grid[i][0]
    grids = df_22_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_22_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_22_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['정규화_상권_score_stay'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['정규화_상권_score_stay']
except:
    pass

grid_22=pd.DataFrame(grid_22_list)
grid_22.columns = ["grid_id","정규화_상권_score_stay"]

df_result = pd.merge(df_result, grid_22, on = 'grid_id')
df_result


# +
# %%time
# grid 마다 정규화_상권_score_stay 부여
df_22_grid = []
df_superset = df_22

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_22_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_22_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_22_grid)):
    id_idx = df_22_grid[i][0]
    grids = df_22_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_22_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_22_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['상권_동_유동인구'])])
    except:
        pass

#혼잡시간강도합 관련 정보
try:
    del df_result['상권_동_유동인구']
except:
    pass

grid_22=pd.DataFrame(grid_22_list)
grid_22.columns = ["grid_id","상권_동_유동인구"]

df_result = pd.merge(df_result, grid_22, on = 'grid_id')
df_result

# -

# ## 100X100 Point에 교통량, 혼잡빈도강도, 혼잡시간강도,   관련 요소 부여
#
# **목적: grid 마다 교통량 관련 요소 부여**
#
# **분석 데이터 종류**
# - df_11: 11.평일_일별_시간대별__추정교통량.csv
# - df_12: 12.평일_혼잡빈도강도_강원도 속초시, 고성군.csv
# - df_13: 13.평일_혼잡시간강도_강원도 속초시, 고성군.csv
#
# **분석 설명**
# - 각 100X100 Point 마다 11시 교통량, 14시 교통량, 혼잡빈도강도합, 혼잡시간강도합을 부여
# - 각 요소바다 부여하는데 시간이 다소 소요됨 (약 10분)

# +
# %%time
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


## 교통량_11 2798개 중 2381개가 point와 관련되어있는 grid 개수이고
## 임의 gird 중 71581개 중 교통량_11 5754개가 개발가능여부 부여

# +
# %%time
# grid 마다 14시 교통량 부여
df_10_11_time14_grid = []
df_superset = df_10_11_time14[df_10_11_time14['교통량']>0]

#시간이 오래 걸립니다.
for i in tqdm(range(len(df_superset))): #len(df_superset)
    try:
        grid_ids = point_cent[point_cent.within(df_superset.loc[i,'geometry'].buffer(0.001))]['grid_id']
        if len(grid_ids) != 0:
            df_10_11_time14_grid.append([i,str(tuple(grid_ids))])
    except :
        pass

print('Point와 관련된 grid 개수: ',len(df_10_11_time14_grid))

df_superset['grid_ids'] = 0
for i in range(len(df_10_11_time14_grid)):
    id_idx = df_10_11_time14_grid[i][0]
    grids = df_10_11_time14_grid[i][1]
    df_superset['grid_ids'][id_idx] = grids


#시간이 오래 걸립니다.
grid_혼잡빈도_list = []
for i in tqdm(df_result['grid_id']):
    try:
        grid_혼잡빈도_list.append([i, sum(df_superset[df_superset['grid_ids'].str.contains(i)==True]['교통량'])])
    except:
        pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_14"]
grid_혼잡빈도[grid_혼잡빈도['교통량_14']>0]


#17시 승용차 혼잡 빈도 관련 정보
try:
    del df_result['교통량_14']
except:
    pass

grid_혼잡빈도=pd.DataFrame(grid_혼잡빈도_list)
grid_혼잡빈도.columns = ["grid_id","교통량_14"]
#grid_혼잡빈도[grid_혼잡빈도['승용차_혼잡빈도강도합']>0]
df_result = pd.merge(df_result, grid_혼잡빈도, on = 'grid_id')
df_result[df_result['교통량_14']>0]




## 교통량_17 2559개 중 2079개가 point와 관련되어있는 grid 개수이고
## 임의 gird 중 71581개 중 교통량_17 4743개가 개발가능여부 부여

# +
# %%time
# grid 마다 혼잡빈도강도 부여
df_10_grid = []
df_superset = df_10_12[df_10_12['혼잡빈도강도합']>0]

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



## 혼잡빈도강도합 2878개 중 2507개가 point와 관련되어있는 grid 개수이고
## 임의 gird 중 71581개 중 혼잡빈도강도합 6624개가 개발가능여부 부여

# +
# %%time
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

## 혼잡시간강도합 2904개 중 2555개가 point와 관련되어있는 grid 개수이고
## 임의 gird 중 71581개 중 혼잡시간강도 6861개가 개발가능여부
# -

# ---------------------------------
# ## 기존 충전소 위치 분석
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

df_01 = pd.read_csv('01.고성군_속초시_충전기설치현황.csv')

# +
# %%time
# 기존 완속/ 급속 충전소가 커버하는 위치 제거
df_01_geo = []
for i in range(len(df_01)):
    df_01_geo.append([df_01.loc[i,'충전소명'],Point(df_01.loc[i,'lon'],df_01.loc[i,'lat']).buffer(0.003)])
# df_01[df_01['급속/완속']=='완속']
df_01_geo = pd.DataFrame(df_01_geo)
df_01_geo.columns = ["충전소명", "geometry"]
df_01_geo = pd.merge(df_01, df_01_geo, on = '충전소명')
df_01_geo['coordinates'] = df_01_geo['geometry'].apply(polygon_to_coordinates) 
df_01_geo = pd.DataFrame(df_01_geo)





center = [128.5918, 38.27701] # 센터 [128.5918, 38.27701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 
layer1 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='급속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[50, 50, 200]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 

layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_01_geo[df_01_geo['급속/완속']=='완속'][['coordinates']], # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[100, 200, 100]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
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

df_result.columns

# ### 분석 결과
# - 전기자동차 충전소가 cover가능한 거리는 임의로 정한 것으로 바뀔 필요가 있다.
# - 현재는 전기자동차가 많이 등록되어 있지 않아 거리로 임의로 정하였지만 대수가 많아 진다면 전기자동차 등록 대수에 따라 cover 가능한 거리가 바뀌어야 할 것이다.
# - 즉, 전기자동차 등록이 많은 곳은 cover 가능한 거리를 줄여 더 많은 곳에 충전소를 설치해야 한다.
# -------------------------------------
#
#

# ---------------------------------
# ## 전기자동차 충전소 위치선정에 대한 영향 요소 분석 및 상관관계 분석
#
# 본 지원자는 전기자동차 충전소 위치 선정을 최적화 문제로 풀 것이다. 이를 위해 목적함수가 필요하며 다음과 같은 기준으로 식을 세웠다. 
#
# **가정**
# 1. 전기자동차 충전소 위치는 인구현황, 교통량, 혼잡빈도강도, 혼잡시간강도 만 고려하여 위치를 선정한다.
# 2. 기존 설치된 전기자동차 충전소는 위 고려사항을 충분히 고려하여 만들어진 곳이다.
# 3. 전기자동차 충전소는 전방 약 500m를 커버할 수 있다. 
#
#
# **분석 방법**
#
# - 고려되는 모든 변수들은 정규화 하였다.
# - 선형회귀분석을 이용해 현재 제공받은 데이터로부터 전기자동차 충전소 위치에 영향을 주는 요소의 관계를 분석하였다.(Linear Regrssion)
#
# - 이때 급속 충전소와 완속 충전소 각각을 따로 분석하였다.  
#
# - **분석 Input**: (정규화된) 인구현황, 11시 교통량, 14시 교통량, 혼잡빈도강도, 혼잡시간강도
#
# - **분석 Output**: 고려되는 요소들과 각 충전소 사이의 상관계수 
#
#

df_21.columns

df_22.columns

# +
## 상권, 교통량 혼잡강도 Linear Regresiion 비교 설정

df_result['정규화_교통량_11'] = df_result['교통량_11'] / df_result['교통량_11'].max()
df_result['정규화_교통량_14'] = df_result['교통량_14'] / df_result['교통량_14'].max()
df_result['정규화_혼잡빈도강도합'] = df_result['혼잡빈도강도합'] / df_result['혼잡빈도강도합'].max()
df_result['정규화_혼잡시간강도합'] = df_result['혼잡시간강도합'] / df_result['혼잡시간강도합'].max()
df_result['정규화_관광지_동_유동인구'] = df_result['관광지_동_유동인구'] / df_result['관광지_동_유동인구'].max()
df_result['정규화_상권_동_유동인구'] = df_result['상권_동_유동인구'] / df_result['상권_동_유동인구'].max()



# 급속/ 완속 관련 objective function 만들기
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

df_LR = df_result
X = df_LR[["정규화_교통량_11","정규화_교통량_14","정규화_혼잡빈도강도합","정규화_혼잡시간강도합",
           '정규화_관광지_score_stay', '정규화_관광지_네비검색량', '정규화_관광지_방문객수', '정규화_관광지_동_유동인구',
           '정규화_상권_score_search_point', '정규화_상권_score_view_count',
           '정규화_상권_score_review_count', '정규화_상권_score_stay', '정규화_상권_동_유동인구']]

y = df_LR["FS_station"]
regr = linear_model.LinearRegression()
regr.fit(X, y)
FS_coeff = regr.coef_
print('급속충전소 Intercept: ', regr.intercept_)
print('급속충전소 Coefficients: \n', FS_coeff)

# df_LR = df_result
# X = df_LR[["정규화_교통량_11","정규화_교통량_14","정규화_혼잡빈도강도합","정규화_혼잡시간강도합",
#            '정규화_관광지_score_stay', '정규화_관광지_네비검색량', '정규화_관광지_방문객수', '정규화_관광지_동_유동인구',
#            '정규화_상권_score_search_point', '정규화_상권_score_view_count',
#            '정규화_상권_score_review_count', '정규화_상권_score_stay', '정규화_상권_동_유동인구']]
# y = df_LR["SS_station"]
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# SS_coeff = regr.coef_
# print('완속충전소 Intercept: ', regr.intercept_)
# print('완속충전소 Coefficients: \n', SS_coeff)
# -


df_result

df_result.columns

# OLS Regression Results 
import statsmodels.api as sm
X = sm.add_constant(X, has_constant='add')
modeling = sm.OLS(y, X)
modeling.fit().summary()

import seaborn as sb
import warnings
import platform
from matplotlib import font_manager, rc
## 운영체제별 글꼴 세팅
path = "c:/Windows/Fonts/malgun.ttf"
if platform.system() == 'Darwin':
    font_name = 'AppleGothic'
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    font_name = font_manager.FontProperties(fname="/usr/share/fonts/nanumfont/NanumGothic.ttf")
    rc('font', family="NanumGothic")
warnings.simplefilter(action='ignore')

mask = np.zeros_like(df_result.corr())
mask[np.triu_indices_from(mask)] = True
plt.rcParams["figure.figsize"] = (20,10) # 그림 크기 조정
sb.heatmap(data = df_result.corr(), mask=mask, annot=True, fmt = '.3f', linewidths=0, cmap='Greens')

df_result['w_FS'] = 0 
df_result['w_FS'] = (FS_coeff[0]*df_result['정규화_교통량_11']+
                     FS_coeff[1]*df_result['정규화_교통량_14']+
                     FS_coeff[2]*df_result['정규화_혼잡빈도강도합']+
                     FS_coeff[2]*df_result['정규화_혼잡시간강도합']+
                     FS_coeff[3]*df_result['정규화_관광지_score_stay']+
                     FS_coeff[3]*df_result['정규화_관광지_네비검색량']+
                     FS_coeff[3]*df_result['정규화_관광지_방문객수']+
                     FS_coeff[3]*df_result['정규화_관광지_동_유동인구']+
                     FS_coeff[4]*df_result['정규화_상권_score_search_point']+                     
                     FS_coeff[4]*df_result['정규화_상권_score_view_count']+
                     FS_coeff[4]*df_result['정규화_상권_score_review_count']+
                     FS_coeff[4]*df_result['정규화_상권_score_stay']+
                     FS_coeff[4]*df_result['정규화_상권_동_유동인구']
                    )
df_result['w_SS'] = 0 
df_result['w_SS'] = (SS_coeff[0]*df_result['정규화_교통량_11']+
                     SS_coeff[1]*df_result['정규화_교통량_14']+
                     SS_coeff[2]*df_result['정규화_혼잡빈도강도합']+
                     SS_coeff[2]*df_result['정규화_혼잡시간강도합']+
                     SS_coeff[3]*df_result['정규화_관광지_score_stay']+
                     SS_coeff[3]*df_result['정규화_관광지_네비검색량']+
                     SS_coeff[3]*df_result['정규화_관광지_방문객수']+
                     SS_coeff[3]*df_result['정규화_관광지_동_유동인구']+
                     SS_coeff[4]*df_result['정규화_상권_score_search_point']+                     
                     SS_coeff[4]*df_result['정규화_상권_score_view_count']+
                     SS_coeff[4]*df_result['정규화_상권_score_review_count']+
                     SS_coeff[4]*df_result['정규화_상권_score_stay']+
                     SS_coeff[4]*df_result['정규화_상권_동_유동인구']
                    )


try:    
    df_result[['grid_id','geometry',
               '정규화_교통량_11','정규화_교통량_14',
               '정규화_혼잡빈도강도합', '정규화_혼잡시간강도합','정규화_관광지_score_stay','정규화_관광지_네비검색량',
               '정규화_관광지_방문객수','정규화_관광지_동_유동인구','정규화_상권_score_search_point',
               '정규화_상권_score_view_count','정규화_상권_score_review_count','정규화_상권_score_stay',
               '정규화_상권_동_유동인구',
               'w_FS','w_SS','개발가능','FS_station','SS_station']].to_file("df_result.geojson", driver="GeoJSON")
except:
    pass

df_result


# ### 분석 결과
# - 금속충전소와 완속충전소는 고려되는 요소들의 영향이 차이가 있다.
# - 급속충전소 경우, 정규화_혼잡빈도강도합에 가장 많은 영향을 받았고, 완속충전소 경우, 정규화_인구에 가장 많은 영향을 받았다. 
# - 이 결과는 차량이 많이 다는 곳에 급속 충전소를 설치하고, 주거 공간이 많이 있는 곳에 완속 충전소를 설치했다고 해석할 수 있다.
# - 완속 충전소의 경우 급속충전소가 가장 크게 영향을 받은 정규화_혼잡빈도강도합의 변수값이 음수로 분석되었으며, 이는 혼잡한 곳에는 오히려 완속충전소가 설치 되지 않는 것을 의미한다. 
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

df_test=gpd.read_file('df_result.geojson')
df_test
# 100X100 grid에서 central point 찾기
df_list = []
df_list2 = []
for i in df_test['geometry']:
    cent = [[i.centroid.coords[0][0],i.centroid.coords[0][1]]]
    df_list.append(cent)
    df_list2.append(Point(cent[0]))
df_test['coord_cent'] = 0
df_test['geo_cent'] = 0
df_test['coord_cent']= pd.DataFrame(df_list) # pydeck을 위한 coordinate type
df_test['geo_cent'] = df_list2 # geopandas를 위한 geometry type
df_test

# +
# %%time
## 급속충전소 500m반경
df_result_fin = df_test[(df_test['개발가능']==1)
                          &(df_test['FS_station']!=1)]
df_result_fin

points = []
for i in df_result_fin['coord_cent'] :
    points.append(i)

w= []
for i in df_result_fin['w_FS'] :
    w.append(i)

radius = radius = (1/88.74/1000)*200     ## 500m 반경을 표현함
K = 50  ## 총 설치해야하는 설비 개수
M = 5000  ## 급속 충전소 입지선정지수가 가장 높은 5000개 point

opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_FS')


df_opt_FS= pd.DataFrame(opt_sites_org)
df_opt_FS.columns = ['lon', 'lat']
df_opt_FS

# +
layer0 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
                  df_14_possible, # 시각화에 쓰일 데이터프레임
                  #df_result_fin[df_result_fin['val']!=0],
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
                  get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
                  pickable=True, # 지도와 interactive 한 동작 on 
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
                 ) 


# layer1 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_14_possible, # 시각화에 쓰일 데이터프레임
#                   #df_result_fin[df_result_fin['val']!=0],
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  ) 

# layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[900, 255*정규화인구, 0, 정규화인구*10000 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  )  

# layer3 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_14_possible, # 시각화에 쓰일 데이터프레임
#                   #df_result_fin[df_result_fin['val']!=0],
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  ) 

# layer4 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[900, 255*정규화인구, 0, 정규화인구*10000 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  )  

# layer5 = pdk.Layer( 'PathLayer', 
#                   df_10_11_time14, 
#                   get_path='coordinate', 
#                   get_width='교통량/2', 
#                   get_color='[255, 255 * 정규화도로폭, 120]', 
#                   pickable=True, auto_highlight=True 
#                  ) 


# layer6 = pdk.Layer( 'PathLayer', 
#                   df_10_12, 
#                   get_path='coordinate', 
#                   get_width='혼잡빈도강도합/2', 
#                   get_color='[255, 255 * 정규화도로폭, 120,140]', 
#                   pickable=True, auto_highlight=True 
#                  ) 


# layer7 = pdk.Layer( 'PathLayer', 
#                   df_10_13, 
#                   get_path='coordinate', 
#                   get_width='혼잡시간강도합/10', 
#                   get_color='[255, 255 * 정규화도로폭, 120]', 
#                   pickable=True, auto_highlight=True 
#                  ) 

# Set the viewport location 
center = [128.5918, 38.27701] # 속초 센터 [128.5918, 38.20701]
view_state = pdk.ViewState( 
    longitude=center[0], 
    latitude=center[1], 
    zoom=10
) 

## 기존 충전소 위치
scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[50, 50, 200]',
    pickable=True)

## 최적화 충전소 위치
opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[0, 255, 255]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)



# Render 
r = pdk.Deck(layers=[layer0,scatt,opt], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g"

    
r.to_html('급속_반경200m.html')


## 노란색 : 제안된 최적화 지역
## 파란색 : 기존 급속 충전소

# +
#GridLayer 시각화

layer2 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입
                  sdf05, # 시각화에 쓰일 데이터프레임
                  get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름
                  get_fill_color='[255*정규화유동인구,150*정규화유동인구, 255*정규화유동인구, 1000*정규화유동인구]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
                  pickable=True, # 지도와 interactive 한 동작 on
                  auto_highlight=True # 마우스 오버(hover) 시 박스 출력
                 )

layer1 = pdk.Layer(
    'ScreenGridLayer', # 대용량 데이터의 경우 'GPUGridLayer' , screentype 'ScreenGridLayer'
    sdf_00,
    get_position='[경도, 위도]',
    pickable=True,
    auto_highlight=True
)
layer1.cellSizePixels = 10 # screen 사이즈 조정, default 100

## 기존 충전소 위치
scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[50, 50, 200]',
    pickable=True)

## 최적화 충전소 위치
opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=300,
    get_fill_color='[0, 255, 255]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)



# Render 
r = pdk.Deck(layers=[layer2, layer1, scatt, opt], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g"

    
r.to_html('급속_반경200m.html')


## 노란색 : 제안된 최적화 지역
## 파란색 : 기존 급속 충전소

# +
layer7 = pdk.Layer( 'PathLayer', 
                   df_10_13, 
                   get_path='coordinate', 
                   get_width='혼잡시간강도합/10', 
                   get_color='[255, 255 * 정규화도로폭, 120]', 
                   pickable=True, auto_highlight=True 
                  ) 

## 기존 충전소 위치
scatt = pdk.Layer(
    'ScatterplotLayer',
    df_01_geo[df_01_geo['급속/완속']=='급속'][['lon','lat']],
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[50, 50, 200]',
    pickable=True)

## 최적화 충전소 위치
opt = pdk.Layer(
    'ScatterplotLayer',
    df_opt_FS,
    get_position = ['lon','lat'],
    auto_highlight=True,
    get_radius=200,
    get_fill_color='[0, 255 , 0]',
    get_line_color = '[0, 0, 0]',
    line_width_min_pixels=5,
    pickable=True)

# Render 
r = pdk.Deck(layers=[layer7,scatt,opt], initial_view_state=view_state)
#             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g"

    
r.to_html('급속_반경150m.html')


## 노란색 : 제안된 최적화 지역
## 파란색 : 기존 급속 충전소

# +
###### 완속 충전소




# df_result_fin = df_test[(df_test['SS_station']!=1)]
# df_result_fin

# points = []
# for i in df_result_fin['coord_cent'] :
#     points.append(i)

# w= []
# for i in df_result_fin['w_SS'] :
#     w.append(i)

# radius = (1/88.74/1000)*50   
# K = 30
# M = 5000

# opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_SS')


# df_opt_SS= pd.DataFrame(opt_sites_org)
# df_opt_SS.columns = ['lon', 'lat']
# df_opt_SS



# # Make layer 

# layer = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_14_possible, # 시각화에 쓰일 데이터프레임
#                   #df_result_fin[df_result_fin['val']!=0],
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[0, 255*1, 0,140]', # 각 데이터 별 rgb 또는 rgba 값 (0~255) 
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  ) 

# layer0 = pdk.Layer( 'PolygonLayer', # 사용할 Layer 타입 
#                   df_08[(df_08['val'].isnull()==False) & df_08['val']!=0], # 시각화에 쓰일 데이터프레임 
#                   get_polygon='coordinates', # geometry 정보를 담고있는 컬럼 이름 
#                   get_fill_color='[900, 255*정규화인구, 0, 정규화인구*10000 ]', # 각 데이터 별 rgb 또는 rgba 값 (0~255)
#                   pickable=True, # 지도와 interactive 한 동작 on 
#                   auto_highlight=True # 마우스 오버(hover) 시 박스 출력 
#                  )  

# # layer1 = pdk.Layer( 'PathLayer', 
# #                   df_10_11_time14, 
# #                   get_path='coordinate', 
# #                   get_width='교통량/2', 
# #                   get_color='[255, 255 * 정규화도로폭, 120]', 
# #                   pickable=True, auto_highlight=True 
# #                  ) 


# # layer2 = pdk.Layer( 'PathLayer', 
# #                   df_10_12, 
# #                   get_path='coordinate', 
# #                   get_width='혼잡빈도강도합/2', 
# #                   get_color='[255, 255 * 정규화도로폭, 120,140]', 
# #                   pickable=True, auto_highlight=True 
# #                  ) 


# layer3 = pdk.Layer( 'PathLayer', 
#                   df_10_13, 
#                   get_path='coordinate', 
#                   get_width='혼잡시간강도합/2', 
#                   get_color='[255, 255 * 정규화도로폭, 120]', 
#                   pickable=True, auto_highlight=True 
#                  ) 


# # Set the viewport location 
# center = [128.5918, 38.27701] # 속초 센터 [128.5918, 38.20701]
# view_state = pdk.ViewState( 
#     longitude=center[0], 
#     latitude=center[1], 
#     zoom=10
# ) 


# scatt = pdk.Layer(
#     'ScatterplotLayer',
#     df_01_geo[df_01_geo['급속/완속']=='완속'][['lon','lat']],
#     get_position = ['lon','lat'],
#     auto_highlight=True,
#     get_radius=200,
#     get_fill_color='[100, 200, 100,140]',
#     pickable=True)

# opt = pdk.Layer(
#     'ScatterplotLayer',
#     df_opt_SS,
#     get_position = ['lon','lat'],
#     auto_highlight=True,
#     get_radius=200,
#     get_fill_color='[255, 255, 0]',
#     get_line_color = '[0, 0, 0]',
#     line_width_min_pixels=5,
#     pickable=True)



# # Render 
# r = pdk.Deck(layers=[layer,layer0,layer3,scatt,opt], initial_view_state=view_state)
# #             mapbox_key = "pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrZzF4bWNhdTBpNnEydG54dGpxNDEwajAifQ.XWxOKQ-2HqFBVBYa-XoS-g") 


    
# r.to_html('녹색_기존완속충전소_노란색_제안된 완속충전소 최적화지역.html')

# ## 노란색 : 제안된 최적화 지역
# ## 녹색 : 기존 완속 충전소
# -

df_opt_FS['충전소구분']='급속'
# df_opt_SS['충전소구분']='완속'
pd.concat([df_opt_FS]).to_csv("충전소결과.csv", index=False)






