# -*- coding: utf-8 -*-
import time
import math
import utm
from scipy.spatial import distance
from math import radians, cos, sin, asin, sqrt
import shapely.affinity
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPoint
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.spatial.distance import cdist
from haversine import Haversine
from geopy.distance import great_circle

# =============================================================================      

#------------------------START OF EXECUTION------------------------
start = time.time()

#------------------------DISTANCES TO CENTROIDS------------------------
def get_Distance(vec_inside_cluster, centroids, v):

      for i in range(len(vec_inside_cluster)):

            distance = 0
            dist_flood = 0
            
            for j in vec_inside_cluster[i]:

                  dist_flood = Haversine([j[1], j[0]], [centroids.loc[i]['longitude'],centroids.loc[i]['latitude']]).meters
                  
                  if dist_flood > distance:
                        distance = dist_flood
            
            limit = v * 100
            
            if distance < limit:
                  distance_lim = limit
            else:
                  distance_lim = distance

            centroids.loc[i, 'dist_flood'] = round(distance,4)
            centroids.loc[i, 'dist_flood_lim'] = round(distance_lim,4)

      return centroids

#------------------------CENTERING------------------------
def get_most_central_point(flood_areas):
    centroid_of_flood_areas = (MultiPoint(flood_areas).centroid.x, MultiPoint(flood_areas).centroid.y)
    most_central_point = min(flood_areas, key=lambda point: great_circle(point, centroid_of_flood_areas).m)
    return tuple(most_central_point)

#------------------------HAVERSINE DISTANCE------------------------
def haversine(coordinates_1, coordinates_2):
    
    latitude_1, longitude_1 = coordinates_1
    latitude_2, longitude_2 = coordinates_2
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    # haversine formula 
    distance_longitude = longitude_2 - longitude_1 # diference between two longitude
    distance_latitude = latitude_2 - latitude_1    # diference between two latitude
    
    aux = sin(distance_latitude/2)**2 + cos(latitude_1) * cos(latitude_2) * sin(distance_longitude/2)**2
    result = 2 * asin(sqrt(aux)) 
    
    radius = 6371 # Radius of earth in kilometers
    
    return result * radius

#------------------------REDUCE CLUSTER------------------------
def reduce_cluster(dataframe, labels_unique, count):

    vec = []
    vec_lat = []
    vec_lon = []
    vec_ok = []
    vec_notok = []

    for i in labels_unique:

        result = dataframe[dataframe['cluster'] == i]
        #print(result)

        sum_count = result['counts'].sum()
        #print(sum_count)

        tamanho = len(result)
        #print(tamanho)

        cond = sum_count/tamanho
        #print(cond)

        #-----remove clusters where there is only 1 point and that occurred only once-----
        if cond > count:

            vec_result = []
            vec_lat_result = []
            vec_lon_result = []

            indexes_result = result.index

            for j in indexes_result:

                vec_result.append([result.loc[j]['latitude'], result.loc[j]['longitude']])

            #----vec result----
            vec_ok.append(result)
            #----vec lat and long----
            vec.append(vec_result)

        else:
            #----vec not result----
            vec_notok.append(result)

    #----dataframe result----
    vec_ok = pd.concat(vec_ok)

    #----dataframe not result----
    vec_notok = pd.concat(vec_notok)

    return [vec, vec_ok, vec_notok]

#------------------------REDUCE CLUSTER------------------------
def notreduce_cluster(dataframe, labels_unique):

    vec = []
    vec_ok = []

    for i in labels_unique:

      result = dataframe[dataframe['cluster'] == i]

      vec_result = []

      indexes_result = result.index

      for j in indexes_result:

            vec_result.append([result.loc[j]['latitude'], result.loc[j]['longitude']])

      #----vec result----
      vec_ok.append(result)
      
      #----vec lat and long----
      vec.append(vec_result)

    #----dataframe result----
    vec_ok = pd.concat(vec_ok)

    return [vec, vec_ok]

#******************************************************************

#------------------------data location-----------------------

# flood-features folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/flood-features/data/'

data = pd.read_csv(local + 'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_countOK.csv')

print(data)

#data = data[data['counts'] != 1]

# reset index of the data
data = data.reset_index()

data = data.drop('Unnamed: 0', 1)

# COORDINATES
print('----------COORDINATES----------')
print(data)

# DESCRIBE FLOODING POINTS
print('----------DESCRIBE FLOODING POINTS----------')
print(data['counts'].describe())

# MEDIAN OF FLOODING POINTS
print('----------MEDIAN OF FLOODING POINTS----------')
print(data['counts'].median())

colunas = ['algorithm', 'number_of_clusters', 'value_of_distance']


# group formation distance list (empirical)
vector_formation_dist = [0.1, 0.2, 0.24, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 
                        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
                        3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4,
                        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5]

# group formation distance list (statistical)
#vector_formation_dist = [0.24]

# group formation distance obtained from best model
# vector_formation_dist = [0.9]

vector_result = []

for v in vector_formation_dist:
      
      df_result_cluster = pd.DataFrame(columns = colunas)
      
      #---------------------DBSCAN---------------------
      try:
            print('-------------------------CLUSTERS DBSCAN--------------------------------')

            df_dbscan = data
            df_dbscan = df_dbscan.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_dbscan[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            #---------------------DBSCAN---------------------
            dbscan = DBSCAN(eps=v, min_samples=1, metric='precomputed')
            rotulo_dbscan = dbscan.fit(distance_matrix)

            #---------------------DBSCAN LABELS---------------------
            df_dbscan['cluster'] = rotulo_dbscan.labels_

            #---------------------DBSCAN UNIQUE LABELS---------------------
            labels_unique_dbscan = np.unique(df_dbscan['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_dbscan = notreduce_cluster(df_dbscan, labels_unique_dbscan)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_dbscan[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_dbscan[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_dbscan = rc_dbscan[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_dbscan.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_dbscan.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_dbscan.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[0, 'algorithm'] = 'cluster_dbscan'
            df_result_cluster.loc[0, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[0, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[0, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[0, 'number_of_clusters'] = len(labels_unique_dbscan)
            df_result_cluster.loc[0, 'value_of_distance'] = v

            print(df_result_cluster)
            
            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue
            
      #---------------------CLUSTERS AGGLOMERATIVE - SINGLE---------------------
      try:
            print('-------------------------CLUSTERS AGGLOMERATIVE - SINGLE--------------------------------')

            df_agglomerative = data
            df_agglomerative = df_agglomerative.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_agglomerative[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            #---------------------CLUSTERS AGGLOMERATIVE - SINGLE---------------------
            
            agglomerative_clustering = AgglomerativeClustering(n_clusters = None ,affinity='precomputed', linkage='single', distance_threshold=v,compute_full_tree=True)  
            rotulo_agglomerative = agglomerative_clustering.fit(distance_matrix)

            #---------------------CLUSTERS AGGLOMERATIVE - SINGLE LABELS---------------------
            df_agglomerative['cluster'] = rotulo_agglomerative.labels_

            #---------------------CLUSTERS AGGLOMERATIVE - SINGLE UNIQUE LABELS---------------------
            labels_unique_agglomerative = np.unique(df_agglomerative['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_agglomerative = notreduce_cluster(df_agglomerative, labels_unique_agglomerative)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_agglomerative[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_agglomerative[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_agglomerative = rc_agglomerative[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_agglomerative.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_agglomerative.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_agglomerative.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[1, 'algorithm'] = 'cluster_agglomerative - single'
            df_result_cluster.loc[1, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[1, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[1, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[1, 'number_of_clusters'] = len(labels_unique_agglomerative)
            df_result_cluster.loc[1, 'value_of_distance'] = v

            print(df_result_cluster)
            
            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue
      
      #---------------------CLUSTERS AGGLOMERATIVE - AVERAGE---------------------
      try:
            print('-------------------------CLUSTERS AGGLOMERATIVE - AVERAGE--------------------------------')

            df_agglomerative = data
            df_agglomerative = df_agglomerative.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_agglomerative[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            #---------------------CLUSTERS AGGLOMERATIVE - AVERAGE---------------------
            
            agglomerative_clustering = AgglomerativeClustering(n_clusters = None ,affinity='precomputed', linkage='average', distance_threshold=v,compute_full_tree=True)  
            rotulo_agglomerative = agglomerative_clustering.fit(distance_matrix)

            #---------------------CLUSTERS AGGLOMERATIVE - AVERAGE LABELS---------------------
            df_agglomerative['cluster'] = rotulo_agglomerative.labels_

            #---------------------CLUSTERS AGGLOMERATIVE - AVERAGE UNIQUE LABELS---------------------
            labels_unique_agglomerative = np.unique(df_agglomerative['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_agglomerative = notreduce_cluster(df_agglomerative, labels_unique_agglomerative)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_agglomerative[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_agglomerative[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_agglomerative = rc_agglomerative[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_agglomerative.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_agglomerative.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_agglomerative.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[2, 'algorithm'] = 'cluster_agglomerative - average'
            df_result_cluster.loc[2, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[2, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[2, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[2, 'number_of_clusters'] = len(labels_unique_agglomerative)
            df_result_cluster.loc[2, 'value_of_distance'] = v

            print(df_result_cluster)
            
            #---------------------------------------------------
            
            rc_agglomerative[1].to_csv(local + 'ALAGAMENTOS-2015_2019-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_REDUCED_ALAG_point_900m.csv')

            centroids.to_csv(local + 'ALAGAMENTOS-2015_2019-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_REDUCED_ALAG_cluster_900m.csv')

            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue
            
      #---------------------CLUSTERS AGGLOMERATIVE - COMPLETE---------------------
      try:
            print('-------------------------CLUSTERS AGGLOMERATIVE - COMPLETE--------------------------------')

            df_agglomerative = data
            df_agglomerative = df_agglomerative.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_agglomerative[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            #---------------------CLUSTERS AGGLOMERATIVE - COMPLETE---------------------
            
            agglomerative_clustering = AgglomerativeClustering(n_clusters = None ,affinity='precomputed', linkage='complete', distance_threshold=v,compute_full_tree=True)  
            rotulo_agglomerative = agglomerative_clustering.fit(distance_matrix)

            #---------------------CLUSTERS AGGLOMERATIVE - COMPLETE LABELS---------------------
            df_agglomerative['cluster'] = rotulo_agglomerative.labels_

            #---------------------CLUSTERS AGGLOMERATIVE - COMPLETE UNIQUE LABELS---------------------
            labels_unique_agglomerative = np.unique(df_agglomerative['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_agglomerative = notreduce_cluster(df_agglomerative, labels_unique_agglomerative)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_agglomerative[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_agglomerative[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_agglomerative = rc_agglomerative[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_agglomerative.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_agglomerative.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_agglomerative.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[3, 'algorithm'] = 'cluster_agglomerative - average'
            df_result_cluster.loc[3, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[3, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[3, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[3, 'number_of_clusters'] = len(labels_unique_agglomerative)
            df_result_cluster.loc[3, 'value_of_distance'] = v

            print(df_result_cluster)
            
            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue
      
      #---------------------CLUSTERS AGGLOMERATIVE - WARD---------------------
      try:
            print('-------------------------CLUSTERS AGGLOMERATIVE - WARD--------------------------------')

            df_agglomerative = data
            df_agglomerative = df_agglomerative.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_agglomerative[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            thresold = v * 1000
            #---------------------CLUSTERS AGGLOMERATIVE - WARD---------------------
            agglomerative_clustering = AgglomerativeClustering(n_clusters = None, affinity='euclidean', linkage='ward', distance_threshold=thresold, compute_full_tree=True)  
            rotulo_agglomerative = agglomerative_clustering.fit(distance_matrix)

            #---------------------CLUSTERS AGGLOMERATIVE - WARD LABELS---------------------
            df_agglomerative['cluster'] = rotulo_agglomerative.labels_

            #---------------------CLUSTERS AGGLOMERATIVE - WARD UNIQUE LABELS---------------------
            labels_unique_agglomerative = np.unique(df_agglomerative['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_agglomerative = notreduce_cluster(df_agglomerative, labels_unique_agglomerative)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_agglomerative[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_agglomerative[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_agglomerative = rc_agglomerative[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_agglomerative.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_agglomerative.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_agglomerative.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[4, 'algorithm'] = 'cluster_agglomerative - ward'
            df_result_cluster.loc[4, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[4, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[4, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[4, 'number_of_clusters'] = len(labels_unique_agglomerative)
            df_result_cluster.loc[4, 'value_of_distance'] = v

            print(df_result_cluster)
            
            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue
      
      #---------------------CLUSTERS OPTICS---------------------
      try:
            print('-------------------------CLUSTERS OPTICS--------------------------------')

            df_optics = data
            df_optics = df_optics.sample(n = 1433)

            #---------------------DISTANCE MATRIX---------------------
            coordinates = df_optics[['latitude', 'longitude']]
            distance_matrix = squareform(pdist(coordinates, (lambda u,v: haversine(u,v))))

            #---------------------CLUSTERS OPTICS---------------------
            cluster_optics = OPTICS(eps=v, metric='precomputed')
            rotulo_optics = agglomerative_clustering.fit(distance_matrix)

            #---------------------CLUSTERS OPTICS LABELS---------------------
            df_optics['cluster'] = rotulo_optics.labels_

            #---------------------CLUSTERS OPTICS LABELS---------------------
            labels_unique_optics = np.unique(df_optics['cluster'])
            
            #---------------------CLUSTERS REDUCTION---------------------
            rc_optics = notreduce_cluster(df_optics, labels_unique_optics)

            #---------------------CENTROIDS---------------------
            clusters = pd.Series(rc_optics[0])
            most_central_point = clusters.map(get_most_central_point)

            latitudes, longitudes = zip(*most_central_point)
            points = pd.DataFrame({'longitude':longitudes, 'latitude':latitudes})

            centroids = get_Distance(rc_optics[0], points, v)

            #---------------------COUNT CLUSTERS---------------------
            vec_ok_optics = rc_optics[1].groupby(['cluster']).size().reset_index(name='counts')

            #---------------------EVALUATION METRICS---------------------
            silhoutte_metric = metrics.silhouette_score(distance_matrix, rotulo_agglomerative.labels_)
            calinski_harabasz_metric = metrics.calinski_harabasz_score(distance_matrix, rotulo_agglomerative.labels_)
            davies_bouldin_metric = metrics.davies_bouldin_score(distance_matrix, rotulo_agglomerative.labels_)

            #---------------------DATAFRAME---------------------
            df_result_cluster.loc[5, 'algorithm'] = 'cluster_optics'
            df_result_cluster.loc[5, 'silhoutte_metric'] = silhoutte_metric
            df_result_cluster.loc[5, 'calinski_harabasz_metric'] = calinski_harabasz_metric
            df_result_cluster.loc[5, 'davies_bouldin_metric'] = davies_bouldin_metric
            df_result_cluster.loc[5, 'number_of_clusters'] = len(labels_unique_agglomerative)
            df_result_cluster.loc[5, 'value_of_distance'] = v

            print(df_result_cluster)
            
            print('------------------------------------------------------------------------')
      except:
            print("Found error!")
            continue

      
      vector_result.append(df_result_cluster)
      
      result = pd.concat(vector_result)

result.to_csv(local+'results_clustering.csv')   

#------------------------END OF EXECUTION------------------------

end = time.time()

print('End of Execution: ',end - start)
# ===============================================================