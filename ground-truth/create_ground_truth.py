import pandas as pd
from haversine import Haversine
from datetime import datetime, timedelta, date
import time
import geopandas as gpd
from inside_shapefile import InsideShape

#-------------TWEETS INSIDE CLUSTERS-------------
def tweet_inside_cluster(tweets, cluster):

    columns = ['id_str','created_at','longitude',
               'latitude','data','hora','text',
               'related','inside_cluster']

    dataframe = pd.DataFrame(columns = columns)

    # loop of the tweets data
    #i = 1610
    #while i <= len(tweets):
    for i in range(len(tweets)):
        
        inside_cluster = 0

        # loop of the cluster data
        for j in range(len(cluster)):

            distance_flood = Haversine([tweets.loc[i]['lon'],tweets.loc[i]['lat']], [cluster.loc[j]['longitude'],cluster.loc[j]['latitude']]).meters

            if distance_flood <= 900:
                
                inside_cluster = 1
                break

        print(i)
        print(inside_cluster)
        
        size = len(dataframe)

        dataframe.loc[size, 'id_str'] = tweets.loc[i]['id_str']
        dataframe.loc[size, 'created_at'] = tweets.loc[i]['created_at']
        dataframe.loc[size, 'longitude'] = tweets.loc[i]['lon']
        dataframe.loc[size, 'latitude'] = tweets.loc[i]['lat']
        dataframe.loc[size, 'data'] = tweets.loc[i]['data']
        dataframe.loc[size, 'hora'] = tweets.loc[i]['hora']
        dataframe.loc[size, 'text'] = tweets.loc[i]['text']
        dataframe.loc[size, 'related'] = tweets.loc[i]['related']
        dataframe.loc[size, 'inside_cluster'] = inside_cluster

        print('----------------')

        #i +=1
    return dataframe

#-------------GET ALL DAYS DATES BETWEEN TWO DATES-------------
def getAlldays(minimum_limit, maximum_limit):
    
    minimum_date = minimum_limit.split("/")

    maximum_date = maximum_limit.split("/")

    start_date = date(int(minimum_date[2]), int(minimum_date[1]), int(minimum_date[0]))
    end_date = date(int(maximum_date[2]), int(maximum_date[1]), int(maximum_date[0]))

    # Diference between end of the date and the initial of the date
    diference = end_date - start_date 
    
    days = []
    for i in range(diference.days + 1):
        day = start_date + timedelta(days=i)
        days.append(day.strftime('%d/%m/%Y'))
    
    return days

#-------------Processing of events without flooding-------------
def processing_without_floods(floods, climatics):

    floods['data'] = floods.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

    floods.data = floods.data.astype(str)
    
    climatics['data'] = climatics.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

    climatics.data = climatics.data.astype(str)
    
    climatics[['horas','minutos','segundos']] = climatics.hora.str.split(':',expand=True)

    #--------------all dates range--------------
    start_date_flood = date(2015, 1, 1)   # start date
    end_date_flood = date(2018, 10, 30)   # end date

    delta = end_date_flood - start_date_flood       # as timedelta

    vec_dates = []

    for i in range(delta.days + 1):
        day = start_date_flood + timedelta(days=i)
        
        vec_dates.append(day.strftime("%Y-%m-%d"))

    print('All the dates: ', len(vec_dates))

    #--------------all data flooding--------------
    unique_dates = floods['data'].unique()

    print('Flooding dates: ', len(unique_dates))
    
    #--------------all data without flooding--------------
    date_without_floods = []

    for d in vec_dates:
        if d not in unique_dates:
            date_without_floods.append(d)

    print('All the dates without flooding: ', len(date_without_floods))

    #--------------all point alagamento--------------
    
    colunas = ['data','horas','minutos','segundos',
               'temperatura', 'umidade', 'temperatura_ponto_orvalho',
               'temperatura_maxima', 'temperatura_minima',
               'temperatura_maxima_ponto_orvalho', 'temperatura_minima_ponto_orvalho',
               'umidade_maxima', 'umidade_minima', 'vento_velocidade',
               'vento_direcao', 'precipitacao', 'vento_rajada_maxima',
               'pressao_atmosferica','pressao_atmosferica_maxima',
               'pressao_atmosferica_minima']

    features_without_flood = pd.DataFrame(columns = colunas)

    climatics['data'] = climatics.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

    for i in date_without_floods:

        #----------Arrange Date---------
        date_aux = i.split('-')

        date_aux = date_aux[0] + '/' + date_aux[1] + '/' + date_aux[2]

        date_aux = datetime.strptime(date_aux, '%Y/%m/%d')

        print(date_aux)

        climatics_hour = climatics[climatics['data'] == date_aux]

        print(climatics_hour)

        cont = 0

        while cont < 24:

            hour = str(cont).zfill(2)

            print(hour)

            climatics_hour_aux = climatics_hour[climatics_hour['horas'] == hour]
            
            size = len(features_without_flood)
            
            features_without_flood.loc[size,'data'] = date_aux
            features_without_flood.loc[size,'horas'] = hour
            features_without_flood.loc[size,'minutos'] = '00'
            features_without_flood.loc[size,'segundos'] = '00'

            features_without_flood.loc[size,'temperatura'] = climatics_hour_aux['temperatura'].values[0]
            features_without_flood.loc[size,'umidade'] = climatics_hour_aux['umidade'].values[0]
            features_without_flood.loc[size,'temperatura_ponto_orvalho'] = climatics_hour_aux['temperatura_ponto_orvalho'].values[0]
            features_without_flood.loc[size,'temperatura_maxima'] = climatics_hour_aux['temperatura_maxima'].values[0]
            features_without_flood.loc[size,'temperatura_minima'] = climatics_hour_aux['temperatura_minima'].values[0]
            features_without_flood.loc[size,'temperatura_maxima_ponto_orvalho'] = climatics_hour_aux['temperatura_maxima_ponto_orvalho'].values[0]
            features_without_flood.loc[size,'temperatura_minima_ponto_orvalho'] = climatics_hour_aux['temperatura_minima_ponto_orvalho'].values[0]
            features_without_flood.loc[size,'umidade_maxima'] = climatics_hour_aux['umidade_maxima'].values[0]
            features_without_flood.loc[size,'umidade_minima'] = climatics_hour_aux['umidade_minima'].values[0]
            features_without_flood.loc[size,'vento_velocidade'] = climatics_hour_aux['vento_velocidade'].values[0]
            features_without_flood.loc[size,'vento_direcao'] = climatics_hour_aux['vento_direcao'].values[0]
            features_without_flood.loc[size,'precipitacao'] = climatics_hour_aux['precipitacao'].values[0]
            features_without_flood.loc[size,'vento_rajada_maxima'] = climatics_hour_aux['vento_rajada_maxima'].values[0]
            features_without_flood.loc[size,'pressao_atmosferica'] = climatics_hour_aux['pressao_atmosferica'].values[0]
            features_without_flood.loc[size,'pressao_atmosferica_maxima'] = climatics_hour_aux['pressao_atmosferica_maxima'].values[0]
            features_without_flood.loc[size,'pressao_atmosferica_minima'] = climatics_hour_aux['pressao_atmosferica_minima'].values[0]

            print(features_without_flood)

            cont+=1

        print('------------------------')
    return features_without_flood

#--------------preprocessing of the Twitter messages--------------
def preprocessing_tweets(tweets, cluster):
        
    tweets['created_at'] = tweets.apply(lambda x: datetime.strptime(x['created_at'], '%Y-%m-%d %H:%M:%S'), axis=1)

    print(tweets['created_at'])

    tweets['created_at'] = tweets.apply(lambda x: (x['created_at'] + timedelta(hours = -3)).strftime('%Y-%m-%d %H:%M:%S'), axis=1)

    print(tweets['created_at'])

    #tweets['created_at'] = tweets.apply(lambda x: datetime.strftime(x['created_at'], '%Y-%m-%d %H:%M:%S'), axis=1)

    tweets[['data', 'hora']] = tweets.created_at.str.split(' ',expand=True)

    tweets['data'] = tweets.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

    minimum_limit = datetime.strptime('2016-11-01', '%Y-%m-%d')
    maximum_limit = datetime.strptime('2018-10-30', '%Y-%m-%d')

    tweets = tweets[(tweets['data'] >= minimum_limit) & (tweets['data'] <= maximum_limit)]

    print(tweets)

    tweets = tweets[['id_str','created_at','lon','lat','data','hora','text','related']]

    tweets.reset_index(drop=True, inplace=True)
    
    tweets.to_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE.csv')

    print(tweets)
    
    #-----------------CHECKING WHICH TWEETS ARE WITHIN FLOODING ZONES-----------------
    
    tweets_inside_flood_areas = tweet_inside_cluster(tweets, cluster)

    return tweets_inside_flood_areas

#--------------processing the data about floods--------------
def processing_during_floods(tweets, floods):
    
    #-----------------CHECKING WHICH TWEETS OCCURRED DURING FLOODING-----------------

    tweets['data'] = pd.to_datetime(tweets['data'], format='%Y-%m-%d')

    tweets[['horas','minutos', 'segundos']] = tweets.hora.str.split(':',expand=True)

    tweets['horas'] = tweets.apply(lambda x: x.horas.zfill(2), axis=1)

    tweets = tweets[['id_str', 'created_at','longitude', 'latitude',
                     'data', 'hora', 'text', 'related',
                     'inside_cluster', 'horas', 'minutos',
                     'segundos']]

    print(tweets)

    floods = floods[['longitude', 'latitude', 'data',
                     'periodo', 'periodo_inicial', 'periodo_final',
                     'hora_inicial', 'minute_inicial', 'hora_final',
                     'minute_final', 'precipitacao', 'temperatura',
                     'umidade', 'temperatura_ponto_orvalho', 'temperatura_maxima',
                     'vento_velocidade','vento_velocidade', 'vento_direcao',
                     'vento_rajada_maxima', 'temperatura_minima',
                     'temperatura_maxima_ponto_orvalho', 'temperatura_minima_ponto_orvalho',
                     'umidade_maxima', 'umidade_minima', 'pressao_atmosferica',
                     'pressao_atmosferica_maxima', 'pressao_atmosferica_minima']]

    floods['data'] = pd.to_datetime(floods['data'], format='%Y-%m-%d')

    floods['hora_inicial'] = floods['hora_inicial'].astype(str)

    floods['hora_inicial'] = floods.apply(lambda x: x.hora_inicial.zfill(2), axis=1)

    print(floods)

    #------------------------------------------------------------------------------------
    columns = ['id_str', 'created_at','longitude',
               'latitude', 'data', 'hora_complete',
               'text', 'inside_cluster', 'horas',
               'minutos', 'segundos','temperatura',
               'umidade', 'temperatura_ponto_orvalho',
               'pressao_atmosferica', 'precipitacao',
               'related']

    df_ground_truth_flood = pd.DataFrame(columns = columns)

    for i in range(len(tweets)):

        print(i)

        aux_flood = floods[floods['data'] == tweets.loc[i]['data']]

        aux_flood_hour = aux_flood[aux_flood['hora_inicial'] == tweets.loc[i]['horas']]

        if (len(aux_flood_hour) > 0):
            
            tam = len(df_ground_truth_flood)

            df_ground_truth_flood.loc[tam, 'id_str'] = tweets.loc[i]['id_str']
            df_ground_truth_flood.loc[tam, 'created_at'] = tweets.loc[i]['created_at']
            df_ground_truth_flood.loc[tam, 'longitude'] = tweets.loc[i]['longitude']
            df_ground_truth_flood.loc[tam, 'latitude'] = tweets.loc[i]['latitude']
            df_ground_truth_flood.loc[tam, 'data'] = tweets.loc[i]['data']
            df_ground_truth_flood.loc[tam, 'hora_complete'] = tweets.loc[i]['hora']
            df_ground_truth_flood.loc[tam, 'text'] = tweets.loc[i]['text']
            df_ground_truth_flood.loc[tam, 'horas'] = tweets.loc[i]['horas']
            df_ground_truth_flood.loc[tam, 'minutos'] = tweets.loc[i]['minutos']
            df_ground_truth_flood.loc[tam, 'segundos'] = tweets.loc[i]['segundos']
            
            df_ground_truth_flood.loc[tam, 'precipitacao'] = aux_flood_hour['precipitacao'].values[0]
            df_ground_truth_flood.loc[tam, 'temperatura'] = aux_flood_hour['temperatura'].values[0]
            df_ground_truth_flood.loc[tam, 'umidade'] = aux_flood_hour['umidade'].values[0]
            df_ground_truth_flood.loc[tam, 'temperatura_ponto_orvalho'] = aux_flood_hour['temperatura_ponto_orvalho'].values[0]
            df_ground_truth_flood.loc[tam, 'pressao_atmosferica'] = aux_flood_hour['pressao_atmosferica'].values[0]
            df_ground_truth_flood.loc[tam, 'related'] = tweets.loc[i]['related']
            df_ground_truth_flood.loc[tam, 'inside_cluster'] = tweets.loc[i]['inside_cluster']
            df_ground_truth_flood.loc[tam, 'is_alag'] = 1


        print(df_ground_truth_flood)
        print('----------------------------------------')

    print(df_ground_truth_flood)
    
    return df_ground_truth_flood

#---------------------------ARRANGE NO FLOODS + POINTS/FEATURES (TWEETS)--------------------------

def processing_during_without_floods(tweets, feat_climatics_not_flood):
    
    tweets['data'] = pd.to_datetime(tweets['data'], format='%Y-%m-%d')

    tweets[['horas','minutos', 'segundos']] = tweets.hora.str.split(':',expand=True)

    tweets['horas'] = tweets.apply(lambda x: x.horas.zfill(2), axis=1)

    tweets = tweets[['id_str', 'created_at','longitude', 'latitude',
                     'data', 'hora', 'text', 'related', 'inside_cluster',
                     'horas', 'minutos', 'segundos']]

    print(tweets)

    feat_climatics_not_flood[['data', 'not']] = feat_climatics_not_flood.data.str.split(' ',expand=True)

    feat_climatics_not_flood['data'] = pd.to_datetime(feat_climatics_not_flood['data'], format='%Y-%m-%d')

    feat_climatics_not_flood = feat_climatics_not_flood[['data', 'horas',
    'minutos', 'segundos', 'precipitacao', 'temperatura',
    'umidade', 'temperatura_ponto_orvalho', 'temperatura_maxima',
    'vento_velocidade','vento_velocidade', 'vento_direcao','vento_rajada_maxima',
    'temperatura_minima', 'temperatura_maxima_ponto_orvalho',
    'temperatura_minima_ponto_orvalho', 'umidade_maxima', 'umidade_minima',
    'pressao_atmosferica', 'pressao_atmosferica_maxima',
    'pressao_atmosferica_minima']]

    feat_climatics_not_flood.horas = feat_climatics_not_flood.horas.astype(str)

    feat_climatics_not_flood.minutos = feat_climatics_not_flood.minutos.astype(str)

    feat_climatics_not_flood['horas'] = feat_climatics_not_flood.apply(lambda x: x.horas.zfill(2), axis=1)
        
    feat_climatics_not_flood['minutos'] = feat_climatics_not_flood.apply(lambda x: x.minutos.zfill(2), axis=1)
        
    #------------------------------------------------------------------------------------
    columns = ['id_str', 'created_at','longitude',
               'latitude', 'data', 'hora_complete',
               'text','inside_cluster', 'horas', 'minutos',
               'segundos','temperatura', 'umidade',
               'temperatura_ponto_orvalho', 'pressao_atmosferica',
               'precipitacao','related']

    df_ground_truth = pd.DataFrame(columns = columns)

    for i in range(len(tweets)):

        print(i)

        aux_flood = feat_climatics_not_flood[feat_climatics_not_flood['data'] == tweets.loc[i]['data']]

        aux_flood_hour = aux_flood[aux_flood['horas'] == tweets.loc[i]['horas']]

        if (len(aux_flood_hour) > 0):
            
            tam = len(df_ground_truth)

            df_ground_truth.loc[tam, 'id_str'] = tweets.loc[i]['id_str']
            df_ground_truth.loc[tam, 'created_at'] = tweets.loc[i]['created_at']
            df_ground_truth.loc[tam, 'longitude'] = tweets.loc[i]['longitude']
            df_ground_truth.loc[tam, 'latitude'] = tweets.loc[i]['latitude']
            df_ground_truth.loc[tam, 'data'] = tweets.loc[i]['data']
            df_ground_truth.loc[tam, 'hora_complete'] = tweets.loc[i]['hora']
            df_ground_truth.loc[tam, 'text'] = tweets.loc[i]['text']
            df_ground_truth.loc[tam, 'horas'] = tweets.loc[i]['horas']
            df_ground_truth.loc[tam, 'minutos'] = tweets.loc[i]['minutos']
            df_ground_truth.loc[tam, 'segundos'] = tweets.loc[i]['segundos']
            
            df_ground_truth.loc[tam, 'precipitacao'] = aux_flood_hour['precipitacao'].values[0]
            df_ground_truth.loc[tam, 'temperatura'] = aux_flood_hour['temperatura'].values[0]
            df_ground_truth.loc[tam, 'umidade'] = aux_flood_hour['umidade'].values[0]
            df_ground_truth.loc[tam, 'temperatura_ponto_orvalho'] = aux_flood_hour['temperatura_ponto_orvalho'].values[0]
            df_ground_truth.loc[tam, 'pressao_atmosferica'] = aux_flood_hour['pressao_atmosferica'].values[0]
            df_ground_truth.loc[tam, 'related'] = tweets.loc[i]['related']
            df_ground_truth.loc[tam, 'inside_cluster'] = tweets.loc[i]['inside_cluster']
            df_ground_truth.loc[tam, 'is_alag'] = 0


        print(df_ground_truth)
        print('----------------------------------------')

    print(df_ground_truth)
    
    return df_ground_truth                   
    
#----------------directory of locations----------------

# ground_truth folder location

ground_truth_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/ground-truth/data/'

# textual-features folder location

textual_features_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/'

# coefficient-of-agreement folder location

coefficient_of_agreement_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

# flood-features folder location

flood_features_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/flood-features/data/'

# meteorological-features folder location

climatics_features_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/meteorological-features/data/'

# train_test folder location

train_test_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

#----------------start of the execution----------------

start = time.time()

#----------------preprocessing tweets----------------

tweets = pd.read_csv(coefficient_of_agreement_location + 'THIAGO_NOV2016_NOV2018_judge.csv')

tweets_default = pd.read_csv(textual_features_location + 'tweets_NOV2016_NOV2018_CERTO_keywords.csv')

#----------------arranging the latitude and longitude----------------

for i in range(len(tweets)):

    tweets.loc[i, 'lon'] = tweets_default.loc[i]['lon']
    tweets.loc[i, 'lat'] = tweets_default.loc[i]['lat']
    
print(tweets)

tweets.to_csv(ground_truth_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO.csv')

#----------------preprocessing of the Twitter messages----------------

tweets = pd.read_csv(ground_truth_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO.csv')

cluster = pd.read_csv(flood_features_location + 'ALAGAMENTOS-2015_2019-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_REDUCED_ALAG_cluster_900m.csv')

df_preprocessed_tweets = preprocessing_tweets(tweets, cluster)

df_preprocessed_tweets.to_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster.csv')

#----------------processing the data about floods----------------

preprocessed_tweets = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster.csv')

floods = pd.read_csv(flood_features_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv')

result_about_floods = processing_during_floods(preprocessed_tweets, floods)

result_about_floods.to_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES.csv')

#----------------processing the data about without floods----------------

#-------------ARRANGE NO FLOODS + POINTS/FEATURES (DAYS)------------------

feat_climatics = pd.read_csv(flood_features_location+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE_CERTO.csv')

floods = pd.read_csv(flood_features_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv')

arrange_without_alag = processing_without_floods(floods,feat_climatics)

arrange_without_alag.to_csv(ground_truth_location+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE_CERTO_SEM_ALAGAMENTO.csv')

#-------------------------------------------

#-------------ARRANGE NO FLOODS + POINTS/FEATURES (TWEETS-------------

feat_climatics_not_alag = pd.read_csv(ground_truth_location+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE_CERTO_SEM_ALAGAMENTO.csv')

tweets = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster.csv')

df_ground_truth_without_floods = processing_during_without_floods(tweets, feat_climatics_not_alag)

df_ground_truth_without_floods.to_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_SEM_ALAGAMENTO.csv')

#----------------------------------------------------------------------------------------------

#-----------------------------------GROUND TRUTH-----------------------------------

tw_rel_inside = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_SEM_ALAGAMENTO.csv')

tw_rel_inside2 = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES.csv')

vec_rel = pd.concat([tw_rel_inside, tw_rel_inside2])

vec_rel = vec_rel.reset_index()

vec_rel = vec_rel[['id_str','created_at','longitude','latitude','data','hora_complete','text','inside_cluster','horas','minutos','segundos','temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','related','is_alag']]

vec_rel = vec_rel.reset_index()

vec_rel = vec_rel[['index','id_str','created_at','longitude','latitude','data','hora_complete','text','inside_cluster','horas','minutos','segundos','temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','related','is_alag']]

vec_rel.to_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_COM_SEM_ALAG.csv')

print(vec_rel)


#--------CHECKING WHETHER FLOODING IS WITHIN THE CITY OF SÃO PAULO----------

tweets = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_COM_SEM_ALAG.csv')

print(tweets)

shape = gpd.read_file(r''+ground_truth_location+'Sao_Paulo_city_WGS84.shp')

vec_rel = InsideShape(tweets, shape).geographic_information

vec_rel.reset_index(drop=True, inplace=True)

print(vec_rel)

#----------------------------------

vec = []

print(vec_rel)

tw_rel_inside = vec_rel[vec_rel['is_alag'] == 0.0]

print('----------------------------TARGET 0----------------------------')

tw_rel0_inside = tw_rel_inside[tw_rel_inside['related'] == 0.0]

print('----------tw_rel0_inside0----------')

tw_rel0_inside0 = tw_rel0_inside[tw_rel0_inside['inside_cluster'] == 0.0]

print(tw_rel0_inside0)

tw_rel0_inside0 = tw_rel0_inside0.sample(n = 16, random_state=1)

vec.append(tw_rel0_inside0)

print('----------tw_rel0_inside1----------')

tw_rel0_inside1 = tw_rel0_inside[tw_rel0_inside['inside_cluster'] == 1.0]

print(tw_rel0_inside1)

tw_rel0_inside1 = tw_rel0_inside1.sample(n = 16, random_state=1)

vec.append(tw_rel0_inside1)

tw_rel1_inside = tw_rel_inside[tw_rel_inside['related'] == 1.0]

print('----------tw_rel1_inside0----------')

tw_rel1_inside0 = tw_rel1_inside[tw_rel1_inside['inside_cluster'] == 0.0]

print(tw_rel1_inside0)

tw_rel1_inside0 = tw_rel1_inside0.sample(n = 16, random_state=1)

vec.append(tw_rel1_inside0)

print('----------tw_rel1_inside1----------')

tw_rel1_inside1 = tw_rel1_inside[tw_rel1_inside['inside_cluster'] == 1.0]

print(tw_rel1_inside1)

tw_rel1_inside1 = tw_rel1_inside1.sample(n = 16, random_state=1)

vec.append(tw_rel1_inside1)

#--------------------------------------------------------------------------
tw_rel_inside2 = vec_rel[vec_rel['is_alag'] == 1.0]

print('----------------------------TARGET 1----------------------------')

tw_rel0_inside2 = tw_rel_inside2[tw_rel_inside2['related'] == 0.0]

print('----------tw_rel0_inside0----------')

tw_rel0_inside02 = tw_rel0_inside2[tw_rel0_inside2['inside_cluster'] == 0.0]

print(tw_rel0_inside02)

tw_rel0_inside02 = tw_rel0_inside02.sample(n = 16, random_state=1)

vec.append(tw_rel0_inside02)

print('----------tw_rel0_inside1----------')

tw_rel0_inside12 = tw_rel0_inside2[tw_rel0_inside2['inside_cluster'] == 1.0]

print(tw_rel0_inside12)

tw_rel0_inside12 = tw_rel0_inside12.sample(n = 16, random_state=1)

vec.append(tw_rel0_inside12)

tw_rel1_inside2 = tw_rel_inside2[tw_rel_inside2['related'] == 1.0]

print('----------tw_rel1_inside0----------')

tw_rel1_inside02 = tw_rel1_inside2[tw_rel1_inside2['inside_cluster'] == 0.0]

print(tw_rel1_inside02)

tw_rel1_inside02 = tw_rel1_inside02.sample(n = 16, random_state=1)

vec.append(tw_rel1_inside02)

print('----------tw_rel1_inside1----------')

tw_rel1_inside12 = tw_rel1_inside2[tw_rel1_inside2['inside_cluster'] == 1.0]

print(tw_rel1_inside12)

tw_rel1_inside12 = tw_rel1_inside12.sample(n = 112, random_state=1)

vec.append(tw_rel1_inside12)

vec_df = pd.concat(vec)

vec_df = vec_df.reset_index()

vec_df = vec_df[['index','id_str','created_at','longitude','latitude','data','hora_complete','text','inside_cluster','horas','minutos','segundos','temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','related','is_alag']]

print(vec_df)

vec_df.to_csv(ground_truth_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade.csv')

#--------------------------------------------------------------------------

conjunto_vdd_ok = vec_df[vec_df['index'] != 8559]

conjunto_vdd_ok = conjunto_vdd_ok[conjunto_vdd_ok['index'] != 9318]

print(conjunto_vdd_ok)

conjunto_vdd_ok = conjunto_vdd_ok.sample(n = 222)

conjunto_vdd_ok.to_csv(train_test_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade_OK.csv')


#----------------end of the execution----------------

end = time.time()

print('Execution time: ',end - start)