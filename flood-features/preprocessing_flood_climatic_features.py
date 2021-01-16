from inside_shapefile import InsideShape
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from haversine import Haversine 
from datetime import datetime, timedelta, date
import time
import seaborn as sns
import matplotlib.pyplot as plt

# flood-features folder location
flood_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/flood-features/data/'

# meteorological-features folder location
climatic_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/meteorological-features/data/'

# root folder location
root_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/'
 
start = time.time()

# merging flood and climatic features
def arrange_flood_features(floods, climatics):

    floods = floods.reset_index(drop=True)
    
    #floods['data'] = floods.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)
    #climatics['data'] = climatics.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

    climatics[['horas','minutos','segundos']] = climatics.hora.str.split(':',expand=True)

    print(climatics)
    print(floods)
    
    for i in range(len(floods)):
        
        print('-------------------------------------------------------')
        print(i)
        print('Date: ',floods.loc[i]['data'])
        print('Initial period: ',floods.loc[i]['periodo_inicial'])
        print('Final period: ',floods.loc[i]['periodo_final'])
        print('Initial hour: ',floods.loc[i]['hora_inicial'])
        
        initial_hour = str(floods.loc[i]['hora_inicial']).zfill(2)
        
        print('Initial hour processed: ',initial_hour)

        climatics_hour = climatics[climatics['data'] == floods.loc[i]['data']]

        climatics_hour = climatics_hour[climatics_hour['horas'] == initial_hour]

        print(climatics_hour)
        
        # temperature
        floods.loc[i,'temperatura'] = climatics_hour['temperatura'].values[0]
        
        # humidity
        floods.loc[i,'umidade'] = climatics_hour['umidade'].values[0]
        
        # dew point temperature
        floods.loc[i,'temperatura_ponto_orvalho'] = climatics_hour['temperatura_ponto_orvalho'].values[0]
        
        # maximum temperature
        floods.loc[i,'temperatura_maxima'] = climatics_hour['temperatura_maxima'].values[0]
        
        # minimum temperature
        floods.loc[i,'temperatura_minima'] = climatics_hour['temperatura_minima'].values[0]
        
        # maximum dew point temperature
        floods.loc[i,'temperatura_maxima_ponto_orvalho'] = climatics_hour['temperatura_maxima_ponto_orvalho'].values[0]
        
        # minimum dew point temperature
        floods.loc[i,'temperatura_minima_ponto_orvalho'] = climatics_hour['temperatura_minima_ponto_orvalho'].values[0]
        
        # maximum humidity
        floods.loc[i,'umidade_maxima'] = climatics_hour['umidade_maxima'].values[0]
        
        # minimum humidity
        floods.loc[i,'umidade_minima'] = climatics_hour['umidade_minima'].values[0]
        
        # atmospheric pressure
        floods.loc[i,'pressao_atmosferica'] = climatics_hour['pressao_atmosferica'].values[0]
        
        # wind speed
        floods.loc[i,'vento_velocidade'] = climatics_hour['vento_velocidade'].values[0]
        
        # wind direction
        floods.loc[i,'vento_direcao'] = climatics_hour['vento_direcao'].values[0]
        
        # precipitation
        floods.loc[i,'precipitacao'] = climatics_hour['precipitacao'].values[0]
        
        # maximum gusting wind
        floods.loc[i,'vento_rajada_maxima'] = climatics_hour['vento_rajada_maxima'].values[0]
        
        # maximum atmospheric pressure
        floods.loc[i,'pressao_atmosferica_maxima'] = climatics_hour['pressao_atmosferica_maxima'].values[0]
        
        # minimum atmospheric pressure
        floods.loc[i,'pressao_atmosferica_minima'] = climatics_hour['pressao_atmosferica_minima'].values[0]

    return floods

# flooding processing
def process_floods(floods):

    minimum_limit = datetime.strptime('2015-01-01', '%Y-%m-%d')
    maximum_limit = datetime.strptime('2019-10-01', '%Y-%m-%d')

    floods['data_aux'] = floods.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)
    
    floods = floods[(floods['data_aux'] >= minimum_limit) & (floods['data_aux'] <= maximum_limit)]

    #floods[['hora_inicial','minute_inicial']] = floods.periodo_inicial.str.split(':',expand=True)
    
    #floods[['hora_final','minute_final']] = floods.periodo_final.str.split(':',expand=True)
    
    #-------------------------Saving null values----------------------------

    null_floods = floods[floods.isnull().any(axis=1)]

    null_floods = null_floods.reset_index(drop=True)

    #-------------------------Removing null values----------------------------
    floods.dropna(axis=0, inplace=True)

    floods = floods.reset_index(drop=True)

    #-----Checking when the final period is shorter than the initial period-----
    
    # initial period
    floods['periodo_inicial'] = floods.apply(lambda x: x.periodo_inicial.replace(' ', ''), axis=1)
    
    # final period
    floods['periodo_final'] = floods.apply(lambda x: x.periodo_final.replace(' ', ''), axis=1)
    
    # initial period aux
    floods['periodo_inicial_aux'] = floods.apply(lambda x: datetime.strptime(x['periodo_inicial'], '%H:%M'), axis=1)
    
    # final period aux
    floods['periodo_final_aux'] = floods.apply(lambda x: datetime.strptime(x['periodo_final'], '%H:%M'), axis=1)
    
    floods['need_other_day'] = floods.apply(lambda x: 1 if x.periodo_inicial_aux > x.periodo_final_aux else 0, axis=1)

    for i in range(len(floods)):
        
        if floods.loc[i]['need_other_day']:
            
            tam = len(floods) -1
            tam_next = len(floods) 

            #print('----------')
            #print(floods.loc[i]['data'])
            #print((floods.loc[i]['data_aux'] + timedelta(days = 1)).strftime('%Y-%m-%d'))

            #-------------New Date-------------
            floods.loc[tam_next, 'data']  = (floods.loc[i]['data_aux'] + timedelta(days = 1)).strftime('%Y-%m-%d')
            floods.loc[tam_next, 'periodo'] = 'De 00:00 a '+floods.loc[i]['periodo_final']
            floods.loc[tam_next, 'longitude'] = floods.loc[i]['longitude']
            floods.loc[tam_next, 'latitude'] = floods.loc[i]['latitude']
            floods.loc[tam_next, 'periodo_inicial'] = '00:00'
            floods.loc[tam_next, 'periodo_final'] = floods.loc[i]['periodo_final']

            #-------------Corrected Date--------
            floods.loc[i, 'periodo'] = 'De '+floods.loc[i]['periodo_inicial']+' a 23:59'
            floods.loc[i, 'periodo_final'] = '23:59'

    floods = floods[['data', 'periodo', 'longitude', 'latitude', 'periodo_inicial', 'periodo_final']]

    floods['periodo_inicial_full_aux'] = floods.apply(lambda x: x['data'] + ' ' + x['periodo_inicial'], axis=1)
    floods['periodo_inicial_full_aux'] = floods.apply(lambda x: datetime.strptime(x['periodo_inicial_full_aux'], '%Y-%m-%d %H:%M'), axis=1)
    floods['periodo_final_full_aux'] = floods.apply(lambda x: x['data'] + ' ' + x['periodo_final'], axis=1)
    floods['periodo_final_full_aux'] = floods.apply(lambda x: datetime.strptime(x['periodo_final_full_aux'], '%Y-%m-%d %H:%M'), axis=1)
    
    #-----------------------------flooding duration-----------------------------

    floods['duracao_alagamento'] = floods.apply(lambda x: x['periodo_final_full_aux'] - x['periodo_inicial_full_aux'], axis=1)
    
    floods = floods[['data', 'periodo', 'longitude', 'latitude', 'periodo_inicial', 'periodo_final', 'duracao_alagamento']]

    during_floods = floods['duracao_alagamento']

    during_floods = during_floods.describe()

    #mean time flooding = 2:16:21
    print(during_floods)
    #-------------------------------------------------------------------------------
    
    #------------------------Treatment of null final periods------------------------

    null_floods['periodo_inicial'] = null_floods.apply(lambda x: x.periodo_inicial.replace(' ', ''), axis=1)
    null_floods['periodo_inicial_full_aux'] = null_floods.apply(lambda x: x['data'] + ' ' + x['periodo_inicial'], axis=1)
    null_floods['periodo_inicial_full_aux'] = null_floods.apply(lambda x: datetime.strptime(x['periodo_inicial_full_aux'], '%Y-%m-%d %H:%M'), axis=1)
    null_floods['periodo_final_full_aux'] = null_floods.apply(lambda x: (x['periodo_inicial_full_aux'] + timedelta(hours=2, minutes=16)).strftime('%Y-%m-%d %H:%M'), axis=1)

    for i in range(len(null_floods)):
        next_size = len(floods) 
        
        aux = null_floods.loc[i]['periodo_final_full_aux'].split(' ')

        floods.loc[next_size, 'data'] = null_floods.loc[i]['data']
        floods.loc[next_size, 'periodo'] = null_floods.loc[i]['periodo'] + ' ' + aux[1]
        floods.loc[next_size, 'periodo_inicial'] = null_floods.loc[i]['periodo_inicial']
        floods.loc[next_size, 'periodo_final'] = aux[1]
        floods.loc[next_size, 'latitude'] = null_floods.loc[i]['latitude']
        floods.loc[next_size, 'longitude'] = null_floods.loc[i]['longitude']
        floods.loc[next_size, 'duracao_alagamento'] = timedelta(hours=2, minutes=16)
    
    return floods

# processing of temporal information of flooding occurrences
def arrange_hours_floods(floods):

    columns = ['longitude', 'latitude','data', 
               'periodo','periodo_inicial',
               'periodo_final','hora_inicial',
               'minute_inicial', 'hora_final',
               'minute_final']

    dataframe_floods = pd.DataFrame(columns = columns)
    
    # start hour and start minute
    floods[['hora_inicial','minute_inicial']] = floods.periodo_inicial.str.split(':',expand=True)
    
    # end hour and end minute
    floods[['hora_final','minute_final']] = floods.periodo_final.str.split(':',expand=True)

    for i in range(len(floods)):
        
        #----------Arrange Data---------
        date_aux = floods.loc[i]['data'].split('-')

        date_aux = date_aux[2] + '/' + date_aux[1] + '/' + date_aux[0]

        #----------Arrange Hours---------

        size = len(dataframe_floods)

        hr = int(floods.loc[i]['hora_final']) - int(floods.loc[i]['hora_inicial'])

        print('-----------------------------------------------------------')
        print(i)
        print('Initial hour: ', floods.loc[i]['hora_inicial'])
        print('Initial minutes: ', floods.loc[i]['minute_inicial'])
        print('Final hour: ', floods.loc[i]['hora_final'])
        print('Final minutes: ', floods.loc[i]['minute_final'])
        print('Diference: ', hr)
        print('-----------------------------------------------------------')


        if hr == 0:
            dataframe_floods.loc[size, 'data'] = date_aux
            dataframe_floods.loc[size, 'longitude'] = floods.loc[i]['longitude']
            dataframe_floods.loc[size, 'latitude'] = floods.loc[i]['latitude']
            dataframe_floods.loc[size, 'periodo'] = floods.loc[i]['periodo']
            dataframe_floods.loc[size, 'periodo_inicial'] = floods.loc[i]['periodo_inicial']
            dataframe_floods.loc[size, 'periodo_final'] = floods.loc[i]['periodo_final']
            dataframe_floods.loc[size, 'hora_inicial'] = floods.loc[i]['hora_inicial']
            dataframe_floods.loc[size, 'minute_inicial'] = floods.loc[i]['minute_inicial']
            dataframe_floods.loc[size, 'hora_final'] = floods.loc[i]['hora_final']
            dataframe_floods.loc[size, 'minute_final'] = floods.loc[i]['minute_final']
        
        else:

            dataframe_floods.loc[size, 'data'] = date_aux
            dataframe_floods.loc[size, 'longitude'] = floods.loc[i]['longitude']
            dataframe_floods.loc[size, 'latitude'] = floods.loc[i]['latitude']
            dataframe_floods.loc[size, 'periodo'] = floods.loc[i]['periodo']
            dataframe_floods.loc[size, 'periodo_inicial'] = floods.loc[i]['periodo_inicial']
            dataframe_floods.loc[size, 'periodo_final'] = floods.loc[i]['hora_inicial'] +':59'
            dataframe_floods.loc[size, 'hora_inicial'] = floods.loc[i]['hora_inicial']
            dataframe_floods.loc[size, 'minute_inicial'] = floods.loc[i]['minute_inicial']
            dataframe_floods.loc[size, 'hora_final'] = floods.loc[i]['hora_inicial']
            dataframe_floods.loc[size, 'minute_final'] = '59'

            cont = 1
            while(cont <= hr):

                hr_end = int(floods.loc[i]['hora_inicial']) + cont

                hr_end = str(hr_end).zfill(2)

                if hr_end != floods.loc[i]['hora_final']:
                    
                    dataframe_floods.loc[size, 'data'] = date_aux
                    dataframe_floods.loc[size, 'longitude'] = floods.loc[i]['longitude']
                    dataframe_floods.loc[size, 'latitude'] = floods.loc[i]['latitude']
                    dataframe_floods.loc[size, 'periodo'] = floods.loc[i]['periodo']
                    dataframe_floods.loc[size, 'periodo_inicial'] = str(hr_end) + ':00'
                    dataframe_floods.loc[size, 'periodo_final'] = str(hr_end) + ':59'
                    dataframe_floods.loc[size, 'hora_inicial'] = str(hr_end)
                    dataframe_floods.loc[size, 'minute_inicial'] = '00'
                    dataframe_floods.loc[size, 'hora_final'] = str(hr_end)
                    dataframe_floods.loc[size, 'minute_final'] = '59'
                
                else:   
                    dataframe_floods.loc[size, 'data'] = date_aux
                    dataframe_floods.loc[size, 'longitude'] = floods.loc[i]['longitude']
                    dataframe_floods.loc[size, 'latitude'] = floods.loc[i]['latitude']
                    dataframe_floods.loc[size, 'periodo'] = floods.loc[i]['periodo']
                    dataframe_floods.loc[size, 'periodo_inicial'] = str(hr_end) + ':00'
                    dataframe_floods.loc[size, 'periodo_final'] = floods.loc[i]['periodo_final']
                    dataframe_floods.loc[size, 'hora_inicial'] = str(hr_end)
                    dataframe_floods.loc[size, 'minute_inicial'] = '00'
                    dataframe_floods.loc[size, 'hora_final'] = floods.loc[i]['hora_final']
                    dataframe_floods.loc[size, 'minute_final'] = floods.loc[i]['minute_final']
                
                cont+=1
    
    return dataframe_floods

#==========================================================================
#----------------------FLOODING EXPERIMENTS----------------------

# csv with flood ocurrences

flood = pd.read_csv(flood_location + 'alagamentos_EXTEND2018_2019_COORDS.csv')

# date, period, initial_period, final period, latitude, longitude, address, reference, preprocessing_address

flood = flood[['data','periodo','periodo_inicial', 'periodo_final', 'latitude', 'longitude', 'endereco_CERTO', 'referencia_CERTA', 'endereco_formatado']]

#--------CHECKING WHETHER FLOODING IS WITHIN THE CITY OF SÃO PAULO----------

# shapefile of the city of São Paulo

shape = gpd.read_file(r''+flood_location+'Sao_Paulo_city_WGS84.shp')

# Is it inside a city of São Paulo?

flood_inside = InsideShape(flood, shape).geographic_information    

flood_inside.reset_index(drop=True, inplace=True)

flood_inside.to_csv(flood_location+'ALAGAMENTOS-2018_2019-GEO_CERTO_INSIDE.csv')

print(flood_inside)

#----------------------PREPROCESSING FLOODS----------------------

# csv with flood ocurrences

old_floods = pd.read_csv(flood_location+'ALAGAMENTOS-GEO_CERTO_ALL_INSIDE.csv')

# date, period, initial_period, final period, latitude, longitude
old_floods = old_floods[['data', 'periodo', 'periodo_inicial', 'periodo_final', 'latitude', 'longitude']]

new_floods = pd.read_csv(flood_location+'ALAGAMENTOS-2018_2019-GEO_CERTO_INSIDE.csv')

# date, period, initial_period, final period, latitude, longitude
new_floods = new_floods[['data', 'periodo', 'periodo_inicial', 'periodo_final', 'latitude', 'longitude']]

floods = old_floods.append(new_floods)

floods.reset_index(drop=True, inplace=True)

floods = process_floods(floods)

floods.to_csv(flood_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_OK.csv')

floods = arrange_hours_floods(floods)

print(floods)

floods.to_csv(flood_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS.csv')

#==========================================================================

#---------------------------ARRANGE FEATURES -LAT/LON------------------------

feat_climatics = pd.read_csv(climatic_location +'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE.csv')

feat_floods = pd.read_csv(flood_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS.csv')

feat_floods['data'] = feat_floods.apply(lambda x: datetime.strptime(x['data'], '%d/%m/%Y'), axis=1)

feat_climatics['data'] = feat_climatics.apply(lambda x: datetime.strptime(x['data'], '%d/%m/%Y'), axis=1)

#-----------teste-----------
#lim_min = datetime.strptime('2018-10-20', '%Y-%m-%d')

lim_min = datetime.strptime('2015-01-01', '%Y-%m-%d')
lim_max = datetime.strptime('2018-10-30', '%Y-%m-%d')

feat_floods = feat_floods[(feat_floods['data'] >= lim_min) & (feat_floods['data'] <= lim_max)]

feat_floods.to_csv(flood_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_CERTO.csv')

feat_climatics.to_csv(climatic_location+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE_CERTO.csv')

#---------------------------ARRANGE FLOOD FEATURES------------------------

feat_floods = feat_floods[['longitude','latitude','data',
                           'periodo','periodo_inicial','periodo_final',
                           'hora_inicial','minute_inicial','hora_final',
                           'minute_final']]
print(feat_floods)

feat_climatics = feat_climatics[['datahora','data', 'hora', 'temperatura',
                                 'umidade','temperatura_ponto_orvalho', 'temperatura_maxima',
                                 'temperatura_minima','temperatura_maxima_ponto_orvalho','temperatura_minima_ponto_orvalho',
                                 'umidade_maxima', 'umidade_minima', 'pressao_atmosferica',
                                 'vento_velocidade', 'vento_direcao', 'precipitacao',
                                 'vento_rajada_maxima', 'pressao_atmosferica_maxima',
                                 'pressao_atmosferica_minima']]

print(feat_climatics)

# flood points + climatic features

arrange_floods = arrange_flood_features(feat_floods, feat_climatics)

print(arrange_floods)

arrange_floods.to_csv(flood_location+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv')

end = time.time()

print('Execution time: ',end - start)
# =============================================================================

