import pandas as pd
import time
from datetime import datetime, timedelta
# =============================================================================      

start = time.time()

# meteorological-features folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/meteorological-features/data/'


#----------------------------MERGING DATAFRAMES--------------------------------

def merging_data(inmet):

    features = ['TEMPERATURA', 'UMIDADE', 'TEMPERATURA_PONTO_ORVALHO', 'TEMPERATURA_MAX','TEMPERATURA_MIN','TEMPERATURA_MAX_PONTO_ORVALHO','TEMPERATURA_MIN_PONTO_ORVALHO','UMIDADE_MAX', 'UMIDADE_MIN', 'PRESSAO_ATMOSFERICA', 'VENTO_VELOCIDADE', 'VENTO_DIRECAO', 'PRECIPITACAO', 'VENTO_RAJADA_MAX', 'PRESSAO_ATMOSFERICA_MAX', 'PRESSAO_ATMOSFERICA_MIN']

    hours = [None] * 24 

    #-------Dataframe------
    col = ['data', 'hora', 'temperatura', 'umidade', 'temperatura_ponto_orvalho', 'temperatura_maxima','temperatura_minima','temperatura_maxima_ponto_orvalho','temperatura_minima_ponto_orvalho','umidade_maxima', 'umidade_minima', 'pressao_atmosferica', 'vento_velocidade', 'vento_direcao', 'precipitacao', 'vento_rajada_maxima', 'pressao_atmosferica_maxima', 'pressao_atmosferica_minima']

    df_acumulacao = pd.DataFrame(columns = col)

    #******************************************************************************

    mes = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']

    vector_inmet = []

    for i in range(len(inmet)):
        
        #-------Date-------
        
        data  = str(inmet.loc[i]['HORA_UTC']).replace('-','/')
        
        data = data.split(' ')
        
        data = data[0].split('/')
        
        if data[1] == 'jan':
            mes = '01'
        elif data[1] == 'fev':
            mes = '02'
        elif data[1] == 'mar':
            mes = '03'
        elif data[1] == 'abr':
            mes = '04'
        elif data[1] == 'mai':
            mes = '05'
        elif data[1] == 'jun': 
            mes = '06'
        elif data[1] == 'jul':
            mes = '07'
        elif data[1] == 'ago':
            mes = '08'
        elif data[1] == 'set':
            mes = '09'
        elif data[1] == 'out':
            mes = '10'
        elif data[1] == 'nov':
            mes = '11'
        elif data[1] == 'dez':
            mes = '12'

        data = str(data[0]) + '/' + str(mes) + '/' + str(data[2])
        
        print(data)
        
        for t in range(len(hours)):
            
            v = []
            
            for j in features:
            
                hora = j +'-'+ str(t)
                
                if(t != 0):
                
                    hora += '00' 
                
                try:
                
                    medida = str(inmet.loc[i][hora]).replace(',','.')
                    
                    v.append(medida)
                    
                    #print(hora+'-->', float(medida))
                        
                except ValueError as e:
                    print('Erro:', e)
                
                #-------Date-------
                
                hora_aux = hora.split('-')
                
                hora_aux = hora_aux[1].zfill(4)
                
                hora_aux = hora_aux[:2] + ':' + hora_aux[2:] + ':00'
                
                #------------------
            
            print('Data: '+str(data) + ' Hora: '+str(hora_aux))
            #-------Dataframe------
            index_df = len(df_acumulacao) + 1
            
            df_acumulacao.loc[index_df] = [data, hora_aux, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]]
            #------------------

#----------------------------MISSING DATA--------------------------------
def missing_data(df):
    
    df['datahora'] = pd.to_datetime(df['datahora'], format='%d/%m/%Y %H:%M:%S')
    df = df.set_index('datahora')
    
    # temperature
    df['temperatura'] = df.temperatura.interpolate(method='linear')
    
    # humidity
    df['umidade'] = df.umidade.interpolate(method='linear')
    
    # dew point temperature
     
    df['temperatura_ponto_orvalho'] = df.temperatura_ponto_orvalho.interpolate(method='linear')
    
    # max temperature
    df['temperatura_maxima'] = df.temperatura_maxima.interpolate(method='linear')
    
    # min temperature
    df['temperatura_minima'] = df.temperatura_minima.interpolate(method='linear')
    
    # max dew point temperature
    df['temperatura_maxima_ponto_orvalho'] = df.temperatura_maxima_ponto_orvalho.interpolate(method='linear')
    
    # min dew point temperature
    
    df['temperatura_minima_ponto_orvalho'] = df.temperatura_minima_ponto_orvalho.interpolate(method='linear')
    
    # max humidity
    df['umidade_maxima'] = df.umidade_maxima.interpolate(method='linear')
    
    # min humidity
    df['umidade_minima'] = df.umidade_minima.interpolate(method='linear')
    
    # atmospheric pressure
    df['pressao_atmosferica'] = df.pressao_atmosferica.interpolate(method='linear')
    
    # wind speed
    df['vento_velocidade'] = df.vento_velocidade.interpolate(method='linear')
    
    # wind direction
    df['vento_direcao'] = df.vento_direcao.interpolate(method='linear')
    
    # precipitation
    df['precipitacao'] = df.precipitacao.interpolate(method='linear')
    
    # max wind speed
    df['vento_rajada_maxima'] = df.vento_rajada_maxima.interpolate(method='linear')
    
    # max atmospheric pressure
    df['pressao_atmosferica_maxima'] = df.pressao_atmosferica_maxima.interpolate(method='linear')
    
    # min atmospheric pressure
    df['pressao_atmosferica_minima'] = df.pressao_atmosferica_minima.interpolate(method='linear')

    return(df)

#----------------------------MERGING DATAFRAMES--------------------------------

climatic_features = pd.read_csv(local+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK.csv', sep=';')

result_df = merging_data(climatic_features)

print(result_df)
print(len(result_df))
result_df.to_csv(local+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA.csv')

#*************************CONVERSÃO DE FUSO HORÁRIO****************************

inmet = pd.read_csv(local+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA.csv')

limite_minimo = datetime.strptime('01-01-2015 00:00:00', '%d-%m-%Y %H:%M:%S')
limite_maximo = datetime.strptime('30-10-2018 23:59:59', '%d-%m-%Y %H:%M:%S')

inmet['datahora'] = inmet.apply(lambda x: x['data'] + ' ' + x['hora'], axis=1)

inmet['datahora'] = inmet.apply(lambda x: datetime.strptime(x.datahora, '%d/%m/%Y %H:%M:%S'), axis=1)

inmet['datahora'] = inmet.apply(lambda x: x.datahora + timedelta(hours = -3), axis=1)

inmet = inmet[(inmet['datahora'] >= limite_minimo) & (inmet['datahora'] <= limite_maximo)]

inmet['data'] = inmet.apply(lambda x: x.datahora.strftime('%d/%m/%Y'), axis=1)

inmet['hora'] = inmet.apply(lambda x: x.datahora.strftime('%H:%M:%S'), axis=1)

print(inmet)

df_interpolate = missing_data(inmet)

print(df_interpolate)

df_interpolate.to_csv(local+'SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE.csv')
