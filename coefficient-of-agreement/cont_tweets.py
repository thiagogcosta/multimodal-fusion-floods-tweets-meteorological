import pandas as pd

# coefficient-of-agreement folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

#-----------------------------------------------------------------------

data_thiago = local + 'THIAGO_NOV2016_NOV2018_KEY.csv'
data_luan = local + 'LUAN_NOV2016_NOV2018_KEY.csv'
data_aysla = local + 'AYSLA_NOV2016_NOV2018_KEY.csv'

df_thiago = pd.read_csv(data_thiago)
df_thiago = df_thiago[['id_str', 'created_at', 'lon', 'lat', 'date', 'hour', 'text', 'related_thiago']]

df_luan = pd.read_csv(data_luan)
df_luan = df_luan[['id_str', 'created_at', 'lon', 'lat', 'date', 'hour', 'text', 'related_luan']]

df_aysla = pd.read_csv(data_aysla)
df_aysla = df_aysla[['id_str', 'created_at', 'lon', 'lat', 'date', 'hour', 'text', 'related_aysla']]

colunas = ['id_str', 'created_at', 'lon', 'lat', 'date', 'hour', 'text', 'related']

#-------------------------------DF thiago judge-------------------------------

print('-------------------------------DF thiago judge-------------------------------')
df_thiago_judge = pd.DataFrame(columns = colunas)

count_same = 0 
count_not_same = 0

for i in range(len(df_thiago)):
    
    if df_thiago.loc[i]['id_str'] == df_luan.loc[i]['id_str'] and df_thiago.loc[i]['id_str'] == df_aysla.loc[i]['id_str']:
        
        df_thiago_judge.loc[i, 'id_str'] = df_thiago.loc[i]['id_str']
        df_thiago_judge.loc[i, 'created_at'] = df_thiago.loc[i]['created_at'] 
        df_thiago_judge.loc[i, 'lon'] = df_thiago.loc[i]['lon'] 
        df_thiago_judge.loc[i, 'lat'] = df_thiago.loc[i]['lat'] 
        df_thiago_judge.loc[i, 'date'] = df_thiago.loc[i]['date'] 
        df_thiago_judge.loc[i, 'hour'] = df_thiago.loc[i]['hour'] 
        df_thiago_judge.loc[i, 'text'] = df_thiago.loc[i]['text'] 
        
        if df_luan.loc[i]['related_luan'] == df_aysla.loc[i]['related_aysla']:
            df_thiago_judge.loc[i, 'related'] = df_luan.loc[i]['related_luan']
            count_same +=1
        else:
            df_thiago_judge.loc[i, 'related'] = df_thiago.loc[i]['related_thiago']
            count_not_same +=1
            
print('count_same: ', count_same)
print('count_not_same: ', count_not_same)

df_thiago_judge.to_csv(local + 'THIAGO_NOV2016_NOV2018_judge.csv')

#-------------------------------DF aysla judge-------------------------------

print('-------------------------------DF aysla judge-------------------------------')
df_aysla_judge = pd.DataFrame(columns = colunas)

count_same = 0 
count_not_same = 0

for i in range(len(df_aysla)):
    
    if df_aysla.loc[i]['id_str'] == df_luan.loc[i]['id_str'] and df_aysla.loc[i]['id_str'] == df_thiago.loc[i]['id_str']:
        
        df_aysla_judge.loc[i, 'id_str'] = df_aysla.loc[i]['id_str']
        df_aysla_judge.loc[i, 'created_at'] = df_aysla.loc[i]['created_at'] 
        df_aysla_judge.loc[i, 'lon'] = df_aysla.loc[i]['lon'] 
        df_aysla_judge.loc[i, 'lat'] = df_aysla.loc[i]['lat'] 
        df_aysla_judge.loc[i, 'date'] = df_aysla.loc[i]['date'] 
        df_aysla_judge.loc[i, 'hour'] = df_aysla.loc[i]['hour'] 
        df_aysla_judge.loc[i, 'text'] = df_aysla.loc[i]['text'] 
        
        if df_luan.loc[i]['related_luan'] == df_thiago.loc[i]['related_thiago']:
            df_aysla_judge.loc[i, 'related'] = df_luan.loc[i]['related_luan']
            count_same +=1
        else:
            df_aysla_judge.loc[i, 'related'] = df_aysla.loc[i]['related_aysla']
            count_not_same +=1
            
print('count_same: ', count_same)
print('count_not_same: ', count_not_same)     

df_aysla_judge.to_csv(local + 'AYSLA_NOV2016_NOV2018_judge.csv')

#-------------------------------DF luan judge-------------------------------

print('-------------------------------DF luan judge-------------------------------')
df_luan_judge = pd.DataFrame(columns = colunas)

count_same = 0 
count_not_same = 0

for i in range(len(df_luan)):
    
    if df_luan.loc[i]['id_str'] == df_thiago.loc[i]['id_str'] and df_luan.loc[i]['id_str'] == df_aysla.loc[i]['id_str']:
        
        df_luan_judge.loc[i, 'id_str'] = df_luan.loc[i]['id_str']
        df_luan_judge.loc[i, 'created_at'] = df_luan.loc[i]['created_at'] 
        df_luan_judge.loc[i, 'lon'] = df_luan.loc[i]['lon'] 
        df_luan_judge.loc[i, 'lat'] = df_luan.loc[i]['lat'] 
        df_luan_judge.loc[i, 'date'] = df_luan.loc[i]['date'] 
        df_luan_judge.loc[i, 'hour'] = df_luan.loc[i]['hour'] 
        df_luan_judge.loc[i, 'text'] = df_luan.loc[i]['text'] 
        
        if df_aysla.loc[i]['related_aysla'] == df_thiago.loc[i]['related_thiago']:
            df_luan_judge.loc[i, 'related'] = df_aysla.loc[i]['related_aysla']
            count_same +=1
        else:
            df_luan_judge.loc[i, 'related'] = df_luan.loc[i]['related_luan']
            count_not_same +=1
            
print('count_same: ', count_same)
print('count_not_same: ', count_not_same)     

df_luan_judge.to_csv(local + 'LUAN_NOV2016_NOV2018_judge.csv')