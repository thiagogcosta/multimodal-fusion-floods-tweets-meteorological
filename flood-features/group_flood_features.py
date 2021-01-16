# -*- coding: utf-8 -*-

import pandas as pd
import time 
# =============================================================================      

start = time.time()

# flood_areas folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/flood-features/data/'

df = pd.read_csv(local+'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv')

print(df)

#---------------GROUP BY COUNT---------------

df_count = df[['latitude', 'longitude']]

df_count = df_count.groupby(['latitude', 'longitude']).size().reset_index(name='counts')

df_count = df_count.sort_values('counts')

print(df_count)

df_count.to_csv(local + 'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_countOK.csv')

#---------------GROUP BY FEATURE---------------

df_feature = df.groupby(['latitude','longitude'])['umidade'].sum().reset_index()

print(df_feature)

df_feature.to_csv(local + 'ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_umidadeOK.csv')

end = time.time()

print('Execution time: ',end - start)
# =============================================================================
