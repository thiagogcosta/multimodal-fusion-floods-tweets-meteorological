import pandas as pd
import geopandas as gpd
from inside_shapefile import InsideShape

#----------directory location of the features----------

# ground_truth folder location

ground_truth_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/ground-truth/data/'

# train_test folder location

train_test_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

#-------------------------------------------------------------------------
ground_truth = pd.read_csv(ground_truth_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade.csv')

ground_truth = ground_truth[['index', 'id_str','created_at','longitude','latitude','data','hora_complete','text','inside_cluster','horas','minutos','segundos','temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','related','is_alag']]

ids = ground_truth['index'].values

print(ground_truth)

tweets = pd.read_csv(ground_truth_location+'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_COM_SEM_ALAG.csv')

shape = gpd.read_file(r''+ground_truth_location+'Sao_Paulo_city_WGS84.shp')

vec_rel = InsideShape(tweets, shape).geographic_information 

vec_rel.reset_index(drop=True, inplace=True)

print(vec_rel)

vec_rel = vec_rel[~vec_rel['index'].isin(ids)]

print(vec_rel)

#--------------------------MODEL EARLY FUSION--------------------------

vec = []

tw_rel_inside = vec_rel[vec_rel['is_alag'] == 0.0]

print('----------------------------TARGET 0----------------------------')

tw_rel0_inside = tw_rel_inside[tw_rel_inside['related'] == 0.0]

print('----------tw_rel0_inside0----------')

tw_rel0_inside0 = tw_rel0_inside[tw_rel0_inside['inside_cluster'] == 0.0]

tw_rel0_inside0 = tw_rel0_inside0.sample(n = 65, random_state=1)

vec.append(tw_rel0_inside0)

print(tw_rel0_inside0)

print('----------tw_rel0_inside1----------')

tw_rel0_inside1 = tw_rel0_inside[tw_rel0_inside['inside_cluster'] == 1.0]

tw_rel0_inside1 = tw_rel0_inside1.sample(n = 65, random_state=1)

vec.append(tw_rel0_inside1)

print(tw_rel0_inside1)

tw_rel1_inside = tw_rel_inside[tw_rel_inside['related'] == 1.0]

print('----------tw_rel1_inside0----------')

tw_rel1_inside0 = tw_rel1_inside[tw_rel1_inside['inside_cluster'] == 0.0]

tw_rel1_inside0 = tw_rel1_inside0.sample(n = 65, random_state=1)

vec.append(tw_rel1_inside0)

print(tw_rel1_inside0)

print('----------tw_rel1_inside1----------')

tw_rel1_inside1 = tw_rel1_inside[tw_rel1_inside['inside_cluster'] == 1.0]

tw_rel1_inside1 = tw_rel1_inside1.sample(n = 65, random_state=1)

vec.append(tw_rel1_inside1)

print(tw_rel1_inside1)

#--------------------------------------------------------------------------
tw_rel_inside2 = vec_rel[vec_rel['is_alag'] == 1.0]

print('----------------------------TARGET 1----------------------------')

tw_rel0_inside2 = tw_rel_inside2[tw_rel_inside2['related'] == 0.0]

print('----------tw_rel0_inside0----------')

tw_rel0_inside02 = tw_rel0_inside2[tw_rel0_inside2['inside_cluster'] == 0.0]

tw_rel0_inside02 = tw_rel0_inside02.sample(n = 65, random_state=1)

vec.append(tw_rel0_inside02)

print(tw_rel0_inside02)

print('----------tw_rel0_inside1----------')

tw_rel0_inside12 = tw_rel0_inside2[tw_rel0_inside2['inside_cluster'] == 1.0]

tw_rel0_inside12 = tw_rel0_inside12.sample(n = 65, random_state=1)

vec.append(tw_rel0_inside12)

print(tw_rel0_inside12)

tw_rel1_inside2 = tw_rel_inside2[tw_rel_inside2['related'] == 1.0]

print('----------tw_rel1_inside0----------')

tw_rel1_inside02 = tw_rel1_inside2[tw_rel1_inside2['inside_cluster'] == 0.0]

tw_rel1_inside02 = tw_rel1_inside02.sample(n = 65, random_state=1)

vec.append(tw_rel1_inside02)

print(tw_rel1_inside02)

print('----------tw_rel1_inside1----------')

tw_rel1_inside12 = tw_rel1_inside2[tw_rel1_inside2['inside_cluster'] == 1.0]

tw_rel1_inside12 = tw_rel1_inside12.sample(n = 455, random_state=1)

vec.append(tw_rel1_inside12)

print(tw_rel1_inside12)

vec_df = pd.concat(vec)

vec_df = vec_df.reset_index()

vec_df = vec_df[['index','id_str','created_at','longitude','latitude','data','hora_complete','text','inside_cluster','horas','minutos','segundos','temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','related','is_alag']]

print(vec_df)

vec_df.to_csv(train_test_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_early_fusion.csv')

#----------------------------------------------------------------------------------

#--------------------------MODEL LATE FUSION--------------------------

#-------------MODEL FLOOD-------------

print('------------------is_alag == 0.0------------------')
tw_is_alag = vec_rel[vec_rel['is_alag'] == 0.0]

print(tw_is_alag)

print('------------------is_alag == 1.0------------------')
tw_not_is_alag = vec_rel[vec_rel['is_alag'] == 1.0]

print(tw_not_is_alag)

#---------------------------------------------------------------------

tw_alag0 = vec_rel[vec_rel['is_alag'] == 0.0]

tw_alag0 = tw_alag0.sample(n = 2945, random_state=1)

tw_alag0 = tw_alag0[['temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','is_alag']]

print(tw_alag0)

tw_alag1 = vec_rel[vec_rel['is_alag'] == 1.0]

tw_alag1 = tw_alag1.sample(n = 2945, random_state=1)

tw_alag1 = tw_alag1[['temperatura','umidade','temperatura_ponto_orvalho','pressao_atmosferica','precipitacao','is_alag']]

print(tw_alag1)

print('-----------late-fusion-flood-----------')

late_fusion_flood = pd.concat([tw_alag0, tw_alag1])

print(late_fusion_flood)

late_fusion_flood.to_csv(train_test_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_late_fusion_alagamento.csv')

#-------------------------------------------------------------------------------

#-------------MODEL TWEETS-------------

print('------------------tweet related == 0.0------------------')

tw_related= vec_rel[vec_rel['related'] == 0.0]

print(tw_related)

print('------------------tweet related == 1.0------------------')

tw_not_related = vec_rel[vec_rel['related'] == 1.0]

print(tw_not_related)

#-------------------------------------------------------------------------------

tw_rel0 = vec_rel[vec_rel['related'] == 0.0]

tw_rel0 = tw_rel0.sample(n = 2945, random_state=1)

tw_rel0 = tw_rel0[['id_str','created_at','text','related']]

print(tw_rel0)

tw_rel1 = vec_rel[vec_rel['related'] == 1.0]

tw_rel1 = tw_rel1.sample(n = 2945, random_state=1)

tw_rel1 = tw_rel1[['id_str','created_at','text','related']]

print(tw_rel1)

print('-----------late_fusion_tweet-----------')

late_fusion_tweet = pd.concat([tw_rel0, tw_rel1])

print(late_fusion_tweet)

late_fusion_tweet.to_csv(train_test_location + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_late_fusion_tweet.csv')
