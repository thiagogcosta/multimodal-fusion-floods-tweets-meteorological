from nltk import agreement
import pandas as pd

# coefficient-of-agreement folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

data_thiago = local + 'THIAGO_NOV2016_NOV2018_KEY.csv' # first judge
data_luan = local + 'LUAN_NOV2016_NOV2018_KEY.csv'     # second judge
data_aysla = local + 'AYSLA_NOV2016_NOV2018_KEY.csv'   # third judge

df_thiago = pd.read_csv(data_thiago) 
df_luan = pd.read_csv(data_luan)    
df_aysla = pd.read_csv(data_aysla)

thiago_judge = df_thiago['related_thiago'].values
luan_judge = df_luan['related_luan'].values
aysla_judge = df_aysla['related_aysla'].values

output_results=[[0,str(aux),str(thiago_judge[aux])] for aux in range(0,len(thiago_judge))]+[[1,str(aux),str(luan_judge[aux])] for aux in range(0,len(luan_judge))]+[[2,str(aux),str(aysla_judge[aux])] for aux in range(0,len(aysla_judge))]

rating_output = agreement.AnnotationTask(data=output_results)

print("Result of Krippendorff's Alpha coefficient:" +str(rating_output.alpha()))
