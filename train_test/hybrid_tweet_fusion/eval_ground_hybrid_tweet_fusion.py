import pandas as pd
import pickle
import time
from con_mongodb import Connection_Mongo
from preprocessing_tweets_hybrid_tweet_fusion import Preprocessing
from gensim.models import KeyedVectors
from embeddings import Embeddings
import heapq
import numpy as np
from bson.binary import Binary
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import Normalizer  
import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

#----------------------PRÉ-PROCESSAMENTO - BOW / TF-IDF------------------------

def data_transformation_fasttext_100(ground_truth, key_words, dict_hashtag, con_mongo, my_database):
    
    name_type = ['FastText']

    for k in name_type:

        model_skip100 = KeyedVectors.load_word2vec_format(local+'embeddings/'+k+'/skip_s100.txt')

        result_skip100 = Preprocessing(ground_truth,key_words, model_skip100, False,dict_hashtag)

        print('---------------------result_skip100---------------------')
        print(result_skip100)

        sentences_skip100 = result_skip100.matriz_tokens
        ids_skip100 = result_skip100.vec_ids
        index_skip100 = result_skip100.vec_index
        labels_skip100 = result_skip100.vec_labels

        #------------------------------------------------------------------------------------------------------------

        emb_skip100 = Embeddings(sentences_skip100, model_skip100)
        res_emb_skip100 = emb_skip100.getWordVec()

        ids = [ids_skip100]
        index = [index_skip100]
        sentences = [sentences_skip100]
        embeddings = [res_emb_skip100]
        labels = [labels_skip100]

        #------------Dataframe embeddings---------
        columns = ['id_str', 'index', 'sentences', 'embeddings','labels']

        vec_name = [k+'_skip100']

        for i in range(len(ids)):

            con_mongo.clear_collection(my_database, 'model_tweets_final_'+vec_name[i])

            con_mongo.create_collection(my_database, 'model_tweets_final_'+vec_name[i])

            for j in range(len(ids[i])):
                
                data = Binary(pickle.dumps(embeddings[i][j], protocol=2), subtype=128)

                vec_data = [ids[i][j], index[i][j], sentences[i][j], data, labels[i][j]]
                con_mongo.insert_one_collection(my_database, 'model_tweets_final_'+vec_name[i], columns, vec_data)
                print(j)
                
            print(vec_name[i])
        
def create_model_training(con_mongo, my_database):
    
    name_type = ['FastText']
    
    for l in name_type:

        vec_name = [l+'_skip100']
        vec_len_emb = [100]

        for k in range(len(vec_name)):

            model_tweets_embed = con_mongo.get_collection(my_database,'model_tweets_final_'+vec_name[k], {}, True)

            print(len(model_tweets_embed))
            
            #--------------------------------------
            vec_matrix = []
            
            for i in range(len(model_tweets_embed)):
                
                print(i, l)

                aux_emb = pickle.loads(model_tweets_embed.loc[i]['embeddings'])
                aux_emb = np.append(aux_emb, model_tweets_embed.loc[i]['labels'])
                aux_emb = np.append(aux_emb, model_tweets_embed.loc[i]['index'])
                
                vec_matrix.append(aux_emb)    
            
            data_model = pd.DataFrame(vec_matrix)

            print(data_model)
            
            data_model.to_csv(local_model_result+'socialflood_model_conj_vdd_'+l+'_tweets_hybrid_tweet_fusion.csv')

#------------------------------------DATAFRAME early fusion 100 DIMENSÕES------------------------------------

def processing_df_hybrid_tweet_fusion_100(model, ground_truth):
    
    for i in range(len(model)):
        
        print(i)
        
        for j in range(len(ground_truth)):
            
            if model.loc[i]['index'] == ground_truth.loc[j]['index']:
                
                model.loc[i, 'temperatura'] = ground_truth.loc[j]['temperatura']
                model.loc[i, 'temperatura_ponto_orvalho'] = ground_truth.loc[j]['temperatura_ponto_orvalho']
                model.loc[i, 'pressao_atmosferica'] = ground_truth.loc[j]['pressao_atmosferica']
                model.loc[i, 'umidade'] = ground_truth.loc[j]['umidade']
                model.loc[i, 'precipitacao'] = ground_truth.loc[j]['precipitacao']
                model.loc[i, 'related'] = ground_truth.loc[j]['related']
                model.loc[i, 'inside_cluster'] =ground_truth.loc[j]['inside_cluster']
                model.loc[i, 'is_alag'] = ground_truth.loc[j]['is_alag']

                if ground_truth.loc[j]['related'] == 1.0 and ground_truth.loc[j]['inside_cluster'] == 1.0 and ground_truth.loc[j]['is_alag'] == 1.0:
                    model.loc[i, 'target'] = 1
                else:
                    model.loc[i, 'target'] = 0
                
                break   
    
    return model

#-----------------------------DIRECTORIES-----------------------------

# textual-features folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/'

# train_test folder location
local_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

# coefficient-of-agreement folder location
local_coefficient = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

# hybrid_tweet_fusion folder location
local_model_result = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/hybrid_tweet_fusion/data/'

# ground truth folder location
model_ground_truth = 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade_OK.csv'


# =============================================================================      

start = time.time()

#---------------------------Connect MONGO DB-----------------------------------

con_mongo = Connection_Mongo('localhost', '27017')
my_database = con_mongo.create_db('enoesocial-tweets_hybrid_tweet_fusion-EVAL-OK-JOURNAL')

#-----------------------------KEY-WORDS----------------------------------------

columns = ['palavras-chave']

key_words = pd.DataFrame(columns = columns)

phenomenal_words = ["alagamento","alagado","alagada","alagando","alagou","alagar",
                    "chove","chova","chovia","chuva","chuvarada","chuvosa","chuvoso",
                    "chuvona","chuvinha","chuvisco","chuvendo",
                    "diluvio", "dilúvio", "enchente", "enxurrada",
                    "garoa","inundação","inundacao","inundada","inundado",
                    "inundar","inundam","inundou","temporal","temporais", "tromba d'água"]

for i in phenomenal_words:
    
    index = len(key_words)
    
    key_words.loc[index, 'palavras-chave'] = i

#print(kw)

#---------------------------------------------------------------

#-------------------------DICT HASHTAG-------------------------

dict_hashtag = pd.read_csv(local_coefficient + 'THIAGO_NOV2016_NOV2018_judge_hashtags_rel.csv')

#print(dict_hashtag)

dict_hashtag = dict_hashtag[dict_hashtag['correcao'] != None]

dict_hashtag = dict_hashtag.dropna()

dict_hashtag.reset_index(drop=True, inplace=True)

#print(dict_hashtag)

#-------------------------------------------------------------------------------

ground_truth = pd.read_csv(local_model + model_ground_truth)

ground_truth = ground_truth[['id_str','index', 'text', 'related',
                             'inside_cluster', 'is_alag', 'horas',
                             'temperatura', 'umidade', 'temperatura_ponto_orvalho',
                             'pressao_atmosferica', 'precipitacao']]

#print(ground_truth)

#----------------------PRÉ-PROCESSAMENTO - BOW / TF-IDF------------------------

data_transformation_fasttext_100(ground_truth, key_words, dict_hashtag, con_mongo, my_database)

#-------------------------CREATE MODEL TRAINING - BOW-------------------------

create_model_training(con_mongo, my_database)
    
#----------------------------PREPARANDO O DATASET----------------------------

model = pd.read_csv(local_model_result+'socialflood_model_conj_vdd_FastText_tweets_hybrid_tweet_fusion.csv')

model = model.rename(columns={'100': 'target', '101': 'index'})

print(model)

#------------------------------------------------MODEL EARLY FUSION------------------------------------------------

ground_truth_ok = processing_df_hybrid_tweet_fusion_100(model, ground_truth)

print(ground_truth_ok)

ground_truth_ok.to_csv(local_model_result + 'socialflood_model_conj_vdd_FastText_tweets_hybrid_tweet_fusion_CERTO.csv')


#***********Processing***********

ground_truth_ok = pd.read_csv(local_model_result + 'socialflood_model_conj_vdd_FastText_tweets_hybrid_tweet_fusion_CERTO.csv')

data = ground_truth_ok.reset_index()

data = data.drop('Unnamed: 0', 1)

data = data.drop('Unnamed: 0.1', 1)

data = data.drop('index', 1)

data = data.drop('precipitacao', 1)

data = data.drop('level_0', 1)

#print(data)

#-----------------------------PREDICT TWEETS---------------------------------
tweet_model = joblib.load(local_model_result + 'rf_train_modelo_late_fusion_tweets.sav')

X_tweets = data.iloc[:, 0: 100]

#print(X_tweets)

#------Feature scaling------
scaler = Normalizer()
X_tweets = scaler.fit_transform(X_tweets)

result_tweets = tweet_model.predict(X_tweets)

print('----------------------metrics tweets----------------------')
metrics = precision_recall_fscore_support(data['related'].values, result_tweets, average='macro')

acc = accuracy_score(data['related'].values, result_tweets)

print('Acuracia: ', acc)
print('Precision: ', metrics[0])
print('Recall: ', metrics[1])
print('F1-score: ', metrics[2])

data = data.drop('related', 1)

#-----------------------------------------------------------------------------
for i in range(len(result_tweets)):

    #print(i)
    data.loc[i, 'predict_tweet'] = result_tweets[i]

hybrid_tweet_model = joblib.load(local_model_result + 'rf_train_model_hybrid_tweet_fusion.sav')

colunas = ['type', 'precision', 'recall', 'f1score']
df_result = pd.DataFrame(columns= colunas)

data_x = data[['temperatura', 'temperatura_ponto_orvalho', 'pressao_atmosferica', 'umidade', 'inside_cluster', 'predict_tweet']]

data = data.drop('is_alag', 1)

#------Feature scaling------
scaler = Normalizer()
X_hybrid = scaler.fit_transform(data_x)

result_hybrid= hybrid_tweet_model.predict(X_hybrid)

#-----------------------------TARGET---------------------------------
target = data[['target']]

print('----------------------metrics hybrid----------------------')
metrics = precision_recall_fscore_support(target.values, result_hybrid,  average='macro')

print('Precision: ', metrics[0])
print('Recall: ', metrics[1])
print('F1-score: ', metrics[2])

df_result.loc[0, 'type'] = 'metrics alag'
df_result.loc[0, 'precision'] = metrics[0]
df_result.loc[0, 'recall'] = metrics[1]
df_result.loc[0, 'f1score'] = metrics[2]

df_result.to_csv(local_model_result + 'resultado_predict_hybrid_tweet_fusion.csv')

df_target = pd.DataFrame(data=result_hybrid.flatten())

df_target.to_csv(local_model_result + 'resultado_target_hybrid_tweet_fusion.csv')

#=============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================
