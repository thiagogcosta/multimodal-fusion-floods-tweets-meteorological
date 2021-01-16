import pandas as pd
import pickle
import time
from con_mongodb import Connection_Mongo
from preprocessing_tweets_unimodal_fusion import Preprocessing
from gensim.models import KeyedVectors
from embeddings import Embeddings
import heapq
import numpy as np
from bson.binary import Binary
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import Normalizer  
import joblib
from sklearn.metrics import precision_recall_fscore_support


#----------------------PRÉ-PROCESSAMENTO - BOW / TF-IDF------------------------

def data_transformation_bow(ground_truth, key_words, dict_hashtag, con_mongo, my_database):
    
    model = KeyedVectors.load_word2vec_format(local+'embeddings/Word2Vec/skip_s50.txt')

    result = Preprocessing(ground_truth, key_words, model, True, dict_hashtag)

    sentences = result.matriz_tokens
    ids = result.vec_ids
    index = result.vec_index
    labels = result.vec_labels

    #-----------------------------BAG OF WORDS (BOW)-------------------------------

    most_freq = pd.read_csv(local + 'most_freq_words_tweets_early_fusion.csv')

    most_freq = most_freq['palavra'].values

    print(most_freq)

    # checking if the word exists in the most frequent
    sentence_vectors = []                                                                                                                                                                                                                                                                                                        
    for token in sentences:
        sent_vec = []
        for word in most_freq:
            if word in token:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    sentence_vectors = np.asarray(sentence_vectors)

    #------------Dataframe BOW AND TF-IDF---------
    columns = ['id_str', 'index', 'sentences', 'matrix','labels']

    con_mongo.clear_collection(my_database, 'model_tweets_final_BOW')

    con_mongo.create_collection(my_database, 'model_tweets_final_BOW')

    for j in range(len(sentence_vectors)):
        
        data = Binary(pickle.dumps(sentence_vectors[j], protocol=2), subtype=128)

        vec_data = [ids[j], index[j], sentences[j], data, labels[j]]
        con_mongo.insert_one_collection(my_database, 'model_tweets_final_BOW', columns, vec_data)
        print(j)
        
def create_model_training(con_mongo, my_database):
    
    name_type = ['BOW']
    
    for l in name_type:

        model_tweets_embed = con_mongo.get_collection(my_database,'model_tweets_final_'+l, {}, True)
        
        #--------------------------------------
        vec_matrix = []
        
        for i in range(len(model_tweets_embed)):
            
            print(i, l)

            aux_emb = pickle.loads(model_tweets_embed.loc[i]['matrix'])
            aux_emb = np.append(aux_emb, model_tweets_embed.loc[i]['labels'])
            aux_emb = np.append(aux_emb, model_tweets_embed.loc[i]['index'])
            vec_matrix.append(aux_emb)    
        
        data_model = pd.DataFrame(vec_matrix)
                
        print(data_model)
        
        return data_model

#------------------------------------DATAFRAME early fusion 100 DIMENSÕES------------------------------------

def processing_df_unimodal_fusion_100(model, ground_truth):
    
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

# unimodal_fusion folder location
local_model_result = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/unimodal_fusion/data/'

# ground truth folder location
model_ground_truth = 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade_OK.csv'


# =============================================================================      

start = time.time()

#---------------------------Connect MONGO DB-----------------------------------

con_mongo = Connection_Mongo('localhost', '27017')
my_database = con_mongo.create_db('enoesocial-tweets_late_unimodal_fusion-EVAL-OK-JOURNAL')

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

data_transformation_bow(ground_truth, key_words, dict_hashtag, con_mongo, my_database)

#-------------------------CREATE MODEL TRAINING - BOW-------------------------

result = create_model_training(con_mongo, my_database)

result.to_csv(local_model_result+'socialflood_model_conj_vdd_BOW_tweets_unimodal_fusion.csv')
        
#----------------------------PREPARANDO O DATASET----------------------------

model = pd.read_csv(local_model_result+'socialflood_model_conj_vdd_BOW_tweets_unimodal_fusion.csv')

model = model.rename(columns={'100': 'target', '101': 'index'})

#print(model)

#------------------------------------------------MODEL EARLY FUSION------------------------------------------------

ground_truth_ok = processing_df_unimodal_fusion_100(model, ground_truth)

print(ground_truth_ok)

ground_truth_ok.to_csv(local_model_result + 'socialflood_model_conj_vdd_BOW_tweets_unimodal_fusion_CERTO.csv')

#***********Processing***********

ground_truth_ok = pd.read_csv(local_model_result + 'socialflood_model_conj_vdd_BOW_tweets_unimodal_fusion_CERTO.csv')

ground_truth_ok = ground_truth_ok.reset_index()

data = ground_truth_ok.drop('Unnamed: 0', 1)

data = data.drop('Unnamed: 0.1', 1)

data = data.drop('level_0', 1)

data = data.drop('index', 1)

data = data.drop('related', 1)

data = data.drop('is_alag', 1)

data = data.drop('precipitacao', 1)

data = data.drop('temperatura', 1)

data = data.drop('temperatura_ponto_orvalho', 1)

data = data.drop('pressao_atmosferica', 1)

data = data.drop('umidade', 1)

data = data.drop('inside_cluster', 1)

data.insert(len(data.columns) -1,'target', data.pop("target"))

data.reset_index(drop=True, inplace=True)

#--------------features and target--------------

columns = ['type', 'precision', 'recall', 'f1score']
df_result = pd.DataFrame(columns= columns)

X = data.iloc[:, 0: len(data.columns)-1]
y = data.iloc[:, len(data.columns)-1:len(data.columns)]

#------Feature scaling------
scaler = Normalizer()
X = scaler.fit_transform(X)

loaded_model = joblib.load(local_model_result + 'dt_train_model_unimodal_fusion.sav')
result = loaded_model.predict(X)

print('-------------------target metrics-------------------')

metrics = precision_recall_fscore_support(y.values, result, average='macro')

print('Precision: ', metrics[0])
print('Recall: ', metrics[1])
print('F1-score: ', metrics[2])

df_result.loc[0, 'type'] = 'metrics alag'
df_result.loc[0, 'precision'] = metrics[0]
df_result.loc[0, 'recall'] = metrics[1]
df_result.loc[0, 'f1score'] = metrics[2]

df_result.to_csv(local_model_result + 'resultado_predict_unimodal_fusion.csv')

df_target = pd.DataFrame(data=result.flatten())

df_target.to_csv(local_model_result + 'resultado_target_unimodal_fusion.csv')

#=============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================
