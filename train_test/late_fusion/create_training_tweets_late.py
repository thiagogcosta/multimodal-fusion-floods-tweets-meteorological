# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 03:12:48 2019

@author: usuario
"""

import time
import pandas as pd
from random import sample 
import numpy as np
import geopandas as gpd
from gensim.models import KeyedVectors
from embeddings import Embeddings
from bson.binary import Binary
import pickle
import heapq
from sklearn.utils import shuffle
from con_mongodb import Connection_Mongo
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn_pandas import DataFrameMapper
from preprocessing_tweets_late_fusion import Preprocessing
import numpy

#------------------PRE-PROCESSING - FastText / Word2Vec---------------------
def preprocess_tweets_fasttext_word2vec(tweets_late_fusion, dict_hashtag, kw, con_mongo, my_database, type_name, option):
    
    for k in type_name:

        model_skip50 = KeyedVectors.load_word2vec_format(local+'embeddings/'+k+'/skip_s50.txt')
        model_skip100 = KeyedVectors.load_word2vec_format(local+'embeddings/'+k+'/skip_s100.txt')
        model_cbow50 = KeyedVectors.load_word2vec_format(local+'embeddings/'+k+'/cbow_s50.txt')
        model_cbow100 = KeyedVectors.load_word2vec_format(local+'embeddings/'+k+'/cbow_s100.txt')

        result_skip50 = Preprocessing(tweets_late_fusion,kw, model_skip50, False, dict_hashtag, option)

        print('---------------------result_skip50---------------------')
        print(result_skip50)

        result_skip100 = Preprocessing(tweets_late_fusion,kw, model_skip100, False,dict_hashtag, option)

        print('---------------------result_skip100---------------------')
        print(result_skip100)

        result_cbow50 = Preprocessing(tweets_late_fusion,kw, model_cbow50, False,dict_hashtag, option)

        print('---------------------result_cbow50---------------------')
        print(result_cbow50)

        result_cbow100 = Preprocessing(tweets_late_fusion,kw, model_cbow100, False, dict_hashtag, option)

        print('---------------------result_cbow100---------------------')
        print(result_cbow100)

        sentences_skip50 = result_skip50.matriz_tokens
        ids_skip50 = result_skip50.vec_ids
        index_skip50 = result_skip50.vec_index
        labels_skip50 = result_skip50.vec_labels

        sentences_skip100 = result_skip100.matriz_tokens
        ids_skip100 = result_skip100.vec_ids
        index_skip100 = result_skip100.vec_index
        labels_skip100 = result_skip100.vec_labels

        sentences_cbow50 = result_cbow50.matriz_tokens
        ids_cbow50 = result_cbow50.vec_ids
        index_cbow50 = result_cbow50.vec_index
        labels_cbow50 = result_cbow50.vec_labels

        sentences_cbow100 = result_cbow100.matriz_tokens
        ids_cbow100 = result_cbow100.vec_ids
        index_cbow100 = result_cbow100.vec_index
        labels_cbow100 = result_cbow100.vec_labels

        #------------------------------------------------------------------------------------------------------------
        
        emb_skip50 = Embeddings(sentences_skip50, model_skip50)
        res_emb_skip50 = emb_skip50.getWordVec()

        emb_skip100 = Embeddings(sentences_skip100, model_skip100)
        res_emb_skip100 = emb_skip100.getWordVec()

        emb_cbow50 = Embeddings(sentences_cbow50, model_cbow50)
        res_emb_cbow50 = emb_cbow50.getWordVec()

        emb_cbow100 = Embeddings(sentences_cbow100, model_cbow100)
        res_emb_cbow100 = emb_cbow100.getWordVec()

        ids = [ids_skip50, ids_skip100, ids_cbow50, ids_cbow100]
        index = [index_skip50, index_skip100, index_cbow50, index_cbow100]
        sentences = [sentences_skip50, sentences_skip100, sentences_cbow50, sentences_cbow100]
        embeddings = [res_emb_skip50, res_emb_skip100, res_emb_cbow50, res_emb_cbow100]
        labels = [labels_skip50, labels_skip100, labels_cbow50, labels_cbow100]

        #------------Dataframe embeddings---------
        columns = ['id_str', 'sentences', 'embeddings','labels']

        vec_name = [k+'_skip50', k+'_skip100', k+'_cbow50', k+'_cbow100']

        for i in range(len(ids)):

            con_mongo.clear_collection(my_database, 'model_tweets_final_'+vec_name[i])

            con_mongo.create_collection(my_database, 'model_tweets_final_'+vec_name[i])

            for j in range(len(ids[i])):
                
                data = Binary(pickle.dumps(embeddings[i][j], protocol=2), subtype=128)

                vec_data = [ids[i][j], sentences[i][j], data, labels[i][j]]
                con_mongo.insert_one_collection(my_database, 'model_tweets_final_'+vec_name[i], columns, vec_data)
                print(j)
                
            print(vec_name[i])

#------------------------------------------------------------------------------ 

#------------------PRE-PROCESSING - BOW / TF-IDF---------------------

def preprocess_tweets_bow_tfidf(tweets_late_fusion, dict_hashtag, con_mongo, my_database, option):
    
    coluns_most_freq = ['palavra']

    dataframe_most_freq = pd.DataFrame(columns = coluns_most_freq)

    model = KeyedVectors.load_word2vec_format(local+'embeddings/Word2Vec/skip_s50.txt')

    result = Preprocessing(tweets_late_fusion,kw, model, True, dict_hashtag, option)

    sentences = result.matriz_tokens
    ids = result.vec_ids
    index = result.vec_index
    labels = result.vec_labels

    #-----------------------------BAG OF WORDS (BOW)-------------------------------

    frequency_word = {}
    # I check if the word exists in the dictionary and note the frequency
    for words in sentences:
        for token in words:
            if token not in frequency_word.keys():
                frequency_word[token] = 1
            else:
                frequency_word[token] += 1

    # I check the most frequent tokens
    most_frequency = heapq.nlargest(100, frequency_word, key=frequency_word.get)

    #----------------------------SAVING THE MOST FREQUENT WORDS----------------------------
    '''
    for i in range(len(most_frequency)):
        dataframe_most_freq.loc[i, 'palavra'] = most_frequency[i]

    print(dataframe_most_freq)

    dataframe_most_freq.to_csv(local + 'most_freq_words_tweets_early_fusion.csv')
    '''
    #--------------------------------------------------------------------------------------------

    # checking if the word exists in the most frequent
    sentence_vectors = []
    for token in sentences:
        sent_vec = []
        for word in most_frequency:
            if word in token:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    sentence_vectors = np.asarray(sentence_vectors)

    #----------------------------------TF-IDF--------------------------------------

    #----------IDF----------
    word_idf_values = {}
    for token in most_frequency:
        tweet_containing_word = 0
        for words in sentences:
            if token in words:
                tweet_containing_word += 1
        word_idf_values[token] = np.log(len(sentences)/(1 + tweet_containing_word))

    #----------TF----------
    word_tf_values = {}
    for token in most_frequency:
        sentence_tf_vector = []
        for words in sentences:
            doc_freq = 0
            for word in words:
                if token == word:
                    doc_freq += 1
            word_tf = doc_freq/len(words)
            sentence_tf_vector.append(word_tf)
        word_tf_values[token] = sentence_tf_vector

    #--------TF-IDF--------
    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_tweets = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_tweets.append(tf_idf_score)
        tfidf_values.append(tfidf_tweets)

    tf_idf_model = np.asarray(tfidf_values)

    # transposition of the matrix
    tf_idf_model = np.transpose(tf_idf_model)

    #------------Dataframe BOW AND TF-IDF---------
    columns = ['id_str', 'sentences', 'matrix','labels']

    con_mongo.clear_collection(my_database, 'model_tweets_final_BOW')

    con_mongo.create_collection(my_database, 'model_tweets_final_BOW')

    for j in range(len(sentence_vectors)):
        
        data = Binary(pickle.dumps(sentence_vectors[j], protocol=2), subtype=128)

        vec_data = [ids[j], sentences[j], data, labels[j]]
        con_mongo.insert_one_collection(my_database, 'model_tweets_final_BOW', columns, vec_data)
        print(j)


    con_mongo.clear_collection(my_database, 'model_tweets_final_TF-IDF')

    con_mongo.create_collection(my_database, 'model_tweets_final_TF-IDF')

    for j in range(len(tf_idf_model)):

        data = Binary(pickle.dumps(tf_idf_model[j], protocol=2), subtype=128)

        vec_data = [ids[j], sentences[j], data, labels[j]]
        con_mongo.insert_one_collection(my_database, 'model_tweets_final_TF-IDF', columns, vec_data)
        print(j)

#------------------------------------------------------------------------------ 

#------------------CREATE MODEL TRAINING TWEETS - FastText and Word2Vec---------------------

def create_training_data_fasttext_word2vec(con_mongo, my_database):
    
    name_type = ['FastText', 'Word2Vec']

    for l in name_type:

        vec_name = [l+'_skip50', l+'_cbow50',l+'_skip100', l+'_cbow100']
        vec_len_emb = [50, 50, 100, 100]

        for k in range(len(vec_name)):

            model_tweets_embed = con_mongo.get_collection(my_database,'model_tweets_final_'+vec_name[k], {}, True)

            print(len(model_tweets_embed))
            
            #--------------------------------------
            vec_matrix = []
            
            for i in range(len(model_tweets_embed)):
                
                print(i, l)

                aux_emb = pickle.loads(model_tweets_embed.loc[i]['embeddings'])
                aux_emb = numpy.append(aux_emb, model_tweets_embed.loc[i]['labels'])
                
                vec_matrix.append(aux_emb)    
            
            data_model = pd.DataFrame(vec_matrix)
                    
            print(data_model)
    
            data_model.to_csv(local_model_result+'socialflood_modelTraining_'+vec_name[k]+'_tweets_late_fusion.csv')
            print('-----------------------------------------------')

#------------------------------------------------------------------------------

#------------------CREATE MODEL TRAINING TWEETS - BOW and TF-IDF---------------------

def create_training_data_bow_tfidf(con_mongo, my_database):
    
    name_type = ['BOW', 'TF-IDF']
    #name_type = ['BOW']
    for l in name_type:

        model_tweets_embed = con_mongo.get_collection(my_database,'model_tweets_final_'+l, {}, True)
        
        #--------------------------------------
        vec_matrix = []
        
        for i in range(len(model_tweets_embed)):
            
            print(i, l)

            aux_emb = pickle.loads(model_tweets_embed.loc[i]['matrix'])
            aux_emb = numpy.append(aux_emb, model_tweets_embed.loc[i]['labels'])
            vec_matrix.append(aux_emb)    
        
        data_model = pd.DataFrame(vec_matrix)
                
        print(data_model)
        
        data_model.to_csv(local_model_result+'socialflood_modelTraining_'+l+'_tweets_late_fusion.csv')
        print('-----------------------------------------------')
    
#-----------------------------DIRECTORIES-----------------------------

# textual-features folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/'

# train_test folder location
local_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

# coefficient-of-agreement folder location
local_coefficient = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

# late_fusion folder location
local_model_result = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/late_fusion/data/'

# =============================================================================      

start = time.time()

#---------------------------Connect MONGO DB-----------------------------------

con_mongo = Connection_Mongo('localhost', '27017')
my_database = con_mongo.create_db('enoesocial-tweets_late_fusion-OK-JOURNAL')

#-----------------------------KEY-WORDS----------------------------------------

columns = ['palavras-chave']

kw = pd.DataFrame(columns = columns)

words_phenomenon = ["alagamento","alagado","alagada",
                    "alagando","alagou","alagar",
                    "chove","chova","chovia","chuva",
                    "chuvarada","chuvosa","chuvoso",
                    "chuvona","chuvinha","chuvisco","chuvendo",
                    "diluvio", "dilúvio", "enchente", "enxurrada",
                    "garoa","inundação","inundacao","inundada"
                    "inundado","inundar","inundam","inundou",
                    "temporal","temporais","tromba d'água"]

for i in words_phenomenon:
    
    index = len(kw)
    
    kw.loc[index, 'palavras-chave'] = i

#print(kw)
#------------------------------------------------------------------------------

tweets_late_fusion = pd.read_csv(local_model + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_late_fusion_tweet.csv' , sep=(','))

print(tweets_late_fusion)

dict_hashtag = pd.read_csv(local_coefficient + 'THIAGO_NOV2016_NOV2018_judge_hashtags_rel.csv')

dict_hashtag = dict_hashtag[dict_hashtag['correcao'] != None]

dict_hashtag = dict_hashtag.dropna()

dict_hashtag.reset_index(drop=True, inplace=True)

print(dict_hashtag)

#------------------PRE-PROCESSING - FastText / Word2Vec---------------------

type_name = ['FastText']

preprocess_tweets_fasttext_word2vec(tweets_late_fusion, dict_hashtag, kw, con_mongo, my_database, type_name, 'training')

exit(1)

type_name = ['Word2Vec']

preprocess_tweets_fasttext_word2vec(tweets_late_fusion, dict_hashtag, kw, con_mongo, my_database, type_name, 'training')

#------------------PRE-PROCESSING - BOW / TF-IDF---------------------
preprocess_tweets_bow_tfidf(tweets_late_fusion, dict_hashtag, con_mongo, my_database, 'training')

#------------------CREATE MODEL TRAINING DATA - FastText and Word2Vec---------------------

create_training_data_fasttext_word2vec(con_mongo, my_database)

#------------------CREATE MODEL TRAINING DATA - BOW and TF-IDF-------------------------

create_training_data_bow_tfidf(con_mongo, my_database)

#=============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================