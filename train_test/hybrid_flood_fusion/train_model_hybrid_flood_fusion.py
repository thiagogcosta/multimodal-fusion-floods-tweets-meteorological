
import pandas as pd
from ml_hybrid_flood_fusion import ML
import pickle
import time
import joblib

#-------------------PROCESSING DATAFRAME-------------------

def processing_dataframe(vec_data_50, vec_data_100):
    
    new_vec_data_50 = []
    
    new_vec_data_100 = []

    for i in vec_data_50:
        df = pd.read_csv(local_hybrid_flood_fusion_model + i +'.csv')
    
        df = df.rename(columns={'50': 'target', '51': 'index'})
        new_vec_data_50.append(df)

    for i in vec_data_100:
        df = pd.read_csv(local_hybrid_flood_fusion_model + i +'.csv')
        
        df = df.rename(columns={'100': 'target', '101': 'index'})
        new_vec_data_100.append(df)
    
    return [new_vec_data_50, new_vec_data_100]

#-------------------PROCESSING DATAFRAME EARLY FUSION 50 DIMENSIONS-------------------

def processing_df_hybrid_flood_fusion_50(vec_50_df, hybrid_flood_fusion, vec_data_50):
    
    count = 0

    for model in vec_50_df:
        
        print(vec_data_50[count])
        
        for i in range(len(model)):
            
            #print(i, vec_50_df[count])
            print(i)
            
            for j in range(len(hybrid_flood_fusion)):
                
                #print(j)
                
                if model.loc[i]['index'] == hybrid_flood_fusion.loc[j]['index']:
                    
                    model.loc[i, 'temperatura'] = hybrid_flood_fusion.loc[j]['temperatura']
                    model.loc[i, 'temperatura_ponto_orvalho'] = hybrid_flood_fusion.loc[j]['temperatura_ponto_orvalho']
                    model.loc[i, 'pressao_atmosferica'] = hybrid_flood_fusion.loc[j]['pressao_atmosferica']
                    model.loc[i, 'umidade'] = hybrid_flood_fusion.loc[j]['umidade']
                    model.loc[i, 'precipitacao'] = hybrid_flood_fusion.loc[j]['precipitacao']
                    model.loc[i, 'related'] = hybrid_flood_fusion.loc[j]['related']
                    model.loc[i, 'inside_cluster'] = hybrid_flood_fusion.loc[j]['inside_cluster']
                    model.loc[i, 'is_alag'] = hybrid_flood_fusion.loc[j]['is_alag']

                    if hybrid_flood_fusion.loc[j]['related'] == 1.0 and hybrid_flood_fusion.loc[j]['inside_cluster'] == 1.0 and hybrid_flood_fusion.loc[j]['is_alag'] == 1.0:
                        model.loc[i, 'target'] = 1
                    else:
                        model.loc[i, 'target'] = 0
                    
                    #print(model)
                    
                    break   
        
        print(model)
        
        model.to_csv(local_hybrid_flood_fusion_model + vec_data_50[count]+ '_CERTO.csv')
        
        count += 1
        
        print(model)

#-------------------PROCESSING DATAFRAME EARLY FUSION 100 DIMENSIONS-------------------

def processing_df_hybrid_flood_fusion_100(vec_100_df, hybrid_flood_fusion, vec_data_100):

    count = 0

    for model in vec_100_df:
        
        print(vec_data_100[count])
        
        for i in range(len(model)):
            
            #print(i, vec_100_df[count])
            print(i)
            
            for j in range(len(hybrid_flood_fusion)):
                
                if model.loc[i]['index'] == hybrid_flood_fusion.loc[j]['index']:
                    
                    model.loc[i, 'temperatura'] = hybrid_flood_fusion.loc[j]['temperatura']
                    model.loc[i, 'temperatura_ponto_orvalho'] = hybrid_flood_fusion.loc[j]['temperatura_ponto_orvalho']
                    model.loc[i, 'pressao_atmosferica'] = hybrid_flood_fusion.loc[j]['pressao_atmosferica']
                    model.loc[i, 'umidade'] = hybrid_flood_fusion.loc[j]['umidade']
                    model.loc[i, 'precipitacao'] = hybrid_flood_fusion.loc[j]['precipitacao']
                    model.loc[i, 'related'] = hybrid_flood_fusion.loc[j]['related']
                    model.loc[i, 'inside_cluster'] = hybrid_flood_fusion.loc[j]['inside_cluster']
                    model.loc[i, 'is_alag'] = hybrid_flood_fusion.loc[j]['is_alag']

                    if hybrid_flood_fusion.loc[j]['related'] == 1.0 and hybrid_flood_fusion.loc[j]['inside_cluster'] == 1.0 and hybrid_flood_fusion.loc[j]['is_alag'] == 1.0:
                        model.loc[i, 'target'] = 1
                    else:
                        model.loc[i, 'target'] = 0
                    
                    break   
        
        print(model)
        
        model.to_csv(local_hybrid_flood_fusion_model + vec_data_100[count]+ '_CERTO.csv')
        
        count += 1
        
        print(model)

#-------------------PREPROCESSING RESULTS-------------------
def preprocessing_results(data):

    for i in range(len(data)):

        data.loc[i, 'accuracy'] =  str(data.loc[i]['accuracy']).replace('.',',')
        data.loc[i, 'precision'] =  str(data.loc[i]['precision']).replace('.',',')
        data.loc[i, 'recall'] =  str(data.loc[i]['recall']).replace('.',',')
        data.loc[i, 'f1-score'] =  str(data.loc[i]['f1-score']).replace('.',',')
        data.loc[i, 'error'] =  str(data.loc[i]['error']).replace('.',',')
        data.loc[i, 'model_accuraccy'] =  str(data.loc[i]['model_accuraccy']).replace('.',',')
        data.loc[i, 'archive'] = str(data.loc[i]['archive']).replace('.csv','')
        data.loc[i, 'archive'] = str(data.loc[i]['archive']).replace('enoesocial_modelTraining_','')
        data.loc[i, 'algorithm'] = data.loc[i]['algorithm'] + ' - ' +data.loc[i]['archive']


    data = data.drop('Unnamed: 0', 1)
    #data = data.drop('algorithm', 1)
    #data = data.drop('archive', 1)

    return data

#-----------------------------DIRECTORIES-----------------------------

# train_test folder location
local_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

# early_fusion folder location
local_hybrid_flood_fusion_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/hybrid_flood_fusion/data/'

# =============================================================================      

start = time.time()

early_fusion_data = local_model + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_early_fusion.csv'

early_fusion_data  = pd.read_csv(early_fusion_data)

vec_data_50 = ['socialflood_modelTraining_FastText_cbow50_tweets_early_fusion', 'socialflood_modelTraining_FastText_skip50_tweets_early_fusion',
                'socialflood_modelTraining_Word2Vec_cbow50_tweets_early_fusion','socialflood_modelTraining_Word2Vec_skip50_tweets_early_fusion']

vec_data_100 = ['socialflood_modelTraining_FastText_cbow100_tweets_early_fusion', 'socialflood_modelTraining_FastText_skip100_tweets_early_fusion',
                'socialflood_modelTraining_Word2Vec_cbow100_tweets_early_fusion','socialflood_modelTraining_Word2Vec_skip100_tweets_early_fusion',
                'socialflood_modelTraining_BOW_tweets_early_fusion', 'socialflood_modelTraining_TF-IDF_tweets_early_fusion']


#-------------------PROCESSING DATAFRAME-------------------

vector_dataframes = processing_dataframe(vec_data_50, vec_data_100)

#-------------------PROCESSING DATAFRAME EARLY FUSION 50 DIMENSIONS-------------------

processing_df_hybrid_flood_fusion_50(vector_dataframes[0], early_fusion_data, vec_data_50)

#-------------------PROCESSING DATAFRAME EARLY FUSION 100 DIMENSIONS-------------------

processing_df_hybrid_flood_fusion_100(vector_dataframes[1], early_fusion_data, vec_data_100)

#-------------------------------------------------------------------------------------------------------

#-------------------MODELS TYPES-------------------

models_types = [
                'socialflood_modelTraining_FastText_cbow50_tweets_early_fusion_CERTO.csv', 
                'socialflood_modelTraining_FastText_skip50_tweets_early_fusion_CERTO.csv',
                'socialflood_modelTraining_Word2Vec_cbow50_tweets_early_fusion_CERTO.csv',
                'socialflood_modelTraining_Word2Vec_skip50_tweets_early_fusion_CERTO.csv',
                'socialflood_modelTraining_FastText_cbow100_tweets_early_fusion_CERTO.csv', 
                'socialflood_modelTraining_FastText_skip100_tweets_early_fusion_CERTO.csv', 
                'socialflood_modelTraining_Word2Vec_cbow100_tweets_early_fusion_CERTO.csv',
                'socialflood_modelTraining_Word2Vec_skip100_tweets_early_fusion_CERTO.csv',
                'socialflood_modelTraining_BOW_tweets_early_fusion_CERTO.csv', 
                'socialflood_modelTraining_TF-IDF_tweets_early_fusion_CERTO.csv'
                 ]

#----------------------------SALVE THE BEST MODEL - DECISION TREE----------------------------

#models_types = ['socialflood_modelTraining_Word2Vec_skip100_tweets_early_fusion_CERTO.csv']  

df_results = []

df_results_metrics = []

for types in models_types:

    #***********Processing***********
    
    data = pd.read_csv(local_hybrid_flood_fusion_model + types)
    
    data = data.drop('Unnamed: 0', 1)
    
    data = data.drop('Unnamed: 0.1', 1)
    
    data = data.drop('index', 1)
    
    data = data.drop('related', 1)
    
    data = data.drop('is_alag', 1)

    data_alag = data[['temperatura', 'umidade', 'temperatura_ponto_orvalho', 'pressao_atmosferica']]

    #-----------------------------PREDICT TWEETS---------------------------------
    alag_model = joblib.load(local_hybrid_flood_fusion_model + 'dt_train_modelo_late_fusion_alagamento.sav')
    
    #------Feature scaling------
    
    scaler = joblib.load(local_hybrid_flood_fusion_model + 'modelo_late_fusion_alagamento_scaler.sav')
    
    data_alag = scaler.fit_transform(data_alag)

    result_alag = alag_model.predict(data_alag)
    
    for j in range(len(result_alag)):

        data.loc[j, 'predict_alag'] = result_alag[j]
        #print(j)

    data.insert(len(data.columns) -1,'target', data.pop("target"))
    
    data = data.drop('temperatura', 1)
    
    data = data.drop('temperatura_ponto_orvalho', 1)
    
    data = data.drop('pressao_atmosferica', 1)
    
    data = data.drop('precipitacao', 1)
    
    data = data.drop('umidade', 1)
    
    #-------------------------------------MODEL HYBRID FUSION-------------------------------------

    data_not_related = data[data['target'] == 0]
    
    data_related = data[data['target'] == 1]

    data_not_related = data_not_related.sample(n = 449) 

    data_related = data_related.sample(n = 449) 

    data = pd.concat([data_related, data_not_related])

    data = data.sample(n = 898) 

    data.reset_index(drop=True, inplace=True)
    
    columns = ['algorithm','accuraccy', 'precision', 'recall', 'f1-score']
    df = pd.DataFrame(columns = columns)

    data = data[0:898]

    aux_ml = ML(0.10)

    #*********************Best Params*********************
    #-----------SVM-----------
    #svm = aux.SVM_bestParams(data)
    #print(svm)
    #{'C': 0.01, 'decision_function_shape': 'ovo', 'gamma': 0.001, 'kernel': 'linear', 'shrinking': True}
    #****************************************************************************
    #-----------Random Forest-----------
    #rf = aux.RF_bestParams(data)
    #------Grid Search------
    #{'bootstrap': True, 'max_depth': 90, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 100}
    #****************************************************************************
    #-----------Decision Tree-----------
    #dt_best = aux.DT_bestParams(data)
    #------DT Search------
    #criterion = gini
    #max_depth = 6
    #****************************************************************************
    #-----------Logistic Regression-----------
    #dt_best = aux.RL_bestParams(data)
    #------LR Search------
    #{'C': 0.09, 'multi_class': 'ovr', 'penalty': 'l1', 'solver': 'liblinear'}
    #****************************************************************************
    #----------GaussianNB------------
    #nb_best = aux.NB_bestParams(data)
    #------NB Search-----
    #'var_smoothing': 0.000000432876
    #****************************************************************************

    #***********Classifier***********
    
    data_train = data.sample(n = 898)

    random_forest = aux_ml.RF_classifier(data)

    data_train = data_train.sample(n = 898)

    naive_bayes = aux_ml.NB_classifier(data_train)

    data_train = data_train.sample(n = 898)

    decision_tree = aux_ml.DT_classifier(data_train)

    data_train = data_train.sample(n = 898)

    logistic_regression = aux_ml.RL_classifier(data_train)

    data_train = data.sample(n = 898)

    svm_class = aux_ml.SVM_classifier(data_train)

    #***********Execution***********
    
    columns = ['y_rf', 'y_nb', 'y_dt', 'y_lr', 'y_svm']

    df_result = pd.DataFrame(columns=columns)

    features = data['target']

    features = features.reset_index()

    columns = ['algorithm', 'accuraccy', 'precision', 'recall', 'f1score', 'error', 'model_accuraccy']

    df_metrics = pd.DataFrame(columns=columns)

    #---------------------------------------------------------------------------------------------

    #-----------prediction-----------
    predictions_rf = random_forest[0]
    predictions_nb = naive_bayes[0]
    predictions_dt = decision_tree[0]
    predictions_lr = logistic_regression[0]
    predictions_svm = svm_class[0]

    #-----------target-----------
    target_rf = random_forest[1]
    target_nb = naive_bayes[1]
    target_dt = decision_tree[1]
    target_lr = logistic_regression[1]
    target_svm = svm_class[1]
    
    #-----------acuraccy-----------
    acuraccy_rf = random_forest[2]
    acuraccy_nb = naive_bayes[2]
    acuraccy_dt = decision_tree[2]
    acuraccy_lr = logistic_regression[2]
    acuraccy_svm = svm_class[2]

    #-----------precision-----------
    precision_rf = random_forest[3]
    precision_nb = naive_bayes[3]
    precision_dt = decision_tree[3]
    precision_lr = logistic_regression[3]
    precision_svm = svm_class[3]

    #-----------recall-----------
    recall_rf = random_forest[4]
    recall_nb = naive_bayes[4]
    recall_dt = decision_tree[4]
    recall_lr = logistic_regression[4]
    recall_svm = svm_class[4]

    #-----------f1score-----------
    f1score_rf = random_forest[5]
    f1score_nb = naive_bayes[5]
    f1score_dt = decision_tree[5]
    f1score_lr = logistic_regression[5]
    f1score_svm = svm_class[5]

    #-----------error-----------
    error_rf = random_forest[6]
    error_nb = naive_bayes[6]
    error_dt = decision_tree[6]
    error_lr = logistic_regression[6]
    error_svm = svm_class[6]

    #-----------acuracia_modelo-----------
    model_accuraccy_rf = random_forest[8]
    model_accuraccy_nb = naive_bayes[8]
    model_accuraccy_dt = decision_tree[8]
    model_accuraccy_lr = logistic_regression[8]
    model_accuraccy_svm = svm_class[8]

    classifier = ['rf', 'nb', 'dt', 'lr','svm']

    vector_acuraccy = [acuraccy_rf, acuraccy_nb, acuraccy_dt, acuraccy_lr,acuraccy_svm]
    vector_precision = [precision_rf, precision_nb, precision_dt, precision_lr,precision_svm]
    vector_recall = [recall_rf, recall_nb, recall_dt, recall_lr,recall_svm]
    vector_f1score = [f1score_rf, f1score_nb, f1score_dt, f1score_lr,f1score_svm]
    vector_error = [error_rf, error_nb, error_dt, error_lr,error_svm]
    vector_model_accuraccy = [model_accuraccy_rf, model_accuraccy_nb, model_accuraccy_dt, model_accuraccy_lr, model_accuraccy_svm]

    for i in range(len(classifier)):
        
        size = len(df_metrics)

        df_metrics.loc[size, 'archive'] = types
        df_metrics.loc[size, 'algorithm'] = classifier[i]
        df_metrics.loc[size, 'accuraccy'] = vector_acuraccy[i]
        df_metrics.loc[size, 'precision'] = vector_precision[i]
        df_metrics.loc[size, 'recall'] = vector_recall[i]
        df_metrics.loc[size, 'f1score'] = vector_f1score[i]
        df_metrics.loc[size, 'error'] =  vector_error[i]
        df_metrics.loc[size, 'model_accuraccy'] =  vector_model_accuraccy[i]

    y_pred_rf = [] 
    y_pred_nb = []
    y_pred_dt = []
    y_pred_lr = []
    y_pred_svm = []

    for j in range(10):

        y_pred_rf += predictions_rf[j].tolist()
        y_pred_nb += predictions_nb[j].tolist()
        y_pred_dt += predictions_dt[j].tolist()
        y_pred_lr += predictions_lr[j].tolist()
        y_pred_svm += predictions_svm[j].tolist()

    df_result['y_rf'] = y_pred_rf
    df_result['y_nb'] = y_pred_nb
    df_result['y_dt'] = y_pred_dt
    df_result['y_lr'] = y_pred_lr
    df_result['y_svm'] = y_pred_svm

    #----------------------------SALVE THE BEST MODEL - DECISION TREE----------------------------
    
    #filename = local_hybrid_flood_fusion_model+'dt_train_model_hybrid_flood_fusion.sav'
    #pickle.dump(decision_tree[7], open(filename, 'wb'))
    
    #print(df_result)

    #print(df_metrics)

    #df_result.to_csv(local_hybrid_flood_fusion_model + 'resultado_target_classificadores_flood_fusion.csv')

    #df_metrics.to_csv(local_hybrid_flood_fusion_model + 'resultado_classificadores_flood_fusion.csv')
    
    df_results_metrics.append(df_metrics)

df_results_metrics = pd.concat([df_results_metrics[i] for i in range(len(df_results_metrics))])

df_results_metrics.reset_index(drop=True, inplace=True)

print(df_results_metrics)

df_results_metrics.to_csv(local_hybrid_flood_fusion_model + 'resultado_classificadores_hybrid_flood_fusion_fusion.csv')

#-----------------PREPROCESSING RESULTS-----------------

#data_results = pd.read_csv(local_hybrid_flood_fusion_model + 'resultado_classificadores_early_fusion.csv')

#data_results_ok = preprocessing_results(data_results)

#data_results_ok.to_csv(local_hybrid_flood_fusion_model + 'resultado_classificadores_tweets_2016_2017_describe_CERTO.csv')

# =============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================
