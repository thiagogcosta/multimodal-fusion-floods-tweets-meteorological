
import pandas as pd
from ml_late_fusion import ML
import pickle
import time

#-------------------PROCESSING DATAFRAME-------------------

def processing_dataframe(vec_data_50, vec_data_100):
    
    new_vec_data_50 = []
    
    new_vec_data_100 = []

    for i in vec_data_50:
        df = pd.read_csv(local_late_fusion_model + i +'.csv')
    
        df = df.rename(columns={'50': 'target'})
        
        df.to_csv(local_late_fusion_model + i + 'OK.csv')
    
    for i in vec_data_100:
        df = pd.read_csv(local_late_fusion_model + i +'.csv')
        
        df = df.rename(columns={'100': 'target'})
        
        df.to_csv(local_late_fusion_model + i + 'OK.csv')
    

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

# late_fusion folder location
local_late_fusion_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/late_fusion/data/'

# =============================================================================      

start = time.time()

vec_data_50 = ['socialflood_modelTraining_FastText_cbow50_tweets_late_fusion', 'socialflood_modelTraining_FastText_skip50_tweets_late_fusion',
                'socialflood_modelTraining_Word2Vec_cbow50_tweets_late_fusion','socialflood_modelTraining_Word2Vec_skip50_tweets_late_fusion']

vec_data_100 = ['socialflood_modelTraining_FastText_cbow100_tweets_late_fusion', 'socialflood_modelTraining_FastText_skip100_tweets_late_fusion',
                'socialflood_modelTraining_Word2Vec_cbow100_tweets_late_fusion','socialflood_modelTraining_Word2Vec_skip100_tweets_late_fusion',
                'socialflood_modelTraining_BOW_tweets_late_fusion', 'socialflood_modelTraining_TF-IDF_tweets_late_fusion']


#-------------------PROCESSING DATAFRAME-------------------

#vector_dataframes = processing_dataframe(vec_data_50, vec_data_100)



#-------------------MODELS TYPES-------------------


models_types = ['socialflood_modelTraining_FastText_cbow50_tweets_late_fusionOK.csv', 
                 'socialflood_modelTraining_FastText_skip50_tweets_late_fusionOK.csv', 
                 'socialflood_modelTraining_Word2Vec_cbow50_tweets_late_fusionOK.csv',
                 'socialflood_modelTraining_Word2Vec_skip50_tweets_late_fusionOK.csv',
                 'socialflood_modelTraining_FastText_cbow100_tweets_late_fusionOK.csv', 
                 'socialflood_modelTraining_FastText_skip100_tweets_late_fusionOK.csv', 
                 'socialflood_modelTraining_FastText_cbow100_tweets_late_fusionOK.csv',
                 'socialflood_modelTraining_Word2Vec_skip100_tweets_late_fusionOK.csv',
                 'socialflood_modelTraining_BOW_tweets_late_fusionOK.csv',
                 'socialflood_modelTraining_TF-IDF_tweets_late_fusionOK.csv'
                 ]

#----------------------------SALVE THE BEST MODEL - SKIP GRAM 100 DIMENSIONS----------------------------
 
#models_types = ['socialflood_modelTraining_FastText_skip100_tweets_late_fusionOK.csv'] 

#--------------------------------------------------------

df_results = []

df_results_metrics = []

for types in models_types:

    #***********Processing***********
    
    data = pd.read_csv(local_late_fusion_model + types)
    
    data_not_related = data[data['target'] == 0]

    data_related = data[data['target'] == 1]

    data_not_related = data_not_related.sample(n = 2846) 

    data_related = data_related.sample(n = 2846) 

    data = pd.concat([data_related, data_not_related])

    data = data.sample(n = 5692) 

    data = data.drop('Unnamed: 0', 1)

    data = data.drop('Unnamed: 0.1', 1)
    
    data.reset_index(drop=True, inplace=True)
    
    columns = ['algorithm','accuraccy', 'precision', 'recall', 'f1-score']
    df = pd.DataFrame(columns = columns)

    data = data[0:5692]

    print(data)
    
    aux_ml = ML(0.10, 'tweet')
    
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
    
    data_train = data.sample(n = 5692)

    random_forest = aux_ml.RF_classifier(data)

    data_train = data_train.sample(n = 5692)

    naive_bayes = aux_ml.NB_classifier(data_train)

    data_train = data_train.sample(n = 5692)

    decision_tree = aux_ml.DT_classifier(data_train)

    data_train = data_train.sample(n = 5692)

    logistic_regression = aux_ml.RL_classifier(data_train)

    data_train = data.sample(n = 5692)

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
    predictions_rf = random_forest[0][0]
    predictions_nb = naive_bayes[0][0]
    predictions_dt = decision_tree[0][0]
    predictions_lr = logistic_regression[0][0]
    predictions_svm = svm_class[0][0]

    #-----------target-----------
    target_rf = random_forest[0][1]
    target_nb = naive_bayes[0][1]
    target_dt = decision_tree[0][1]
    target_lr = logistic_regression[0][1]
    target_svm = svm_class[0][1]
    
    #-----------acuraccy-----------
    acuraccy_rf = random_forest[0][2]
    acuraccy_nb = naive_bayes[0][2]
    acuraccy_dt = decision_tree[0][2]
    acuraccy_lr = logistic_regression[0][2]
    acuraccy_svm = svm_class[0][2]

    #-----------precision-----------
    precision_rf = random_forest[0][3]
    precision_nb = naive_bayes[0][3]
    precision_dt = decision_tree[0][3]
    precision_lr = logistic_regression[0][3]
    precision_svm = svm_class[0][3]

    #-----------recall-----------
    recall_rf = random_forest[0][4]
    recall_nb = naive_bayes[0][4]
    recall_dt = decision_tree[0][4]
    recall_lr = logistic_regression[0][4]
    recall_svm = svm_class[0][4]

    #-----------f1score-----------
    f1score_rf = random_forest[0][5]
    f1score_nb = naive_bayes[0][5]
    f1score_dt = decision_tree[0][5]
    f1score_lr = logistic_regression[0][5]
    f1score_svm = svm_class[0][5]

    #-----------error-----------
    error_rf = random_forest[0][6]
    error_nb = naive_bayes[0][6]
    error_dt = decision_tree[0][6]
    error_lr = logistic_regression[0][6]
    error_svm = svm_class[0][6]

    #-----------acuracia_modelo-----------
    model_accuraccy_rf = random_forest[0][8]
    model_accuraccy_nb = naive_bayes[0][8]
    model_accuraccy_dt = decision_tree[0][8]
    model_accuraccy_lr = logistic_regression[0][8]
    model_accuraccy_svm = svm_class[0][8]

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

    #----------------------------SALVE THE BEST MODEL - RANDOM FOREST---------------------------
    
    #filename = local_late_fusion_model+'rf_train_modelo_late_fusion_tweets.sav'
    #pickle.dump(random_forest[0][7], open(filename, 'wb'))
    
    #filename = local_late_fusion_model+'modelo_late_fusion_tweets_scaler.sav'
    #pickle.dump(random_forest[1], open(filename, 'wb'))
    
    #print(df_result)

    #print(df_metrics)

    #df_result.to_csv(local_late_fusion_model + 'target_resultado_classificadores_modelo_late_fusion_tweets_FastText_skip100.csv')

    #df_metrics.to_csv(local_late_fusion_model + 'resultado_classificadores_modelo_late_fusion_tweets_FastText_skip100.csv')
    
    df_results_metrics.append(df_metrics)
 
df_results_metrics = pd.concat([df_results_metrics[i] for i in range(len(df_results_metrics))])

df_results_metrics.reset_index(drop=True, inplace=True)

print(df_results_metrics)

df_results_metrics.to_csv(local_late_fusion_model + 'resultado_classificadores_modelo_late_fusion_tweets.csv')

#-----------------PREPROCESSING RESULTS-----------------

#data_results = pd.read_csv(local_late_fusion_model + 'resultado_classificadores_early_fusion.csv')

#data_results_ok = preprocessing_results(data_results)

#data_results_ok.to_csv(local_late_fusion_model + 'resultado_classificadores_tweets_2016_2017_describe_CERTO.csv')

# =============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================
