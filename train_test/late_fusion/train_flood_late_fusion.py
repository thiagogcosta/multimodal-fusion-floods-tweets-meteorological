import pandas as pd
from ml_late_fusion import ML
import pickle
import joblib
import time

#-----------------------------DIRECTORIES-----------------------------

# train_test folder location
local_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/data/'

# late_fusion folder location
local_late_fusion_model = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/train_test/late_fusion/data/'

# =============================================================================      

start = time.time()

flood_data = local_model + 'THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_late_fusion_alagamento.csv'

#***********Processing***********

data = pd.read_csv(flood_data)

data_not_related = data[data['is_alag'] == 0]

print(data_not_related)

data_related = data[data['is_alag'] == 1]

print(data_related)

data_not_related = data_not_related.sample(n = 2945) 

data_related = data_related.sample(n = 2945) 

data = pd.concat([data_related, data_not_related])

data = data.sample(n = 5890) 

data = data.drop('Unnamed: 0', 1)

data = data.drop('precipitacao', 1)

data.reset_index(drop=True, inplace=True)

colunas = ['algorithm','accuraccy', 'precision', 'recall', 'f1-score']
df = pd.DataFrame(columns = colunas)

data = data[0:5890]

print(data.columns)

aux_ml = ML(0.10, 'flood')

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

data_train = data.sample(n = 5890)

random_forest = aux_ml.RF_classifier(data)

data_train = data_train.sample(n = 5890)

naive_bayes = aux_ml.NB_classifier(data_train)

data_train = data_train.sample(n = 5890)

decision_tree = aux_ml.DT_classifier(data_train)

data_train = data_train.sample(n = 5890)

logistic_regression = aux_ml.RL_classifier(data_train)

data_train = data.sample(n = 5890)

svm_class = aux_ml.SVM_classifier(data_train)

#***********Execution***********

columns = ['y_rf', 'y_nb', 'y_dt', 'y_lr', 'y_svm']

df_result = pd.DataFrame(columns=columns)

features = data['is_alag']

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

filename = local_late_fusion_model+'dt_train_modelo_late_fusion_alagamento.sav'
pickle.dump(decision_tree[0][7], open(filename, 'wb'))

filename = local_late_fusion_model+'modelo_late_fusion_alagamento_scaler.sav'
pickle.dump(decision_tree[1], open(filename, 'wb'))

print(df_result)

print(df_metrics)

df_result.to_csv(local_late_fusion_model + 'target_classificadores_modelo_late_fusion_alagamento.csv')

df_metrics.to_csv(local_late_fusion_model + 'resultado_classificadores_modelo_late_fusion_alagamento.csv')

# =============================================================================

end = time.time()
 
print('Detailed execution time: ', (end - start))

print('Runtime: ', (end - start)/60)

#=============================================================================
