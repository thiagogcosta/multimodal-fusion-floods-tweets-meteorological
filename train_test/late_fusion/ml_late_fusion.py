import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class ML:

    # cross validatin
    def cross_apply(self,model,X,label):
        
        kfold = KFold(10, shuffle=True, random_state=1)
        
        accuracy_vector = []
        confusion_matrix_vector = []
        precision_vector = []
        recall_vector = []
        f1score_vector = []
        error_vector = []
        predicts_vector = []
        targets_vector = []

        for train_indexes,teste_indexes in kfold.split(X,label):

            X_train, X_test = X[train_indexes],X[teste_indexes]

            y_train, y_test = label[train_indexes], label[teste_indexes]

            # model
            model.fit(X_train,y_train)
            
            # labels
            targets_vector.append(y_test)
            
            # predicted
            predicted = model.predict(X_test)
            predicts_vector.append(predicted)
            
            # accuracy
            accuracy = accuracy_score(y_test, predicted)
            accuracy_vector.append(accuracy)
            
            # precision
            precision = precision_score(y_test, predicted,average='weighted')
            precision_vector.append(precision)
            
            # recall 
            recall = recall_score(y_test, predicted,average='weighted')
            recall_vector.append(recall)
            
            # f1-score
            f1score = (2*precision*recall)/(precision+recall)
            f1score_vector.append(f1score)
            
            # error
            error = mean_squared_error(y_test, predicted)
            error_vector.append(error)
            
            # confusion matrix
            aux_confusion_matrix = confusion_matrix(y_test,predicted)
            confusion_matrix_vector.append(aux_confusion_matrix)

        accuracy_mean = statistics.median(accuracy_vector)
        precision_mean = statistics.median(precision_vector)
        f1score_mean = statistics.median(f1score_vector)
        recall_mean = statistics.median(recall_vector)
        error_mean = statistics.median(error_vector)
        
        print('-------------------------------')
        print('Accuracy:', accuracy_mean)
        print('Precision:', precision_mean)
        print('Recall:', recall_mean)
        print('F1-Score:', f1score_mean)
        print('Error:', error_mean)

        #---------------GET MODEL---------------
        X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=0.20, random_state=42)

        model = model.fit(X_train, Y_train)

        result_aux = model.score(X_test, Y_test)

        print('Aux model accuracy:', result_aux)

        return predicts_vector,targets_vector,accuracy_mean,precision_mean,recall_mean,f1score_mean,error_mean, model, result_aux

    def __init__(self, percentage_test, option):
        self.percentage_test = percentage_test
        self.colunas = ['accuracy', 'precision', 'recall', 'f1-score']
        self.option = option


    def RF_classifier(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        
        #------RF with best params selected-------
        
        if self.option == 'flood':
            random_forest = RandomForestClassifier(n_estimators = 1000, min_samples_split = 8, min_samples_leaf = 3, max_features = 3, max_depth = 90, bootstrap=True)
        else:
            random_forest = RandomForestClassifier(n_estimators = 1000, min_samples_split = 10, min_samples_leaf = 3, max_features = 3, max_depth = 100, bootstrap=True)

        #------Results of cross validation------
        random_forest_values = self.cross_apply(random_forest, X, y.values.ravel())

        return [random_forest_values, scaler]
    
    def RF_bestParams(self, data):

        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)

        #----------------------------Adjust Parameters---------------------------
        
        # PARAMS
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }

        #----------FIND BEST CLASSIFIER----------
        grid_search = GridSearchCV(RandomForestRegressor(), param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)
        grid_search.fit(X, y.values.ravel())
        
        print(grid_search.best_params_)
        
        #------------------Comparisson Base and Best------------------
        
        print('-----------Grid Search-----------')

        #------RF BASE MODEL-------
         
        base_model_random_forest = RandomForestClassifier()
        
        base_model_random_forest = self.cross_apply(base_model_random_forest, X, y.values.ravel())
        
        #------RF BEST MODEL-------
         
        best_model_random_forest = RandomForestClassifier(n_estimators = 100, min_samples_split = 8, min_samples_leaf = 3, max_features = 3, max_depth = 90, bootstrap=True)

        best_model_random_forest = self.cross_apply(best_model_random_forest, X, y.values.ravel())
        
        return 1
    
    def SVM_classifier(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)

        #------SVM with best params selected-------
        
        if self.option == 'flood':
            classifier = SVC(C=10, gamma=1, kernel= 'rbf', decision_function_shape = 'ovo', shrinking= True)
        else:
            classifier = SVC(C=1, gamma=0.01, kernel= 'rbf', decision_function_shape = 'ovo', shrinking= True)
        #------Results of cross validation-----
        svc_values = self.cross_apply(classifier, X, y.values.ravel())

        return [svc_values, scaler]
    
    def SVM_bestParams(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        
        #----------------------------Adjust Parameters---------------------------
        #----------PARAMS----------
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1, 'auto']
        kernel = ['linear', 'rbf']
        decision_function_shape = ['ovo','ovr']
        shrinking = [True,False]
        param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernel, 'decision_function_shape': decision_function_shape, 'shrinking': shrinking}
        
        #----------FIND BEST CLASSIFIER----------
        grid_search_svc = GridSearchCV(SVC(), param_grid, cv=10, n_jobs = -1, verbose = 2)
        grid_search_svc.fit(X, y.values.ravel())
        
        print(grid_search_svc.best_params_)
        
        #----------------Comparisson Base and Best----------------
        
        print('-----------Grid Search-----------')
        
        #------SVM BASE MODEL-------
        base_model_svc = SVC()

        base_model_svc = self.cross_apply(base_model_svc, X, y.values.ravel())

        #------SVM BEST MODEL-------
        
        best_model_svc = SVC(C=0.01, gamma=0.001, kernel= 'linear', decision_function_shape = 'ovo', shrinking= True)

        best_model_svc = self.cross_apply(best_model_svc, X, y.values.ravel())
        
        return 1

    def NB_classifier(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)

        #------NB with best params selected-------
        if self.option == 'flood':
            gaussian_nb = GaussianNB(var_smoothing = 0.005336699231206307)
        else:
            gaussian_nb = GaussianNB(var_smoothing = 0.01)

        #------Results of cross validation-----
        
        gaussian_nb_results = self.cross_apply(gaussian_nb, X, y.values.ravel())

        return [gaussian_nb_results, scaler]

    def NB_bestParams(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        
        #----------------------------Grid Search----------------------------
        
        #----------PARAMS----------
        params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
        
        #----------FIND BEST CLASSIFIER----------
        grid_search = GridSearchCV(GaussianNB(), params_NB, cv=10, n_jobs = -1, verbose = 2)
        grid_search.fit(X, y.values.ravel())
        
        print(grid_search.best_params_)
        
        #-----------------Comparisson Base and Best-----------------
        
        print('-----------Grid Search-----------')
        
        #------NB BASE MODEL-------
        base_gaussian_nb = GaussianNB()

        base_gaussian_nb = self.cross_apply(base_gaussian_nb, X, y.values.ravel())

        #------NB BEST MODEL-------
        
        best_gaussian_nb = GaussianNB(var_smoothing = 0.000000432876)

        best_gaussian_nb = self.cross_apply(best_gaussian_nb, X, y.values.ravel())
        
        return 1

    def DT_classifier(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)

        #------DT with best params selected-------
        if self.option == 'flood':
            decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=47)
        else:
            decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=6)
        
        #------Results of cross validation-----

        decision_tree_values = self.cross_apply(decision_tree_classifier, X, y.values.ravel())

        return [decision_tree_values, scaler]
    
    def DT_bestParams(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        
        #----------------------------Grid Search----------------------------
        
        #----------PARAMS----------
        max_depth = range(1,50)
        criterion = ['gini', 'entropy']
        param_grid = {'max_depth': max_depth, 'criterion' : criterion}
        
        #----------FIND BEST CLASSIFIER----------
        grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, n_jobs = -1, verbose = 2)
        grid_search.fit(X, y.values.ravel())
        
        print(grid_search.best_params_)
        
        #------------------------Comparisson Base and Best---------------------------------
        
        print('-----------Grid Search-----------')
        
        #------DT BASE MODEL-------
        decision_tree_base = DecisionTreeClassifier()

        decision_tree_base = self.cross_apply(decision_tree_base, X, y.values.ravel())

        #------DT BEST MODEL-------
        
        decision_tree_best = DecisionTreeClassifier(criterion = 'gini', max_depth=6)

        decision_tree_best = self.cross_apply(decision_tree_best, X, y.values.ravel())
        
        return 1

    def RL_classifier(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)

        #------LR with best params selected-------
        if self.option == 'flood':
            logistic_regression_classifier = LogisticRegression(C=1, penalty = 'l2', multi_class='ovr', solver='liblinear')
        else:
            logistic_regression_classifier = LogisticRegression(C=0.01, penalty = 'l2', multi_class='ovr', solver='liblinear')
        
        #------Results of cross validation-----
        logistic_regression_values = self.cross_apply(logistic_regression_classifier, X, y.values.ravel())

        return [logistic_regression_values, scaler]
    
    def RL_bestParams(self, data):
        
        X = data.iloc[:, 0: len(data.columns)-1]
        y = data.iloc[:, len(data.columns)-1:len(data.columns)]
        
        #------Feature scaling------
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        
        #----------------------------Grid Search----------------------------
        
        #----------PARAMS----------
        param_grid = {'penalty': ['l1', 'l2'],'C':[0.001,0.009,0.01,0.09,1,5,10,25], 'solver' : ['liblinear','saga'], 'multi_class' : ['ovr','auto']}
        
        #------RL BEST MODEL-------
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=10, n_jobs = -1, verbose = 2)
        grid_search.fit(X, y.values.ravel())
        
        print(grid_search.best_params_)
        
        #------------------------Comparisson Base and Best---------------------------------
        
        print('-----------Grid Search-----------')
        
        #------RL BASE MODEL-------
        
        logistic_regression_base = LogisticRegression()

        logistic_regression_base = self.cross_apply(logistic_regression_base, X, y.values.ravel())

        #------RL BEST MODEL-------
        
        logistic_regression_best = LogisticRegression(C=0.09, penalty = 'l1', multi_class='ovr', solver='liblinear')

        logistic_regression_best = self.cross_apply(logistic_regression_best, X, y.values.ravel())
        
        return 1

