# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:08:16 2020

@author: mishr
"""

#picking models for prediction.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#ensemble models for better performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#Model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

class StatisticalModelImplenter():
    
    def __init__(self):

        self.__all_models = []
        self.__all_model_names = []
        
        
    def fit(self, x_train, y_train):
        self.__x_train = x_train
        self.__y_train = y_train
        
        logistic_regressor = LogisticRegression()
        logistic_regressor.fit(self.__x_train, self.__y_train)
        
        self.__all_model_names.append('Logistic Regression')
        self.__all_models.append(logistic_regressor)

        #Decision Tree Classifier
        dtree = DecisionTreeClassifier(criterion = 'entropy')
        dtree.fit(self.__x_train, self.__y_train)
        
        self.__all_model_names.append('Decision Tree Classifier')
        self.__all_models.append(dtree)


        #Support Vector Classifier
        svc = SVC(kernel='rbf')
        svc.fit(self.__x_train, self.__y_train)
        
        self.__all_model_names.append('Support Vector Classifier')
        self.__all_models.append(svc)

        #Random Forest Classifier
        random_forest = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 42)
        random_forest.fit(self.__x_train, self.__y_train)
        
        self.__all_model_names.append('Random Forest Classifier')
        self.__all_models.append(random_forest)
        
        #Adaboost Classifier
        adaboost_classifier = AdaBoostClassifier(n_estimators=3)
        adaboost_classifier.fit(x_train,y_train)
        
        self.__all_model_names.append('Adaboost Classifier')
        self.__all_models.append(adaboost_classifier)
        
        #Fit complete message
        print('All models have been fit.')
    
    
    def fit_test_set(self, user_x_test, user_y_test):
        
        self.__x_test = user_x_test
        self.__y_test = user_y_test

    
    def apply_metric(self,  metric):
        
        metric_list = []
        for i,model in enumerate(self.__all_models):
            metric_item = metric(self.__y_test, model.predict(self.__x_test))
            metric_list.append(metric_item)

        self.report_printer(metric_list)
    
    def report_printer(self, list_of_metric):
        
        all_model_metrics = dict(zip(self.__all_model_names, list_of_metric))

        for name, matrix in all_model_metrics.items():
            print('{}\n{}\n\n'.format(name, matrix))