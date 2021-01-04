import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# ensemble models for better performance in classification
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

#metrics to check our model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

from MiscellaneousFunctions import root_mean_squared_error

class Regression():
    '''
    This class implements multiple regression algorithms to already encoded and scaled data. 
    It required the data to be numeric and divided into train and test sets.
    '''
    def __init__(self):

        self.linear_regressor = LinearRegression()
        self.support_vector_regressor = SVR()
        self.decision_tree_regressor = DecisionTreeRegressor()
        self.random_forest_regressor = RandomForestRegressor()
        self.adaboost_regressor = AdaBoostRegressor()
                
        self.all_models = [self.linear_regressor, self.support_vector_regressor, self.decision_tree_regressor, 
                           self.random_forest_regressor, self.adaboost_regressor]
        self.all_model_names = ['Linear Regression', 'Support Vector Regressor', 'Decision Tree Regressor', 
                           'Random Forest Regressor', 'Adaboost Regressor']

        self.train_scores = []
        self.test_scores = []        
        self.metric_list = [mean_absolute_error, mean_squared_error, root_mean_squared_error]
        self.metrics = []
        data = {'Model Names': self.all_model_names}
        self.all_model_info = pd.DataFrame(data)
    
    def fit(self, x_train, x_test, y_train, y_test):
        '''
        fits models to data and stores results for metrics
        '''
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        for model in self.all_models:
            model.fit(x_train, y_train)
            
            train_score = model.score(x_train,y_train)
            self.train_scores.append(train_score)
            
            test_score = model.score(x_train,y_train)
            self.test_scores.append(train_score)
            y_predict = model.predict(self.x_test)
            
        self.all_model_info['Train Score'] = self.train_scores
        self.all_model_info['Test Score'] = self.test_scores
        self.apply_metrics()

    def apply_metrics(self):
        self.metrics = []
        for metric in self.metric_list:
            metric_name = str(metric).split(' ')[1]
            for model in self.all_models:
                metric_item = metric(self.y_test, model.predict(self.x_test))
                self.metrics.append(metric_item)
                
            self.all_model_info[metric_name] = self.metrics
            self.metrics = []

    def display_report(self):
        return self.all_model_info