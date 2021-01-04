import numpy as np
import pandas as pd 

# picking models for prediction.
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ensemble models for better performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score

from MiscellaneousFunctions import display_classification_report

class BivariateClassification():
    
    def __init__(self):

        self.support_vector_machine = SVC()
        self.decision_tree_classifier = DecisionTreeClassifier()
        self.random_forest_classifier = RandomForestClassifier()
        self.adaboost_classifer = AdaBoostClassifier()

        self.all_models = [self.support_vector_machine, self.decision_tree_classifier, 
                           self.random_forest_classifier, self.adaboost_classifer]
        self.all_model_names = ['Support Vector Classifier', 'Decision Tree Classifier', 
                           'Random Forest Classifier', 'Adaboost Classifier']

        self.metric_names = ['Accuracy Score', 'Confusion Matrix',
                            'F1 Score']

        self.train_scores = []
        self.test_scores = []        
        self.metric_list = [accuracy_score, confusion_matrix, f1_score]
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
        display_classification_report(self.all_model_names, self.all_models, self.y_test, self.x_test)
        return self.all_model_info
    