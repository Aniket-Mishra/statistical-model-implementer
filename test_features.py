import numpy as np
import pandas as pd

# picking regression models for prediction
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# ensemble models for better performance in classification
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# Picking bivariate classification models
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ensemble models for better performance in classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score


class StatisticalModelImplenter():
    
    '''
    Parent class containing common functions 
    '''
    def __init__(self):

        self.all_models = []
        self.all_model_names = []

    def get_models(self):

        '''
        returns a dictionary containing names with the corosponing model
        '''
        model_dict = {self.all_models[i]: self.all_model_names[i] for i in range(len(self.all_models))}
        return model_dict

    def fit(self, x_train, y_train):
        self.__x_train = x_train
        self.__y_train = y_train

        logistic_regressor = LogisticRegression()
        logistic_regressor.fit(self.__x_train, self.__y_train)

        self.all_model_names.append('Logistic Regression')
        self.all_models.append(logistic_regressor)

        # Decision Tree Classifier
        dtree = DecisionTreeClassifier(criterion='entropy')
        dtree.fit(self.__x_train, self.__y_train)

        self.all_model_names.append('Decision Tree Classifier')
        self.all_models.append(dtree)

        # Support Vector Classifier
        svc = SVC(kernel='rbf')
        svc.fit(self.__x_train, self.__y_train)

        self.all_model_names.append('Support Vector Classifier')
        self.all_models.append(svc)

        # K-Nearest Neighbors Classifier
        self.best_n_estimator()

        knn_classifier = KNeighborsClassifier(
            n_neighbors=self.__best_n_estimator)
        knn_classifier.fit(self.__x_train, self.__y_train)

        self.all_model_names.append('KNeighbors CLassifier')
        self.all_models.append(knn_classifier)

        # Random Forest Classifier
        random_forest = RandomForestClassifier(
            n_estimators=15, criterion='entropy', random_state=42)
        random_forest.fit(self.__x_train, self.__y_train)

        self.all_model_names.append('Random Forest Classifier')
        self.all_models.append(random_forest)

        # Adaboost Classifier
        adaboost_classifier = AdaBoostClassifier(n_estimators=3)
        adaboost_classifier.fit(x_train, y_train)

        self.all_model_names.append('Adaboost Classifier')
        self.all_models.append(adaboost_classifier)

        # Fit complete message
        print('All models have been fit.')

    def best_n_estimator(self):
        error_rate = []
        # Will take some time
        k_values = list(filter(lambda x: x % 2 == 1, range(0, 50)))
        for i in k_values:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.__x_train, self.__y_train)
            pred_i = knn.predict(self.__x_test)
            error_rate.append(np.mean(pred_i != self.__y_test))
        best_k_index = error_rate.index(np.min(error_rate))
        self.__best_n_estimator = best_k_index * 2 + 1

    def fit_test_set(self, user_x_test, user_y_test):

        self.__x_test = user_x_test
        self.__y_test = user_y_test

    def apply_metric(self,  metric):

        metric_list = []
        for _, model in enumerate(self.all_models):
            metric_item = metric(self.__y_test, model.predict(self.__x_test))
            metric_list.append(metric_item)

        self.report_printer(metric_list)

    def report_printer(self, list_of_metric):

        all_model_metrics = dict(zip(self.all_model_names, list_of_metric))

        for name, matrix in all_model_metrics.items():
            print('{}\n{}\n\n'.format(name, matrix))
            
    def count_values(self, list_of_values):
        zeros = list_of_values.count(0)
        ones = list_of_values.count(1)

        if zeros == ones:
            print('zeros == 1')
        elif zeros > ones:        
            return 0
        else:
            return 1
        
            
    def best_model(self):
        
        output_count = []
        for _, model in enumerate(self.all_models):
            
            list_of_outputs = list(model.predict(self.__x_test))
            output_count.append(self.count_values(list_of_outputs))
        
        max_predicted_val = self.count_values(output_count)
        print(max_predicted_val)
            
class Regression(StatisticalModelImplenter):
    def __init__(self):
        self.__linear_regressor = None
        self.__support_vector_regressor = None
        self.__decision_tree_regressor = None
        self.__random_forest_regressor = None
        self.adaboost_regressor = None


class BinomialClassification(StatisticalModelImplenter):
    def __init__(self):
        self.__logistic_regression = None
        self.__gaussian_nb = None


class MulticlassClassification(StatisticalModelImplenter):
    
    def __init__(self):
        self.__decision_tree_classifier = None
        self.__support_vector_machine = None
        self.adaboost_classifer = None
        self.__random_forest_classifier = None

        self.__kneighbors_classifier = None
        self.__best_n_estimator = None
