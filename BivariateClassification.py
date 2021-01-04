import numpy as np
import pandas as pd 

# picking models for prediction.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ensemble models for better performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

def best_n_estimator():
        error_rate = []
        # Will take some time
        k_values = list(filter(lambda x: x % 2 == 1, range(0, 50)))
        for i in k_values:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train, y_train)
            pred_i = knn.predict(x_test)
            error_rate.append(np.mean(pred_i != y_test))
        best_k_index = error_rate.index(np.min(error_rate))
        best_n_estimator = best_k_index * 2 + 1

        return best_n_estimator

def KNNClassifier():
    best_n = best_n_estimator()

    knn_classifier = KNeighborsClassifier(
        n_neighbors= best_n)
    knn_classifier.fit(x_train, y_train)

class BivariateClassification():
    
    def __init__(self):

        self.__logistic_regression = LogisticRegression()
        self.__decision_tree_classifier = DecisionTreeClassifier()
        self.__support_vector_machine = SVC()
        self.__adaboost_classifer = AdaBoostClassifier()
        self.__random_forest_classifier = RandomForestClassifier()