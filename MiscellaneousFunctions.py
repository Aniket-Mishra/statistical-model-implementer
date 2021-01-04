import numpy as np
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.neighbors import KNeighborsClassifier

def root_mean_squared_error(y_real, y_predicted):
    return np.sqrt(mean_squared_error(y_real, y_predicted))

def display_classification_report(all_model_names, all_models, y_test, x_test):
    for name, model in zip(all_model_names, all_models):
            metric_item = classification_report(y_test, model.predict(x_test))
            print(name)
            print(metric_item)

def best_n_estimator(x_train, y_train, x_test, y_test):
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

def KNNClassifier(x_train, y_train, x_test, y_test):
    best_n = best_n_estimator(x_train, y_train, x_test, y_test)

    knn_classifier = KNeighborsClassifier(
        n_neighbors= best_n)
    knn_classifier.fit(x_train, y_train)