from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
from sklearn import ensemble
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from task import DataGenerate

X_train, X_test, y_train, y_test = DataGenerate.generateData()

# Find whether indicator feature will help the result.

# clf_B = linear_model.Lasso()
# for clf in [clf_B]:
#     print("Current model is " + clf.__class__.__name__)
#     predictions = (clf.fit(X_train, y_train)).predict(X_test)
#     print("The final model's mean_squared_error on the testing data is " + str(metrics.mean_squared_error(y_test, predictions)))


# Choose the best model
def choose_model():
    clf_A = tree.DecisionTreeRegressor()
    clf_B = linear_model.Lasso()
    clf_C = linear_model.SGDRegressor()
    clf_D = linear_model.Ridge()
    clf_E = ensemble.RandomForestRegressor()
    for clf in [clf_A, clf_B, clf_C, clf_D, clf_E]:
        print("Current model is " + clf.__class__.__name__)
        predictions = (clf.fit(X_train, y_train)).predict(X_test)
        print("The final model's mean_squared_error on the testing data is " + str(metrics.mean_squared_error(y_test, predictions)))




# Analyse the features
def analyse_feature():
    model = ensemble.RandomForestRegressor()
    model.fit(X_train, y_train)
    data= list(model.feature_importances_)

    head = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'MACD', 'ADX', 'CCI', 'WILLR', 'interest_rate', 'unemployment_rate']
    y_pos = np.arange(len(head))
    plt.bar(y_pos, data)
    plt.xticks(y_pos, head)
    plt.show()

# Analyse the features
def model_tuning():
    clf = ensemble.RandomForestRegressor()

    parameters = {'n_estimators': range(2, 40, 2), 'min_weight_fraction_leaf' :np.arange(0, 0.5, 0.1)}

    scorer = metrics.make_scorer(metrics.r2_score)

    grid_obj = model_selection.GridSearchCV(clf, param_grid=parameters, scoring=scorer)

    grid_fit = grid_obj.fit(X_train, y_train)

    best_clf = grid_fit.best_estimator_
    best_predictions = best_clf.predict(X_test)

    print("The final model's mean_squared_error on the testing data is " + str(
        metrics.mean_squared_error(y_test, best_predictions)))


if __name__ == '__main__':
    # choose_model()
    # analyse_feature()
    model_tuning()
