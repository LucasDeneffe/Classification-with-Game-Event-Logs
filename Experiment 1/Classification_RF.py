#%%

import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.model_selection import train_test_split

# Import data
PF1 = np.load(f"Data collection/Arrays/PF1.npy")
PF2 = np.load(f"Data collection/Arrays/PF2.npy")
X = np.load(f"Data collection/Arrays/X.npy")
yrank = np.load(f"Data collection/Arrays/yrank.npy")
ywin = np.load(f"Data collection/Arrays/ywin.npy")
with open(f"Data collection/Arrays/RFlog.pkl", "rb") as handle:
    RFlog = pickle.load(handle)

PF1_swap = np.swapaxes(PF1, 1, 2)
PF2_swap = np.swapaxes(PF2, 1, 2)

def return_mean(a):
    trimmed_array = np.trim_zeros(a, trim="b")
    if trimmed_array.size == 0:
        trimmed_array = np.array([0])
    return np.mean(trimmed_array)


# Take average SQ and APM
PF1_avg = np.apply_along_axis(return_mean, 2, PF1_swap)
PF2_avg = np.apply_along_axis(return_mean, 2, PF2_swap)

RFlog_array = np.asarray(RFlog)
print(RFlog_array.shape)
print(PF1_avg.shape)

# Combine all inputs
Dataset = np.hstack((RFlog_array, PF1_avg, PF2_avg))

ywin_array = np.asarray(ywin)
yrank_array = np.asarray(yrank)
y_array = np.vstack((ywin_array, yrank_array))
y_array = np.swapaxes(y_array, 0, 1)

# Splitting data
global X_train, X_test, X_val, y_array_train, y_array_test, y_array_val
X_train, X_val, y_array_train, y_array_val = train_test_split(
    Dataset, y_array, test_size=0.3, random_state=15
)
X_val, X_test, y_array_val, y_array_test = train_test_split(
    X_val, y_array_val, test_size=1 / 3, random_state=15
)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


### Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Tune the Random Forest model
# Number of trees in random forest
n_estimators = [500, 1000, 2000]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [10, 20, 50, 100]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
    "random_state": [16],
}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(
    estimator=rf,
    param_grid=random_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
)
# Fit the random search model
rf_random.fit(X_train, y_array_train.T[0])

def evaluate(random, model, test_features, test_labels):
    pred = model.predict(test_features)
    score = model.score(test_features, test_labels)
    confusion = classification_report(test_labels, pred)
    confusions_matrix = confusion_matrix(model.predict(test_features), test_labels)
    #Save parameters from best model
    best_model_parameters = random.best_params_

    #Text file which saves scoer, confusion matrix and parameters
    file = open("Experiment1/RF_models/model_acc_RF_win.txt", "w")
    file.write(str(score))
    file.write(str(confusion))
    file.write(str(best_model_parameters))
    file.write(np.array2string(confusions_matrix, separator=", "))
    file.close()
    data = pd.DataFrame(random.cv_results_)
    #All results of the tuning
    data.to_csv("Experiment1/RF_models/rf_win_results.csv")
    print(f"RFC score = {score} \n \n {confusion} \n\n {confusions_matrix}")


# Final evaluation
best_random_win = rf_random.best_estimator_
evaluate(rf_random, best_random_win, X_test, y_array_test.T[0])

import pickle

# Save the trained model as a pickle string.
with open(
    r"Models/RF_win.pkl",
    "wb",
) as handle:
    pickle.dump(best_random_win, handle, protocol=pickle.HIGHEST_PROTOCOL)


## RANKING
rf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(
    estimator=rf,
    param_grid=random_grid,
    cv=3,
    verbose=1,
    n_jobs=-1,
)
# Fit the random search model
rf_random.fit(X_train, y_array_train.T[1])
# %%


def evaluate2(random, model, test_features, test_labels):
    pred = model.predict(test_features)
    score = model.score(test_features, test_labels)
    confusion = classification_report(test_labels, pred)
    confusions_matrix = confusion_matrix(model.predict(test_features), test_labels)
    best_random_params = random.best_params_

    #Text file which saves scoer, confusion matrix and parameters
    a = open("Experiment1/RF_models/model_acc_RF_rank.txt", "w")
    a.write(str(score))
    a.write(str(confusion))
    a.write(str(best_random_params))
    a.write(np.array2string(confusions_matrix, separator=", "))
    a.close()
    data = pd.DataFrame(random.cv_results_)
    #All results of the tuning
    data.to_csv("Experiment1/RF_models/rf_rank_results.csv")
    print(f"RFC score = {score} \n \n {confusion} \n\n {confusions_matrix}")


with open(
    r"Models/RF_rank.pkl",
    "wb",
) as handle:
    pickle.dump(best_random_win, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Final evaluation
best_random_rank = rf_random.best_estimator_
evaluate2(rf_random, best_random_rank, X_test, y_array_test.T[1])
