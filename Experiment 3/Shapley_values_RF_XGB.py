import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import sklearn
import shap

# Feature names
stat_names = [
    "minerals_current",
    "vespene_current",
    "minerals_collection_rate",
    "vespene_collection_rate",
    "workers_active_count",
    "minerals_used_in_progress_army",
    "minerals_used_in_progress_economy",
    "minerals_used_in_progress_technology",
    "vespene_used_in_progress_army",
    "vespene_used_in_progress_economy",
    "vespene_used_in_progress_technology",
    "minerals_used_current_army",
    "minerals_used_current_economy",
    "minerals_used_current_technology",
    "vespene_used_current_army",
    "vespene_used_current_economy",
    "vespene_used_current_techonology",
    "minerals_lost_army",
    "minerals_lost_economy",
    "minerals_lost_technology",
    "vespene_lost_army",
    "vespene_lost_economy",
    "vespene_lost_technology",
    "minerals_killed_army",
    "minerals_killed_economy",
    "minerals_killed_technology",
    "vespene_killed_army",
    "vespene_killed_economy",
    "vespene_killed_technology",
    "food_used",
    "food_made",
    "minerals_used_active_forces",
    "vespene_used_active_forces",
    "ff_minerals_lost_army",
    "ff_minerals_lost_economy",
    "ff_minerals_lost_technology",
    "ff_vespene_lost_army",
    "ff_vespene_lost_economy",
    "ff_vespene_lost_technology",
]
player_names = ["p1", "p2"]
APM_names = ["aAPM_p1", "aSQ_p1", "aAPM_p2", "aSQ_p2"]

column_names = []
# Combine stats with player
for name in player_names:
    for stat in stat_names:
        column_names.append(f"{name}_{stat}")
# Combine paper features
for name in APM_names:
    column_names.append(name)

# Load data

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


PF1_avg = np.apply_along_axis(return_mean, 2, PF1_swap)
PF2_avg = np.apply_along_axis(return_mean, 2, PF2_swap)

RFlog_array = np.asarray(RFlog)
print(RFlog_array.shape)
print(PF1_avg.shape)

Dataset = np.hstack((RFlog_array, PF1_avg, PF2_avg))

# %%
ywin_array = np.asarray(ywin)
yrank_array = np.asarray([x - 2 for x in yrank])
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

# RF models
## Load Random Forest Win
with open(
    r"Models\RF_win.pkl",
    "rb",
) as handle:
    RF_pickle_win = pickle.load(handle)

# Train the explainer on the training data
explainer = shap.TreeExplainer(
    RF_pickle_win,
    X_train,
    feature_perturbation="interventional",
    feature_names=column_names,
    #Probability outputs
    model_output='probability'

)
# Compute shapley values of X_test
shap_valuesRF_win = explainer.shap_values(X_test)

# Load random forest rank
with open(
    r"Models/RF_rank.pkl",
    "rb",
) as handle:
    RF_pickle_rank = pickle.load(handle)

# Train explainer on train data
explainer2 = shap.TreeExplainer(
    RF_pickle_rank,
    X_train,
    feature_perturbation="interventional",
    feature_names=column_names,
)
shap_valuesRF_rank_2 = explainer2.shap_values(X_test, approximate=True)

# %%
# Figure showing the summary plot of features explaining winning a match
fig1 = shap.summary_plot(
    shap_valuesRF_win[1], X_test, feature_names=column_names, show=False, max_display=10)
plt.savefig('Experiment 3/RF_win_feature_importance_new.jpg',  bbox_inches='tight')

# Figure showing the summary plot of features explaining players rank
fig2 = shap.summary_plot(shap_valuesRF_rank_2, X_test, feature_names=column_names, show=False,
                         max_display=10, class_names=["Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"])
plt.savefig('Experiment 3/RF_rank_feature_importance_new_test.jpg',  bbox_inches='tight')

# XG Models
#Load models
XG_rank = XGBClassifier()
XG_rank.load_model(
    r"Models/XG_rank.json")
XG_win = XGBClassifier()
XG_win.load_model(
    r"Models/XG_win.json")

# Make explainer based on traning data
explainer3 = shap.explainers.Tree(
    XG_win,
    X_train,
    feature_perturbation="interventional",
    feature_names=column_names,
    model_output="probability",
    link="logit"
)
# Compute shap values for X_test
shap_valuesXG_win = explainer3.shap_values(X_test)

# Make explainer for XG_rank
explainer4 = shap.TreeExplainer(
    XG_rank,
    X_train,
    feature_perturbation="interventional",
    feature_names=column_names,
    # No probability possible
)

# Compute shap values for X_test
shap_valuesXG_rank = explainer4.shap_values(X_test, approximate=True)

fig3 = shap.summary_plot(shap_valuesXG_win, X_test,
                        feature_names=column_names, show=False, max_display=10)
plt.savefig('Experiment 3/XG_win_feature_importance.jpg',  bbox_inches='tight')

fig4 = shap.summary_plot(shap_valuesXG_rank, X_test, feature_names=column_names, show=False,
                        max_display=10, class_names=["Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"])
plt.savefig('Experiment 3/XG_rank_feature_importance_new.jpg',  bbox_inches='tight')

# Make waterfall plot for sample 4
chosen_instance = X_test[3]
shap_values_instance = explainer.shap_values(chosen_instance)
shap.waterfall_plot(shap.Explanation(values=shap_values_instance[1],
                                         base_values=explainer.expected_value[1],
                                         data=chosen_instance,  # added this line
                                         feature_names=column_names,), max_display=6,
                                         show=False)
plt.savefig('Experiment 3/RF_win_sample4_new.jpg', bbox_inches='tight')

# Make waterfall plot for sample 5
chosen_instance = X_test[4]
shap_values_instance = explainer.shap_values(chosen_instance)
shap.waterfall_plot(shap.Explanation(values=shap_values_instance[1],
                                         base_values=explainer.expected_value[1],
                                         data=chosen_instance,  # added this line
                                         feature_names=column_names,), max_display=6,
                                         show=False)
plt.savefig('Experiment 3/RF_win_sample5_new.jpg', bbox_inches='tight')

# Make waterfall plot for index 36
chosen_instance = X_test[36]
shap_values_instance = explainer2.shap_values(chosen_instance)
shap.waterfall_plot(shap.Explanation(values=shap_values_instance[5],
                                         base_values=explainer2.expected_value[5],
                                         data=chosen_instance,  # added this line
                                         feature_names=column_names,), max_display=6,
                                         show=False)
plt.savefig('Experiment 3/RF_rank_index36.jpg', bbox_inches='tight')

# Make waterfall plot for index 8
chosen_instance = X_test[8]
shap_values_instance = explainer2.shap_values(chosen_instance)
shap.waterfall_plot(shap.Explanation(values=shap_values_instance[0],
                                         base_values=explainer2.expected_value[0],
                                         data=chosen_instance,  # added this line
                                         feature_names=column_names,), max_display=6,
                                         show=False)
plt.savefig('Experiment 3/RF_rank_index8.jpg', bbox_inches='tight')
