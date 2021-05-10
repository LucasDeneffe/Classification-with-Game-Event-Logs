
import pandas as pd 
import numpy as np
import sklearn
import scipy
import pickle
import gc
from sklearn.model_selection import train_test_split
import tensorflow as tf

pip install git+https://github.com/ck37/coral-ordinal/
import coral_ordinal as coral
import xgboost

def import_data():    
    global yrankcorrect, ywin, X, Xp1, Xp2,PF1, PF2, time1, time2, RFlog, unique
    
    yrank = np.load(r"Data collection/Arrays/yrank.npy")
    yrankcorrect = np.asarray([x - 2 for x in yrank])
    ywin = np.load(r"Data collection/Arrays/ywin.npy")
    ywin = np.asarray(ywin)
    X = np.load(r"Data collection/Arrays/X.npy")
    Xp1 = np.load(r"Data collection/Arrays/Xp1.npy")
    Xp2 = np.load(r"Data collection/Arrays/Xp2.npy")
    PF1=  np.load(r"Data collection/Arrays/PF1.npy")
    PF2=  np.load(r"Data collection/Arrays/PF2.npy")

    with open(r"Data collection/Arrays/unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
        
    with open(r"Data collection/Arrays/RFlog.pkl", 'rb') as handle:
        RFlog = pickle.load(handle)
    
    with open(r"Data collection/Arrays/time1.pkl", 'rb') as handle:
        time1 = pickle.load(handle)
    
    with open(r"Data collection/Arrays/time2.pkl", 'rb') as handle:
        time2 = pickle.load(handle)

        
def combine_PF():
    global PF
    print(PF1.shape)
    print(PF2.shape)
    PF = np.dstack((PF1, PF2))
    gc.collect()

def split_log():
    
    gc.collect()
    #Features from PlayerStatsEvent
    global X_train, X_val, X_test, yrank_train, yrank_val, yrank_test, normalized_X
    X_train, X_val, yrank_train, yrank_val = train_test_split(X, yrankcorrect, test_size = 0.3, random_state = 15)
    X_val, X_test, yrank_val, yrank_test = train_test_split(X_val, yrank_val, test_size = 1/3, random_state = 15)
    #del normalised_X
    gc.collect()
    #sequence of player 1
    global Xp1_train, Xp1_val, Xp1_test, ywin_train, ywin_val, ywin_test, Xp1
    Xp1_train, Xp1_val, ywin_train, ywin_val = train_test_split(Xp1, ywin, test_size = 0.3 , random_state = 15)
    Xp1_val, Xp1_test, ywin_val, ywin_test = train_test_split(Xp1_val, ywin_val, test_size = 1/3 , random_state = 15)
    gc.collect()
    del Xp1
    #sequence of player 2
    global Xp2_train, Xp2_val, Xp2_test, Xp2
    Xp2_train, Xp2_val = train_test_split(Xp2, test_size = 0.3, random_state = 15)
    Xp2_val, Xp2_test = train_test_split(Xp2_val, test_size = 1/3 , random_state = 15)
    del Xp2
    gc.collect()

    #features based on paper
    global PF_train, PF_val, PF_test, normalized_PF
    PF_train, PF_val  = train_test_split(PF, test_size = 0.3, random_state = 15)
    PF_val, PF_test = train_test_split(PF_val, test_size = 1/3 , random_state = 15)
    #del PF
    gc.collect()

    #RFlog
    global RFlog_train, RFlog_val, RFlog_test, RFlog
    RFlog_train, RFlog_val = train_test_split(RFlog, test_size = 0.3, random_state = 15)
    RFlog_val, RFlog_test = train_test_split(RFlog_val, test_size = 1/3 , random_state = 15)

    #time
    global time1_train, time1_val, time1_test, time2_train, time2_val, time2_test
    time1_train, time1_val, time2_train, time2_val = train_test_split(time1, time2, test_size = 0.3, random_state = 15)
    time1_val, time1_test, time2_val, time2_test = train_test_split(time1_val, time2_val, test_size = 1/3, random_state = 15)


# %%
import_data()
combine_PF()
split_log()


# %%
# CASE I 
model_I = tf.keras.models.load_model("Models/CASE1MODEL" , custom_objects = {"OrdinalCrossEntropy": coral.OrdinalCrossEntropy(num_classes=6),"MeanAbsoluteErrorLabels": coral.MeanAbsoluteErrorLabels()})

# CASE II
model_II = tf.keras.models.load_model("Models/CASE2MODEL" , custom_objects = {"OrdinalCrossEntropy": coral.OrdinalCrossEntropy(num_classes=6),"MeanAbsoluteErrorLabels": coral.MeanAbsoluteErrorLabels()})

# CASE III
model_III = tf.keras.models.load_model("Models/CASE3MODEL" , custom_objects = {"OrdinalCrossEntropy": coral.OrdinalCrossEntropy(num_classes=6),"MeanAbsoluteErrorLabels": coral.MeanAbsoluteErrorLabels()})

# RF
with open(
    r"Models/RF_rank",
    "rb",
) as handle:
    model_RF_rank = pickle.load(handle)

with open(
    r"Models/RF_win",
    "rb",
) as handle:
    model_RF_win = pickle.load(handle)

#Gaat niet in google colab voor een reden
# XGB
from xgboost import XGBClassifier
model_XG_rank = XGBClassifier()
model_XG_rank.load_model(r"Models\XG_rank.json")

from xgboost import XGBClassifier
model_XG_win = XGBClassifier()
model_XG_win.load_model(r"Models\XG_win.json")

# %%
# Take 10 random samples of 100 datapoints out of the test set
nfolds = 30
length = 100 #length of each fold

folds = {}

inputs = {"Xp1_test":Xp1_test, "Xp2_test":Xp2_test, "X_test": X_test, "PF_test" : PF_test, "RFLOG": RFlog_test, "ywin": ywin_test, "yrank": yrank_test}
for i in range (nfolds):
  foldlist = []
  R = np.random.default_rng()
  inti = R.integers(100)
  print(inti)
  for name, input in inputs.items():
    discard, temp = train_test_split(input, test_size=0.1, random_state=inti)
    foldlist.append(temp)
  folds[i] = foldlist


#make_predictions
def return_mean(a):
    trimmed_array = np.trim_zeros(a, trim="b")
    if trimmed_array.size == 0:
        trimmed_array = np.array([0])
    return np.mean(trimmed_array)

def classification_dataset(RFlog_array, PF_test):
  PF_temp = np.swapaxes(PF_test, 1, 2)
  PF_avg = np.apply_along_axis(return_mean, 2, PF_temp)
  dataset = np.hstack((RFlog_array, PF_avg))
  print(dataset.shape)
  return dataset 

def classification_accuracy(dataset, y_test):
    try:
      return model.score(dataset, y_test)
    except:
      print("something went wrong here")

def classification_confusion_matrix(dataset, y_test):
    from sklearn.metrics import confusion_matrix
    return np.asarray(confusion_matrix(y_test, model.predict(dataset)))
  
def ordinal_labels(name, Xp1_test, Xp2_test, X_test, PF_test):
    from scipy import special

    if name == "Case_I":
      ordinal_logits = model.predict([Xp1_test, Xp2_test])
    elif name == "Case_II":
      ordinal_logits = model.predict([X_test, PF_test])
    elif name == "Case_III":
      ordinal_logits = model.predict([Xp1_test, Xp2_test, X_test, PF_test])
    else:
      print('something went wrong')
    #  Compare to logit-based cumulative probs
    cum_probs = pd.DataFrame(ordinal_logits[0]).apply(special.expit)
    labels = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
    
    return labels


def ordinal_accuracy(y_true, labels):
    return(np.mean(labels == y_true))

def ordinal_confusion_matrix(y_true, labels):
    return tf.math.confusion_matrix(y_true, labels, num_classes = 6, dtype=tf.dtypes.int32).numpy()

def get_win_accuracy(name, Xp1_test, Xp2_test, X_test, PF_test, yrank_test, ywin_test):
    if name == "Case_I":
      return model.evaluate([Xp1_test, Xp2_test], [yrank_test, ywin_test])[4]
    elif name == "Case_II":
      return model.evaluate([X_test, PF_test], [yrank_test, ywin_test])[4]
    elif name == "Case_III":
      return model.evaluate([Xp1_test, Xp2_test, X_test, PF_test], [yrank_test, ywin_test])[4]
    else:
      print("something went wrong with lstm win acc")

def get_average_distance(matrix):
    #Calculate the average distance the predicted label was from the correct label. Under 1 means the label was on average mostly correct.
    dsilver = (matrix[0,0]*0 + matrix[0,1]*1 + matrix[0,2]*2 + matrix[0,3]*3 + matrix[0,4]*4 + matrix[0,5]*5)/np.sum(matrix[0])
    dgold = (matrix[1,0]*1 + matrix[1,1]*0 + matrix[1,2]*1 + matrix[1,3]*2 + matrix[1,4]*3 + matrix[1,5]*4)/np.sum(matrix[1])
    dplat = (matrix[2,0]*2 + matrix[2,1]*1 + matrix[2,2]*0 + matrix[2,3]*1 + matrix[2,4]*2 + matrix[2,5]*3)/np.sum(matrix[2])
    ddia = (matrix[3,0]*3 + matrix[3,1]*2 + matrix[3,2]*1 + matrix[3,3]*0 + matrix[3,4]*1 + matrix[3,5]*2)/np.sum(matrix[3])
    dma = (matrix[4,0]*4 + matrix[4,1]*3 + matrix[4,2]*2 + matrix[4,3]*1 + matrix[4,4]*0 + matrix[4,5]*1)/np.sum(matrix[4])
    dgm = (matrix[5,0]*5 + matrix[5,1]*4 + matrix[5,2]*3 + matrix[5,3]*2 + matrix[5,4]*1 + matrix[5,5]*0)/np.sum(matrix[5])
    total_acc = ((dsilver*np.sum(matrix[0])) +  (dgold*np.sum(matrix[1])) + (dplat*np.sum(matrix[2])) + (ddia*np.sum(matrix[3])) + (dma*np.sum(matrix[4])) +  (dgm*np.sum(matrix[5])))/sum(sum(matrix))

    return [dsilver, dgold, dplat, ddia, dma, dgm, total_acc]
     

def get_missclassified_distance(matrix):
    #Calculate the average distance of the missclassified instances. Closer to one is better.
    dmsilv = (matrix[0,0]*0 + matrix[0,1]*1 + matrix[0,2]*2 + matrix[0,3]*3 + matrix[0,4]*4 + matrix[0,5]*5 )/(np.sum(matrix[0]) - matrix[0,0])
    dmgold = (matrix[1,0]*1 + matrix[1,1]*0 + matrix[1,2]*1 + matrix[1,3]*2 + matrix[1,4]*3 + matrix[1,5]*4)/(np.sum(matrix[1]) - matrix[1,1])
    dmplat = (matrix[2,0]*2 + matrix[2,1]*1 + matrix[2,2]*0 + matrix[2,3]*1 + matrix[2,4]*2 + matrix[2,5]*3)/(np.sum(matrix[2]) - matrix[2,2])
    dmdia = (matrix[3,0]*3 + matrix[3,1]*2 + matrix[3,2]*1 + matrix[3,3]*0 + matrix[3,4]*1 + matrix[3,5]*2 )/(np.sum(matrix[3]) - matrix[3,3])
    dmma = (matrix[4,0]*4 + matrix[4,1]*3 + matrix[4,2]*2 + matrix[4,3]*1 + matrix[4,4]*0 + matrix[4,5]*1)/(np.sum(matrix[4]) - matrix[4,4])
    dmgm = (matrix[5,0]*5 + matrix[5,1]*4 + matrix[5,2]*3 + matrix[5,3]*2 + matrix[5,4]*1 + matrix[5,5]*0)/(np.sum(matrix[5]) - matrix[5,5])
    total_acc = ((dmsilv*np.sum(matrix[0])) + (dmgold*np.sum(matrix[1])) + (dmplat*np.sum(matrix[2])) + (dmdia*np.sum(matrix[3])) + (dmma*np.sum(matrix[4])) +  (dmgm*np.sum(matrix[5])))/sum(sum(matrix))
    return [dmsilv, dmgold, dmplat, dmdia, dmma, dmgm, total_acc]

def get_one_off_accuracy(matrix):
    #the accuracy for each class if the prediction is allowed to be one off.
    dmsilver = (matrix[0,0]*1 + matrix[0,1]*1)
    dmgold = (matrix[1,0]*1 + matrix[1,1]*1 + matrix[1,2]*1)
    dmplat = (matrix[2,1]*1 + matrix[2,2]*1 + matrix[2,3]*1)
    dmdia = (matrix[3,2]*1 + matrix[3,3]*1 + matrix[3,4]*1)
    dmmas = (matrix[4,3]*1 + matrix[4,4]*1 + matrix[4,5]*1)
    dmgm = ( matrix[5,4]*1 + matrix[5,5]*1)
    totalaccuracy = (dmsilver + dmgold + dmplat + dmdia + dmmas + dmgm)/sum(sum(matrix))
    accuracy = [dmsilver/(np.sum(matrix[0])), dmgold/(np.sum(matrix[1])), dmplat/(np.sum(matrix[2])), dmdia/(np.sum(matrix[3])), dmmas/(np.sum(matrix[4])), dmgm/(np.sum(matrix[5]))]
    return {"accuracyperleague" : accuracy, "total accuracy": totalaccuracy}

global model
enddict = {}
models = {'Case_I': model_I,'Case_II': model_II, 'Case_III': model_III,'Model_RF_rank': model_RF_rank, "Model_XG_rank": model_XG_rank, 'Model_RF_win': model_RF_win, "Model_XG_win": model_XG_win}

for name, model_used in models.items():
  model = model_used
  
  avg_acc_list = []
  avg_dist_list = []
  mis_dist_list = []
  one_off_acc_list = []
  avg_acc_win_list = []
  model_dict = {}
  print(name)
  for i in range(nfolds-1):
    if name in ['Case_I', 'Case_II', 'Case_III']:
      labels = ordinal_labels(name, folds[i][0], folds[i][1], folds[i][2], folds[i][3])
      ordinalaccuracy = ordinal_accuracy(folds[i][-1], labels)
      ordinalconfusion = ordinal_confusion_matrix(folds[i][-1], labels)
      avg_dist = get_average_distance(ordinalconfusion)[-1]
      avg_missclas = get_missclassified_distance(ordinalconfusion)[-1]
      avg_one_off = get_one_off_accuracy(ordinalconfusion)["total accuracy"]
      avg_acc_win =  get_win_accuracy(name, folds[i][0], folds[i][1], folds[i][2], folds[i][3], folds[i][-1], folds[i][-2])
      avg_acc_list.append(ordinalaccuracy)
      avg_dist_list.append(avg_dist)
      mis_dist_list.append(avg_missclas)
      one_off_acc_list.append(avg_one_off)
      avg_acc_win_list.append(avg_acc_win)
    elif name in ['Model_RF_rank', 'Model_XG_rank']:
      dataset = classification_dataset(folds[i][4], folds[i][3])
      accuracy = classification_accuracy(dataset, folds[i][-1])
      confusion = classification_confusion_matrix(dataset, folds[i][-1])
      avg_dist = get_average_distance(confusion)[-1]
      avg_missclas = get_missclassified_distance(confusion)[-1]
      avg_one_off = get_one_off_accuracy(confusion)["total accuracy"]
      avg_acc_list.append(accuracy)
      avg_dist_list.append(avg_dist)
      mis_dist_list.append(avg_missclas)
      one_off_acc_list.append(avg_one_off)
    else:
      dataset = classification_dataset(folds[i][4], folds[i][3])
      accuracy = classification_accuracy(dataset, folds[i][-2])
      avg_acc_win_list.append(accuracy)

  if name in ['Case_I', 'Case_II', 'Case_III']: 
    model_dict.update({
      'avg_acc': avg_acc_list,
      'avg_dist': avg_dist_list,
      'mis_dist': mis_dist_list,
      'one_off_acc': one_off_acc_list,
      'win_acc': avg_acc_win_list  
    })
    enddict[name]= model_dict
  elif name in ['Model_RF_rank', 'Model_XG_rank']:
    model_dict.update({
      'avg_acc': avg_acc_list,
      'avg_dist': avg_dist_list,
      'mis_dist': mis_dist_list,
      'one_off_acc': one_off_acc_list 
    })
    enddict[name]= model_dict
  else:
    model_dict.update({
      'win_acc': avg_acc_win_list  
    })
    enddict[name]= model_dict
  
global avg_acc_dict, avg_dist_dict, avg_mis_dist_dict, avg_one_off_acc_dict, avg_acc_win_dict
avg_acc_dict = {}
avg_dist_dict = {}
avg_mis_dist_dict = {}
avg_one_off_acc_dict = {}
avg_acc_win_dict = {}

for name, values in enddict.items():
  try:
    avg_acc_dict[name] = values['avg_acc']
    avg_dist_dict[name] = values['avg_dist']
    avg_mis_dist_dict[name] = values['mis_dist']
    avg_one_off_acc_dict[name] = values['one_off_acc']
  except:
    pass

  try:
    avg_acc_win_dict[name] = values['win_acc']
  except:
    pass

from scipy.stats import wilcoxon
from itertools import combinations

def calculate_significance(inputDict):
  result_list = list(map(dict, combinations(inputDict.items(), 2)))
  returnDict = dict()
  for combination in result_list:
    name = str(list(combination.keys()))
    returnDict[name] = wilcoxon(list(combination.values())[0], list(combination.values())[1])
  return returnDict

sign_avg_acc = calculate_significance(avg_acc_dict)
sign_avg_dist = calculate_significance(avg_dist_dict)
sign_avg_mis_dist = calculate_significance(avg_mis_dist_dict)
sign_avg_one_off = calculate_significance(avg_one_off_acc_dict)
sign_avg_win = calculate_significance(avg_acc_win_dict)

def calculate_significance(testDict):
  returnDict = dict()
  for key, value in testDict.items():
    for key2, value2 in testDict.items():
      if key != key2:
        returnDict[key+key2] = wilcoxon(value, value2).pvalue
  return returnDict

sign_avg_acc = calculate_significance(avg_acc_dict)
sign_avg_dist = calculate_significance(avg_dist_dict)
sign_avg_mis_dist = calculate_significance(avg_mis_dist_dict)
sign_avg_one_off = calculate_significance(avg_one_off_acc_dict)
sign_avg_win = calculate_significance(avg_acc_win_dict)

Final_dict = {}
Final_dict.update({
    "enddict": enddict,
    "sign_avg_acc": sign_avg_acc,
    "sign_avg_dist": sign_avg_dist,
    "sign_avg_mis_dist": sign_avg_mis_dist,
    "sign_avg_one_off": sign_avg_one_off,
    "sign_acc_win": sign_avg_win
})

import pickle
with open(r"Other\significance.pkl", "wb") as handle:
        pickle.dump(Final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


