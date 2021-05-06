#Module imports
import pandas as pd
import sklearn
import numpy as np
import pickle
import gc
from sklearn.model_selection import train_test_split

def import_data():     
    global yrankcorrect, ywin, X, Xp1, Xp2,PF1, PF2, time1, time2, RFlog, unique
    
    yrank = np.load(r"C:/Users/Lucas/Documents/Arrays//yrank.npy")
    yrankcorrect = yrank
    ywin = np.load(r"C:/Users/Lucas/Documents/Arrays//ywin.npy")
    ywin = np.asarray(ywin)
    X = np.load(r"C:/Users/Lucas/Documents/Arrays//X.npy")
    Xp1 = np.load(r"C:/Users/Lucas/Documents/Arrays//Xp1.npy")
    Xp2 = np.load(r"C:/Users/Lucas/Documents/Arrays//Xp2.npy")
    PF1=  np.load(r"C:/Users/Lucas/Documents/Arrays//PF1.npy")
    PF2=  np.load(r"C:/Users/Lucas/Documents/Arrays//PF2.npy")

    with open(r"C:/Users/Lucas/Documents/Arrays//unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
    
    with open(r"C:/Users/Lucas/Documents/Arrays//time1.pkl", 'rb') as handle:
        time1 = pickle.load(handle)
    
    with open(r"C:/Users/Lucas/Documents/Arrays//time2.pkl", 'rb') as handle:
        time2 = pickle.load(handle)

#import Random forest model
with open(
    r"C:\Users\lucas\OneDrive\Thesis\Datasets\Starcraft data\StarcraftParserData\Arrays/RF_unscaled_rank.pkl",
    "rb",
) as handle:
    rf_rank = pickle.load(handle)
with open(
    r"C:\Users\lucas\OneDrive\Thesis\Datasets\Starcraft data\StarcraftParserData\Arrays/RF_unscaled_win.pkl",
    "rb",
) as handle:
    rf_win = pickle.load(handle)

#import Xgboost model
from xgboost import XGBClassifier
XG_rank = XGBClassifier()
XG_rank.load_model(r"C:\Users\lucas\OneDrive\Thesis\Datasets\Starcraft data\StarcraftParserData\Arrays\XG_unscaled_rank.json")
XG_win = XGBClassifier()
XG_win.load_model(r"C:\Users\lucas\OneDrive\Thesis\Datasets\Starcraft data\StarcraftParserData\Arrays\XG_unscaled_win.json")

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

    #time
    global time1_train, time1_val, time1_test, time2_train, time2_val, time2_test
    time1_train, time1_val, time2_train, time2_val = train_test_split(time1, time2, test_size = 0.3, random_state = 15)
    time1_val, time1_test, time2_val, time2_test = train_test_split(time1_val, time2_val, test_size = 1/3, random_state = 15)

def padding1(array, timepoint, variablename):
    """
    :param array: numpy array
    pad the features
    """
    variabledic = {"X": X_test, "PF": PF_test}
    newarray = np.copy(array)
    for idx, sample in enumerate(variabledic[variablename]):
        latestindex = 0
        length_of_replay = -1
        for index, row in enumerate(sample):
            row = np.trim_zeros(row)
            if row.size == 0:
                length_of_replay = index
                break
        if length_of_replay == -1:
            length_of_replay = variabledic[variablename].shape[1]
        for index, row in enumerate(sample):
            if index <= timepoint*length_of_replay/100:
                latestindex = index
            else:
                newarray[idx, latestindex:] = np.zeros((variabledic[variablename].shape[1] - latestindex, variabledic[variablename].shape[2]))
                break
    return newarray

#Trim data so that sequences and features end at the same point in time
def convert_data(Xshort, PFshort):
    PF_good = np.swapaxes(PFshort, 1, 2)
    PF_avg = np.apply_along_axis(return_mean, 2, PF_good)
    last_features = []
    for element in Xshort:
        trimmed = trim_zeros_2D(element, axis=1)
        last_features.append(trimmed[-1])
    test = np.asarray(last_features)
    print(test.shape)
    stack = np.hstack((test, PF_avg))
    print(stack.shape)

    return stack

def return_mean(a):
    trimmed_array = np.trim_zeros(a, trim="b")
    if trimmed_array.size == 0:
        trimmed_array = np.array([0])
    return np.mean(trimmed_array)

# Trim zeros from a 2 dimensional array
def trim_zeros_2D(array, axis=1):
    mask = ~(array==0).all(axis=axis)
    inv_mask = mask[::-1]
    start_idx = np.argmax(mask == True)
    end_idx = len(inv_mask) - np.argmax(inv_mask == True)
    if axis:
        return array[start_idx:end_idx,:]
    else:
        return array[:, start_idx:end_idx]

#### MAIN ###
### FILL in right model
win_model = 
rank_model = 
import_data()
combine_PF()
split_log()

from scipy import special
import gc
accuracylstm = []
oneoffaccuracy = []
accuracywin = []
for i in range(0,101, 1):
    #grow the time by 10 seconds each time.
    X_testshorter = (padding1(X_test, i, "X"))
    PF_testshorter = (padding1(PF_test, i, "PF")) #PF is per minute, while X is per 10 seconds
    
    data = convert_data(X_testshorter, PF_testshorter)
    print(data.shape)
    
    print("Starting Predictions")
    rank_predictions = rank_model.predict(data)
    win_predictions = win_model.predict(data)
    print("Ended Predictions")
    #predictions = model.predict([X_testshorter, PF_testshorter])
    accuracywin.append(SVM_win.score(data, ywin_test))
    #accuracywin.append(model.evaluate([X_testshorter, PF1_testshorter], [yrank_test, ywin_test])[4])
    labels = rank_predictions
    labelsplus = labels + 1
    labelsmin = labels - 1
    labelsmin = labels - 1
    accuracylstm.append(np.mean(labels == yrank_test))
    oneoffaccuracy.append(np.mean((labels == yrank_test) | (labelsplus == yrank_test) | (labelsmin == yrank_test)))
    
    if i%5 == 0: 
        print(f"{i}% completed. Accuracy: {accuracylstm[-1]} -- {oneoffaccuracy[-1]} -- {accuracywin[-1]}")
        gc.collect()
    #accuracyclf.append(scoresclf)
    #accuracylr.append(scoreslr)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(30,10))
ax1.plot(accuracy)
ax2.plot(oneoffaccuracy)
ax3.plot(accuracywin[:101])
#ax.plot(accuracyclf, label = 'adaboost')
#ax.plot(accuracylr, label = 'log reg')
#a=ax.get_xticks()
#b = ax.get_yticks()
#plot_labels = np.around(a*/100)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xlabel("Percentage of game completed")
ax1.set_ylabel("Accuracy in %")
ax1.title.set_text('Rank Regular Accuracy')
ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_xlabel("Percentage of game completed")
ax2.set_ylabel("Accuracy in %")
ax2.title.set_text('Rank One-Off Accuracy')
ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
ax3.set_xlabel("Percentage of game completed")
ax3.set_ylabel("Accuracy in %")
ax3.title.set_text('Win Accuracy')

#ax.set_xticklabels(plot_labels.astype(int))
#ax.set_yticklabels(np.around(b*100).astype(int))
ax1.legend()
ax2.legend()
ax3.legend()

#SLA DE VOLGENDE WAARDEN ERGENS OP VOOR IN LATEX ALS COORDINATEN TE GEBRUIKEN. BETER OM LATEX GRAPH TE GEBRUIKEN IPV AAFBEELDING VOOR SCHERPTE?
for i, value in enumerate(oneoffaccuracy):
    print(f"({i},{value})")

for i, value in enumerate(accuracywin):
    print(f"({i},{value})")

for i, value in enumerate(accuracylstm):
    print(f"({i},{value})")




