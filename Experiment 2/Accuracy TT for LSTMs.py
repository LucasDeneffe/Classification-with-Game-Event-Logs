#Install libraries and import dependencies
import pandas as pd
import pickle 
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import regularizers
import gc
import coral_ordinal as coral #Allows to use ordinal variables in Tensorflow
#make sure the latest git version is installed in order for it to work
#https://github.com/ck37/coral-ordinal/

def import_data():    
    global yrankcorrect, ywin, Xp1, Xp2, X, PF1, PF2, unique
    yrank = np.load(filepath + "/yrank.npy")
    yrankcorrect = np.asarray([x - 2 for x in yrank])
    ywin = np.load(filepath + "/ywin.npy")
    ywin = np.asarray(ywin)
    Xp1 =  np.load(filepath + "/Xp1.npy")
    Xp2 =  np.load(filepath + "/Xp2.npy")
    X = np.load(filepath + "/X.npy")
    PF1 = np.load(filepath + "/PF1.npy")
    PF2 = np.load(filepath + "/PF1.npy")
    
    with open(filepath + "/unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
        
    with open(filepath + "/time1.pkl", 'rb') as handle:
        time1 = pickle.load(handle)
    
    with open(filepath + "/time2.pkl", 'rb') as handle:
        time2 = pickle.load(handle)
    
def combine_PF():
    global PF
    print(PF1.shape)
    print(PF2.shape)
    PF = np.dstack((PF1, PF2))
    gc.collect()
    
def split_log():
    #sequence of player 1
    global Xp1_train, Xp1_val, Xp1_test, Xp1
    Xp1_train, Xp1_val = train_test_split(Xp1, test_size = 0.3 , random_state = 15)
    Xp1_val, Xp1_test = train_test_split(Xp1_val, test_size = 1/3 , random_state = 15)
    gc.collect()
    del Xp1

    #sequence of player 2
    global Xp2_train, Xp2_val, Xp2_test, Xp2
    Xp2_train, Xp2_val = train_test_split(Xp2, test_size = 0.3, random_state = 15)
    Xp2_val, Xp2_test = train_test_split(Xp2_val, test_size = 1/3 , random_state = 15)
    del Xp2
    gc.collect()
    
    #Features from PlayerStatsEvent
    global X_train, X_val, X_test, yrank_train, yrank_val, yrank_test, X
    X_train, X_val, yrank_train, yrank_val = train_test_split(X, yrankcorrect, test_size = 0.3, random_state = 15)
    X_val, X_test, yrank_val, yrank_test = train_test_split(X_val, yrank_val, test_size = 1/3, random_state = 15)
    del X
    gc.collect()
    
    #features based on paper
    global PF_train, PF_val, PF_test, ywin_train, ywin_test, ywin_val, PF
    PF_train, PF_val, ywin_train, ywin_val  = train_test_split(PF, ywin, test_size = 0.3, random_state = 15)
    PF_val, PF_test, ywin_val, ywin_test = train_test_split(PF_val, ywin_val, test_size = 1/3 , random_state = 15)
    del PF
    gc.collect()

def get_model_summary():
    global model
    model.summary()

def load_model(modelname):
    import keras
    global model
    model = keras.models.load_model(input(f"Filepath to model {modelname}: "), custom_objects = {"MeanAbsoluteErrorLabels()" : coral.MeanAbsoluteErrorLabels()})

def padding1(array, timepoint, variablename):
    """
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

def padding2(array, timepoint, variablename):
    """
    Only pad the sequences up to a certain timepoint
    """
    newarray = np.copy(array)
    variabledic = {"xp1": Xp1_test, "xp2": Xp2_test}
    timedic = {"xp1": time1_test, "xp2" : time2_test}
    for idx, sample in enumerate(variabledic[variablename]):
        latestindex = 0
        try: 
            length_of_replay = np.max(np.unique(timedic[variablename][idx]))
        except Exception as e: continue
        for index, element in enumerate(variabledic[variablename][idx]): 
            try:
                if timedic[variablename][idx][index] <= timepoint*length_of_replay/100:
                    latestindex = index
                else: 
                    newarray[idx, latestindex:] = np.zeros(variabledic[variablename].shape[1]-latestindex)
                    break
            except: 
                newarray[idx, latestindex:] = np.zeros(variabledic[variablename].shape[1]-latestindex)
                break

    return newarray
        
def accuracy_through_time(modelname):
    import gc
    accuracyrank = []
    oneoffaccuracy = []
    accuracywin = []
    load_model(modelname)
    for i in range(0,101, 1):
        if modelname == "Case I":
            Xp1_testshorter = (padding2(Xp1_test, i, "xp1"))
            Xp2_testshorter = (padding2(Xp2_test, i, "xp2"))
            predictions = model.predict([Xp1_testshorter, Xp2_testshorter])
            accuracywin.append(model.evaluate([Xp1_testshorter, Xp2_testshorter], [yrank_test, ywin_test])[4])
            from scipy import special
    
        if modelname == "Case II" :
            X_testshorter = (padding1(X_test, i, "X"))
            PF_testshorter = (padding1(PF_test, i, "PF"))
            predictions = model.predict([X_testshorter, PF_testshorter])
            accuracywin.append(model.evaluate([X_testshorter, PF_testshorter], [yrank_test, ywin_test])[4])
 

        if modelname == "Case III" :
            Xp1_testshorter = (padding2(Xp1_test, i, "xp1"))
            Xp2_testshorter = (padding2(Xp2_test, i, "xp2"))
            X_testshorter = (padding1(X_test, i, "X"))
            PF_testshorter = (padding1(PF_test, i, "PF"))
            predictions = model.predict([Xp1_testshorter, Xp2_testshorter, X_testshorter, PF_testshorter])
            accuracywin.append(model.evaluate([Xp1_testshorter, Xp2_testshorter, X_testshorter, PF_testshorter], [yrank_test, ywin_test])[4])

        from scipy import special
        cum_probs = pd.DataFrame(predictions[0]).apply(special.expit)
        labels = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
        labelsplus = labels + 1
        labelsmin = labels - 1
        labelsmin = labels - 1
        accuracyrank.append(np.mean(labels == yrank_test))
        oneoffaccuracy.append(np.mean((labels == yrank_test) | (labelsplus == yrank_test) | (labelsmin == yrank_test)))      

        if i%5 == 0: 
            print(f"{i}% completed. Accuracy: {accuracylstm[-1]} -- {oneoffaccuracy[-1]} -- {accuracywin[-1]}")
            gc.collect()
    return accuracyrank, oneoffaccuracy, accuracywin

def create_graph(accuracyrank, oneoffaccuracy, accuracywin, modelname):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(30,10))
    ax1.plot(accuracyrank, label = modelname)
    ax2.plot(oneoffaccuracy, label = modelname)
    ax3.plot(accuracywin[:101], label = modelname)
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

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()
    
#Main
filepath = input("Give the filepath where the arrays are stored: ")
import_data()
combine_pf()
split_log()

accuracyrank1, oneoffaccuracy1, accuracywin1 = accuracy_through_time("Case I")
create_graph(accuracyrank1, oneoffaccuracy1, accuracywin1, "Case I")

accuracyrank2, oneoffaccuracy2, accuracywin2 = accuracy_through_time("Case II")
create_graph(accuracyrank2, oneoffaccuracy2, accuracywin2, "Case II")

accuracyrank3, oneoffaccuracy3, accuracywin3 = accuracy_through_time("Case III")
create_graph(accuracyrank3, oneoffaccuracy3, accuracywin3, "Case III")

