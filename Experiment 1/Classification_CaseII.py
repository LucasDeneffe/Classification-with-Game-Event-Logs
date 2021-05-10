pip install git+https://github.com/ck37/coral-ordinal/
pip install pickle

#Install libraries and import dependencies
pip install git+https://github.com/ck37/coral-ordinal/
pip install pickle

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
    global yrankcorrect, ywin, X, PF1, PF2, unique
    yrank = np.load(filepath + "/yrank.npy")
    yrankcorrect = np.asarray([x - 2 for x in yrank])
    ywin = np.load(filepath + "/ywin.npy")
    ywin = np.asarray(ywin)
    X = np.load(filepath + "/X.npy")
    PF1 = np.load(filepath + "/PF1.npy")
    PF2 = np.load(filepath + "/PF1.npy")
    
    with open(filepath + "/unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
    
def combine_PF():
    global PF
    print(PF1.shape)
    print(PF2.shape)
    PF = np.dstack((PF1, PF2))
    gc.collect()
    
def split_log():
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

def create_model():
    from keras.layers import Input, Dropout, Embedding, Flatten, Dense, Concatenate, LSTM, Bidirectional, Masking
    from keras.models import Model
    import tensorflow as tf

    #the features from PlayerStatsEvent
    features_input = Input((X_train.shape[1],X_train.shape[2]), name = "PlayerStatsInput")
    #the features from the paper
    PF_input = Input((PF_train.shape[1],PF_train.shape[2]), name = "APMandSQInput")

    #features from PlayerStatsEvent
    features = Masking(mask_value = 0, name = "PlayerStatsMasking")(features_input)
    features = LSTM(64, name = "PlayerStatsLSTM")(features)
    features = Dropout(0.3, name = "PlayerStatsDropout")(features)
      
    #paper features
    PFlayer = Masking(mask_value = 0, name = "APMandSQMasking")(PF1_input)
    PFlayer = LSTM(64, name = "APMandSQLSTM")(PFlayer1)
    PFlayer = Dropout(0.3, name = "APMandSQDropout")(PFlayer)
      
    
    #Concatenate all
    combined_data = Concatenate(name = "Concatenate")([features, PFlayer])

    dense1 = Dense(32, activation = "relu", name = 'DenseRank')(combined_data)
    dropout1 = Dropout(0.3, name = "DropoutRank")(dense1)
    output1 = coral.CoralOrdinal(num_classes = 6, name = "RankClassifier")(dropout1) # Ordinal variable has 7 labels, 0 through 6.
      
    dense2 = Dense(32, activation = "relu", name = 'DenseWin')(combined_data)
    dropout2 = Dropout(0.3, name = "DropoutWin")(dense2)
    output2 = Dense(1, activation = "sigmoid", name = "WinClassifier")(dropout2)
      
    global model
      
    model = Model(inputs=[dictionary_input1, dictionary_input2], outputs=[output1, output2])

    model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                loss = {"RankClassifier": coral.OrdinalCrossEntropy(num_classes = 6), "WinClassifier": 'binary_crossentropy'},
                metrics = {"RankClassifier": [coral.MeanAbsoluteErrorLabels()], "WinClassifier": 'accuracy'})

def fit_model(batch_size):
    global model    
    history = model.fit(
        [X_train, PF_train],
        [yrank_train, ywin_train],
        epochs=200,
        shuffle = True,
        batch_size = batch_size,
        validation_data = ([X_val, PF_val], [yrank_val, ywin_val]),
        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)])

    return history

def save_model():
    model.save(input("Location to save model: ") + "/CASE2MODEL")

def load_model():
    import keras
    model = keras.models.load_model(input("Filepath to model: "), custom_objects = {"MeanAbsoluteErrorLabels()" : coral.MeanAbsoluteErrorLabels()})

def ordinal_labels():
    from scipy import special
    ordinal_logits = model.predict([X_test, PF_test])
    cum_probs = pd.DataFrame(ordinal_logits[0]).apply(special.expit)
    labels = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
    return labels

def ordinal_accuracy(labels = ordinal_labels()):
    return(np.mean(labels == yrank_test))

def ordinal_confusion_matrix(labels = ordinal_labels()):
    return tf.math.confusion_matrix(yrank_test, labels, num_classes = 6, dtype=tf.dtypes.int32).numpy()

def get_average_distance(matrix = ordinal_confusion_matrix()):
    #Calculate the average distance the predicted label was from the correct label. Under 1 means the label was on average mostly correct.
    dsilver = (matrix[0,0]*0 + matrix[0,1]*1 + matrix[0,2]*2 + matrix[0,3]*3 + matrix[0,4]*4 + matrix[0,5]*5)/np.sum(matrix[0])
    dgold = (matrix[1,0]*1 + matrix[1,1]*0 + matrix[1,2]*1 + matrix[1,3]*2 + matrix[1,4]*3 + matrix[1,5]*4)/np.sum(matrix[1])
    dplat = (matrix[2,0]*2 + matrix[2,1]*1 + matrix[2,2]*0 + matrix[2,3]*1 + matrix[2,4]*2 + matrix[2,5]*3)/np.sum(matrix[2])
    ddia = (matrix[3,0]*3 + matrix[3,1]*2 + matrix[3,2]*1 + matrix[3,3]*0 + matrix[3,4]*1 + matrix[3,5]*2)/np.sum(matrix[3])
    dma = (matrix[4,0]*4 + matrix[4,1]*3 + matrix[4,2]*2 + matrix[4,3]*1 + matrix[4,4]*0 + matrix[4,5]*1)/np.sum(matrix[4])
    dgm = (matrix[5,0]*5 + matrix[5,1]*4 + matrix[5,2]*3 + matrix[5,3]*2 + matrix[5,4]*1 + matrix[5,5]*0)/np.sum(matrix[5])

    return [dsilver, dgold, dplat, ddia, dma, dgm]
     

def get_missclassified_distance(matrix = ordinal_confusion_matrix()):
    #Calculate the average distance of the missclassified instances. Closer to one is better.
    dmsilv = (matrix[0,0]*0 + matrix[0,1]*1 + matrix[0,2]*2 + matrix[0,3]*3 + matrix[0,4]*4 + matrix[0,5]*5 )/(np.sum(matrix[0]) - matrix[0,0])
    dmgold = (matrix[1,0]*1 + matrix[1,1]*0 + matrix[1,2]*1 + matrix[1,3]*2 + matrix[1,4]*3 + matrix[1,5]*4)/(np.sum(matrix[1]) - matrix[1,1])
    dmplat = (matrix[2,0]*2 + matrix[2,1]*1 + matrix[2,2]*0 + matrix[2,3]*1 + matrix[2,4]*2 + matrix[2,5]*3)/(np.sum(matrix[2]) - matrix[2,2])
    dmdia = (matrix[3,0]*3 + matrix[3,1]*2 + matrix[3,2]*1 + matrix[3,3]*0 + matrix[3,4]*1 + matrix[3,5]*2 )/(np.sum(matrix[3]) - matrix[3,3])
    dmma = (matrix[4,0]*4 + matrix[4,1]*3 + matrix[4,2]*2 + matrix[4,3]*1 + matrix[4,4]*0 + matrix[4,5]*1)/(np.sum(matrix[4]) - matrix[4,4])
    dmgm = (matrix[5,0]*5 + matrix[5,1]*4 + matrix[5,2]*3 + matrix[5,3]*2 + matrix[5,4]*1 + matrix[5,5]*0)/(np.sum(matrix[5]) - matrix[5,5])
    
    return [dmsilv, dmgold, dmplat, dmdia, dmma, dmgm]

def get_one_off_accuracy(matrix = ordinal_confusion_matrix()):
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

#Main
import_data()
split_log()
create_model()
get_model_summary()
fit_model(64)
"Model results on train data."
model.evaluate([X_train, PF_train], [yrank_train, ywin_train])
"Model results on validation data."
model.evaluate([X_val, PF_val], [yrank_val, ywin_val])
"Model results on test data."
model.evaluate([X_test, PF_test], [yrank_test, ywin_test])

labels = ordinal_labels()
ordinalaccuracy = ordinal_accuracy(labels)
print(f"The rank accuracy is: {round(ordinalaccuracy*100,4)}%")
ordinalconfusion = ordinal_confusion_matrix(labels)
print("Confusion Matrix:")
print(ordinalconfusion)
print("The average classification distance for rank is: ")
print(get_average_distance(ordinalconfusion))
print("The average misclassification distance for rank is: ")
print(get_missclassified_distance(ordinalconfusion))
print("The one-off accuracy for rank is: ")
print(get_one_off_accuracy(ordinalconfusion))
