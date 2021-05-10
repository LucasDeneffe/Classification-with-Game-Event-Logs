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
    global yrankcorrect, ywin, Xp1, Xp2, time1, time2, unique
    yrank = np.load(filepath + "/yrank.npy")
    yrankcorrect = np.asarray([x - 2 for x in yrank])
    ywin = np.load(filepath + "/ywin.npy")
    ywin = np.asarray(ywin)
    Xp1 = np.load(filepath + "/Xp1.npy")
    Xp2 = np.load(filepath + "/Xp2.npy")

    with open(filepath + "/unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
    
    with open(filepath + "/time1.pkl", 'rb') as handle:
        time1 = pickle.load(handle)
    
    with open(filepath + "/time2.pkl", 'rb') as handle:
        time2 = pickle.load(handle)

def split_log():
    #sequence of player 1
    global Xp1_train, Xp1_val, Xp1_test, ywin_train, ywin_val, ywin_test, Xp1
    Xp1_train, Xp1_val, ywin_train, ywin_val = train_test_split(Xp1, ywin, test_size = 0.3 , random_state = 15)
    Xp1_val, Xp1_test, ywin_val, ywin_test = train_test_split(Xp1_val, ywin_val, test_size = 1/3 , random_state = 15)
    gc.collect()
    del Xp1
    
    #sequenceof player 2
    global Xp2_train, Xp2_val, Xp2_test, Xp2, yrank_train, yrank_test, yrank_val
    Xp2_train, Xp2_val, yrank_train, yrank_val = train_test_split(Xp2, yrankcorrect, test_size = 0.3, random_state = 15)
    Xp2_val, Xp2_test, yrank_val, yrank_test = train_test_split(Xp2_val, yrank_val, test_size = 1/3 , random_state = 15)
    del Xp2
    #gc.collect()

    global time1_test, time2_test, time1, time2
    time1_train, time1_val, time2_train, time2_val = train_test_split(time1, time2, test_size = 0.3, random_state = 15)
    time1_val, time1_test, time2_val, time2_test = train_test_split(time1_val, time2_val, test_size = 1/3 , random_state = 15)
    del time1, time2

def get_model_summary():
    global model
    model.summary()

def create_model():
    from keras.layers import Input, Dropout, Embedding, Flatten, Dense, Concatenate, LSTM, Bidirectional, Masking
    from keras.models import Model
    import tensorflow as tf

    #the two sequences
    dictionary_input1 = Input((Xp1_train.shape[1], ), name = "Player1InputSequence")
    dictionary_input2 = Input((Xp2_train.shape[1], ), name = "Player2InputSequence")
    
    #sequences
    embedding1 = Embedding(len(unique)+1, 8, input_length=Xp1_train.shape[1], mask_zero= True, name = "Player1Embedding")(dictionary_input1)
    embedding2 = Embedding(len(unique)+1, 8, input_length=Xp2_train.shape[1], mask_zero= True, name = "Player2Embedding")(dictionary_input2)
    embedding1 = LSTM(128, name = "Player1LSTM")(embedding1)
    embedding1 = Dropout(0.1, name = "Player1Dropout")(embedding1)
    embedding2 = LSTM(128, name = "Player2LSTM")(embedding2)
    embedding2 = Dropout(0.1, name = "Player2Dropout")(embedding2)
      
    
    #Concatenate all
    combined_data = Concatenate(name = "Concatenate")([embedding1, embedding2])

    dense1 = Dense(32, activation = "relu", name = 'DenseRank')(combined_data)
    dropout1 = Dropout(0.1, name = "DropoutRank")(dense1)
    output1 = coral.CoralOrdinal(num_classes = 6, name = "RankClassifier")(dropout1) # Ordinal variable has 7 labels, 0 through 6.
      
    dense2 = Dense(32, activation = "relu", name = 'DenseWin')(combined_data)
    dropout2 = Dropout(0.1, name = "DropoutWin")(dense2)
    output2 = Dense(1, activation = "sigmoid", name = "WinClassifier")(dropout2)
      
    global model
      
    model = Model(inputs=[dictionary_input1, dictionary_input2], outputs=[output1, output2])

    model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
                loss = {"RankClassifier": coral.OrdinalCrossEntropy(num_classes = 6), "WinClassifier": 'binary_crossentropy'},
                metrics = {"RankClassifier": [coral.MeanAbsoluteErrorLabels()], "WinClassifier": 'accuracy'})

def fit_model(batch_size):
    global model    
    history = model.fit([Xp1_train, Xp2_train],
                        [yrank_train, ywin_train],
                        epochs=200,
                        batch_size = batch_size,
                        validation_data = ([Xp1_val, Xp2_val],[yrank_val, ywin_val]),
                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 25, restore_best_weights = True)])
    return history

def save_model():
    model.save(input("Location to save model: ") + "/CASE1MODEL")

def load_model():
    import keras
    model = keras.models.load_model(input("Location to load model: ") + "/CASE1MODEL", custom_objects = {"MeanAbsoluteErrorLabels()" : coral.MeanAbsoluteErrorLabels()})

def ordinal_labels():
    from scipy import special
    ordinal_logits = model.predict([Xp1_test, Xp2_test])
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
model.evaluate([Xp1_train, Xp2_train], [yrank_train, ywin_train])
"Model results on validation data."
model.evaluate([Xp1_val, Xp2_val], [yrank_val, ywin_val])
"Model results on test data."
model.evaluate([Xp1_test, Xp2_test], [yrank_test, ywin_test])

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
