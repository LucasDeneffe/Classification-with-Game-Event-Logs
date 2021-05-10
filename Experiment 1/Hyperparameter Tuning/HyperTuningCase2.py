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

from tensorboard.plugins.hparams import api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64, 128, 256, 512, 1024]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.3, 0.5, 0.7]))
HP_OPTIMIZER = hp.HParam('optimizer_lr', hp.Discrete([0.01, 0.001]))

METRIC_ACCURACY = 'accuracy'
METRIC_MAEL = "[coral.MeanAbsoluteErrorLabels()]"

with tf.summary.create_file_writer(filepath + "/hparam_tuning2/").as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'), hp.Metric(METRIC_MAEL, display_name = "MAEL")]
  )

def train_test_model(hparams):
  from keras.layers import Input, Dropout, Embedding, Flatten, Dense, Concatenate, LSTM, Bidirectional, Masking
  from keras.models import Model
  import tensorflow as tf

  #the features from PlayerStatsEvent
  features_input = Input((X_train.shape[1],X_train.shape[2]), name = "PlayerStatsInput")
  #the features from the paper
  PF_input = Input((PF_train.shape[1],PF_train.shape[2]), name = "APMandSQInput")

  #features from PlayerStatsEvent
  features = Masking(mask_value = 0, name = "PlayerStatsMasking")(features_input)
  features = LSTM(hparams[HP_NUM_UNITS], name = "PlayerStatsLSTM")(features)
  features = Dropout(hparams[HP_DROPOUT], name = "PlayerStatsDropout")(features)
    
  #paper features
  PFlayer = Masking(mask_value = 0, name = "APMandSQMasking")(PF_input)
  PFlayer = LSTM(hparams[HP_NUM_UNITS], name = "APMandSQLSTM")(PFlayer)
  PFlayer = Dropout(hparams[HP_DROPOUT], name = "APMandSQDropout")(PFlayer)

  #Concatenate all
  combined_data = Concatenate(name = "Concatenate")([features, PFlayer])
  dense1 = Dense(32, activation = "relu", name = 'DenseRank')(combined_data)
  dropout1 = Dropout(hparams[HP_DROPOUT], name = "DropoutRank")(dense1)
  output1 = coral.CoralOrdinal(num_classes = 6, name = "RankClassifier")(dropout1)
  
  dense2 = Dense(32, activation = "relu", name = 'DenseWin')(combined_data)
  dropout2 = Dropout(hparams[HP_DROPOUT], name = "DropoutWin")(dense2)
  output2 = Dense(1, activation = "sigmoid", name = "WinClassifier")(dropout2)
    
  global model
    
  model = Model(inputs=[features_input, PF_input], outputs= [output1, output2])

  model.compile(optimizer = tf.keras.optimizers.Adam(hparams[HP_OPTIMIZER]),
              loss = {"RankClassifier": coral.OrdinalCrossEntropy(num_classes = 6), "WinClassifier": 'binary_crossentropy'},
              metrics = {"RankClassifier": [coral.MeanAbsoluteErrorLabels()], "WinClassifier": 'accuracy'})

  print(model.summary())

   model.fit([X_train, PF_train],
             [yrank_train,ywin_train],
             epochs=200,
             batch_size = 64,
             shuffle = True,
             validation_data = ([X_val, PF_val], [yrank_val, ywin_val]),
             callbacks = [tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)])


  print("test set accuracy - not used for validation, just to compare")
  model.evaluate([X_test, PF_test], [yrank_test, ywin_test])
  gc.collect()
  print("validation set accuracy")
  metrics = model.evaluate([X_val, PF_val], [yrank_val, ywin_val])
  return metrics

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy[4], step=1)
    tf.summary.scalar(METRIC_MAEL, accuracy[3], step=1)

filepath = input("Give the filepath where the arrays are stored: ")
filepath2 = input("Give the filepath where the hypertuning logs should be stored: ")
session_num = 0
import_data()
combine_PF()
split_log()
print("finished split_log")
for units in HP_NUM_UNITS.domain.values:
  for dropout_rate in HP_DROPOUT.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
          HP_NUM_UNITS: units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(filepath + "/hparam_tuning2/" + run_name, hparams)
        session_num += 1
        
