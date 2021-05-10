#Install libraries and import dependencies
pip install git+https://github.com/ck37/coral-ordinal/
pip install pickle
pip install alibi

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
    
    with open(filepath + "/unique.pkl", 'rb') as handle:
        unique = pickle.load(handle)
    
def split_log():
    #sequence of player 1
    global Xp1_train, Xp1_val, Xp1_test, Xp1
    Xp1_train, Xp1_val = train_test_split(Xp1 test_size = 0.3 , random_state = 15)
    Xp1_val, Xp1_test = train_test_split(Xp1_val, test_size = 1/3 , random_state = 15)
    gc.collect()
    del Xp1

    #sequence of player 2
    global Xp2_train, Xp2_val, Xp2_test, Xp2
    Xp2_train, Xp2_val = train_test_split(Xp2, test_size = 0.3, random_state = 15)
    Xp2_val, Xp2_test = train_test_split(Xp2_val, test_size = 1/3 , random_state = 15)
    del Xp2
    gc.collect()
    
def get_model_summary():
    global model
    model.summary()

def save_model(model):
    model.save(Input("Give filepath to store model: ")

def train_model(label):
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
  
  if label == "rank":
    dense1 = Dense(32, activation = "relu", name = 'DenseRank')(combined_data)
    dropout1 = Dropout(0.1, name = "DropoutRank")(dense1)
    output = coral.CoralOrdinal(num_classes = 6, name = "RankClassifier")(dropout1) # Ordinal variable has 7 labels, 0 through 6.

  else:  
    dense2 = Dense(32, activation = "relu", name = 'DenseWin')(combined_data)
    dropout2 = Dropout(0.1, name = "DropoutWin")(dense2)
    output = Dense(1, activation = "sigmoid", name = "WinClassifier")(dropout2)
    
  global model
    
  model = Model(inputs=[dictionary_input1, dictionary_input2], outputs= output)

  loss =  coral.OrdinalCrossEntropy(num_classes = 6) if label == "rank" else 'binary_crossentropy'
  metrics = [coral.MeanAbsoluteErrorLabels()] if label == "rank" else 'accuracy'
  #ordinal 
  model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
              loss = loss,
              #loss = 'binary_crossentropy',
              metrics = metrics)
              #metrics = 'accuracy')

  print(model.summary())
  if label == "rank":
    model.fit([Xp1_train, Xp2_train], yrank_train, epochs=200, batch_size = 64, shuffle = True, validation_data = ([Xp1_val, Xp2_val], yrank_val), callbacks = [tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)])
  else:
    model.fit([Xp1_train, Xp2_train], ywin_train, epochs=200, batch_size = 64, shuffle = True, validation_data = ([Xp1_val, Xp2_val], ywin_val), callbacks = [tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)])

  return model

from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"
    
#Main
filepath = input("Give the filepath where the arrays are stored: ")
import_data()
split_log()
modelrank = train_model("rank")
modelwin = train_model("win")

#Start IG for win 
from alibi.explainers import IntegratedGradients as IG
n_steps = 8 #Embeddings dimension
method = "gausslegendre"
internal_batch_size = 16
nb_samples = 20
ig  = IG(modelwin,
                          layer=modelwin.layers[2],
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)

ig2 = IG(modelwin,
                          layer=modelwin.layers[3],
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)

x_test_sample = [Xp1_test[:nb_samples],Xp2_test[:nb_samples]]
from scipy import special
predictions = modelwin(x_test_sample).numpy().argmax(axis=1)

#  Compare to logit-based cumulative probs
explanation = ig.explain(x_test_sample,
                         baselines=None,
                         target=predictions)

explanation2 = ig2.explain(x_test_sample,
                         baselines=None,
                         target=predictions)

inv_map = {v: k for k, v in unique.items()}

attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)
attrs = attrs.sum(axis=2)

attrs2 = explanation2.attributions[0]
print('Attributions shape:', attrs2.shape)
attrs2 = attrs2.sum(axis=2)

i = 5
x_i = x_test_sample[0][i]
x2_i = x_test_sample[1][i]
attrs_i = attrs[i]
attrs2_i = attrs2[i]
pred = predictions[i]
pred_dict = {1: 'Player 1 Wins', 0: 'Player 2 Wins'}
print('Predicted label =  {}: {}. Actual winnner: {}.'.format(pred, pred_dict[pred], pred_dict[ywin_test[i]]))

def decode_sentence(x, reverse_index):
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return [reverse_index.get(letter, 'UNK') for letter in x]

def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    dropindexes = []
    for index, value in enumerate(list(map(lambda x: ((norm(x))), attrs))):
      if (value < 0.51 and value > 0.49):
        dropindexes.append(index)
    
    return colors, dropindexes

words = decode_sentence(x_i, inv_map)
for i in range(len(words)):
  if words[i] == "UNK": 
    until = i
    break
colors, dropindexes = colorize(attrs_i)
a = HTML(" > ".join(list(map(hlstr, words[:until], colors[:until]))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(a.data)
for index in sorted(dropindexes, reverse=True):
  colors.pop(index)
  words.pop(index)
c = HTML(" > ".join(list(map(hlstr, words, colors))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(c.data)

words = decode_sentence(x2_i, inv_map)
for i in range(len(words)):
  if words[i] == "UNK": 
    until = i
    break
colors, dropindexes2 = colorize(attrs2_i)
b = HTML(" > ".join(list(map(hlstr, words[:until], colors[:until]))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(b.data)

for index in sorted(dropindexes2, reverse=True):
  colors.pop(index)
  words.pop(index)
d = HTML(" > ".join(list(map(hlstr, words, colors))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(d.data)

#IG for Rank
from alibi.explainers import IntegratedGradients as IG
n_steps = 8 #Embeddings dimension
method = "gausslegendre"
internal_batch_size = 16
nb_samples = 20
ig  = IG(modelrank,
                          layer=modelrank.layers[2],
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)


x_test_sample = [Xp1_test[:nb_samples],Xp2_test[:nb_samples]]
from scipy import special
predictions = modelrank.predict(x_test_sample)
#  Compare to logit-based cumulative probs
cum_probs = pd.DataFrame(predictions).apply(special.expit)
labels = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
labels = np.asarray(labels).astype(int)
print(labels)
#  Compare to logit-based cumulative probs
explanation = ig.explain(x_test_sample,
                         baselines=None,
                         target=labels)

attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)
attrs = attrs.sum(axis=2)

i = 4
x_i = x_test_sample[0][i]
attrs_i = attrs[i]
pred = labels[i]
pred_dict = {0: 'Silver', 1: 'Gold', 2: 'Plat', 3: "Diamond", 4: "Master", 5: "GM"}
print('True label =  {} {}. Predicted: {} - {}.'.format(yrank_test[i], pred_dict[yrank_test[i]], pred, pred_dict[pred]))

def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    dropindexes = []
    for index, value in enumerate(list(map(lambda x: ((norm(x))), attrs))):
      if (value < 0.53 and value > 0.47):
        dropindexes.append(index)
    return colors, dropindexes

words = decode_sentence(x_i, inv_map)
for i in range(len(words)):
  if words[i] == "UNK": 
    until = i
    break
colors, dropindexes = colorize(attrs_i)
a = HTML(" > ".join(list(map(hlstr, words[:until], colors[:until]))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(a.data)
for index in sorted(dropindexes, reverse=True):
  colors.pop(index)
  words.pop(index)
c = HTML(" > ".join(list(map(hlstr, words, colors))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(c.data)

i = 15
x_i = x_test_sample[0][i]
attrs_i = attrs[i]
pred = labels[i]
pred_dict = {0: 'Silver', 1: 'Gold', 2: 'Plat', 3: "Diamond", 4: "Master", 5: "GM"}
print('True label =  {} {}. Predicted: {} - {}.'.format(yrank_test[i], pred_dict[yrank_test[i]], pred, pred_dict[pred]))

words = decode_sentence(x_i, inv_map)
for i in range(len(words)):
  if words[i] == "UNK": 
    until = i
    break
colors, dropindexes = colorize(attrs_i)
b = HTML(" > ".join(list(map(hlstr, words[:until], colors[:until]))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(b.data)
for index in sorted(dropindexes, reverse=True):
  colors.pop(index)
  words.pop(index)
d = HTML(" > ".join(list(map(hlstr, words, colors))))
with open(input("Filepath to store HTML file: "), 'w') as f:
    f.write(d.data)
