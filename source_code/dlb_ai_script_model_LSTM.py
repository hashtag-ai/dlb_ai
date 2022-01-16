#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:56:22 2021

@author: Roland

Spyder Editor

Digital Leadership Barometer Script File

Model with LSTM (RNN)
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objs as go
# import plotly.plotly as py
import chart_studio.plotly as py
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import os

datadir  = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/02_Data_Preprocessing/Output/'
labeldir = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/03_Model/Input/'
datafile  = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'
labelfile = 'url_list_2018_2020_labels.xlsx'
full_path_file       = os.path.expanduser(datadir + datafile)
full_path_label_file = os.path.expanduser(labeldir + labelfile)
datascrape = pd.read_csv(full_path_file, sep = '\t')
labels     = pd.read_excel(full_path_label_file, sheet_name = 0)
labels['agility'] = labels['agility'].round(decimals = 0)
datascrape = datascrape.merge(labels, left_on = ['ID','dl_slot'], right_on = ['id','url'], how = 'left', copy = False)

df = datascrape[['agility','text']]

# df = pd.read_csv('consumer_complaints_small.csv')

df.head()
df.info()
df.agility.value_counts()

# df.loc[df['Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
# df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit card or prepaid card'
# df.loc[df['Product'] == 'Payday loan', 'Product'] = 'Payday loan, title loan, or personal loan'
# df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
# df = df[df.Product != 'Other financial service']

df['agility'].value_counts().sort_values(ascending=False).iplot(kind='bar', yTitle='Number of Websites', 
                                                                title='Number websites per agility')

def print_plot(index):
    example = df[df.index == index][['text', 'agility']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Agility:', example[1])

print_plot(10)
print_plot(100)

df = df.reset_index(drop=True)

# REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))

# def clean_text(text):
#     """
#         text: a string
        
#         return: modified initial string
#     """
#     text = text.lower() # lowercase text
#     text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
#     text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#     text = text.replace('x', '')
# #   text = re.sub(r'\W+', '', text)
#     text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
#     return text

# df['text'] = df['text'].apply(clean_text)
# df['text'] = df['text'].str.replace('\d+', '')

print_plot(10)
print_plot(100)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['agility']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size
                   ,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss'
                   ,patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()






