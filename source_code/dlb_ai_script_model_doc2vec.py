#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:33:16 2021

@author: Roland

Spyder Editor

Digital Leadership Barometer Script File

Model with doc2vec
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

import os

# datafile = 'complaints_subset.csv'

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

# df = pd.read_csv(full_path_file)
# df = df[['Consumer complaint narrative','Product']]
# df = df[pd.notnull(df['Consumer complaint narrative'])]
# df.rename(columns = {'Consumer complaint narrative':'narrative'}, inplace = True)

df.head(10)
df.shape

df.index = range(211)
df['text'].apply(lambda x: len(x.split(' '))).sum()

cnt_pro = df['agility'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Product', fontsize=12)
plt.xticks(rotation=90)
plt.show()

def print_complaint(index):
    example = df[df.index == index][['text', 'agility']].values[0]
    if len(example) > 0:
        print(example[0])
        print('agility:', example[1])

# print_complaint(12)
# print_complaint(20)

# from bs4 import BeautifulSoup
# def cleanText(text):
#     text = BeautifulSoup(text, "lxml").text
#     text = re.sub(r'\|\|\|', r' ', text) 
#     text = re.sub(r'http\S+', r'<URL>', text)
#     text = text.lower()
#     text = text.replace('x', '')
#     return text
# df['text'] = df['text'].apply(cleanText)

train, test = train_test_split(df, test_size=0.3, random_state=42)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

# nltk.download('punkt')

train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.agility]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.agility]), axis=1)

train_tagged.values[30]

import multiprocessing
cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))






