# -*- coding: utf-8 -*-
"""
Spyder Editor

Digital Leadership Barometer Script File

Model with unigrams and bigrams, all Digital Leadership dimensions
"""

import pandas as pd
import texthero as hero
import sklearn
from sklearn.feature_selection import chi2
import numpy as np

import matplotlib.pyplot as plt
from math import pi

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import os

# Read input data

datadir  = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/02_Data_Preprocessing/Output/'
labeldir = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/03_Model/Input/'

# Files questionnaire 2018 subset for testing

# datafile         = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de.csv'
# datafile         = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_nocmp.csv'
# labelfile        = 'url_list_2018_labels_all_DL_dimensions.xlsx'
# tf_file_out      = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_TF.csv'
# tfidf_file_out   = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_TFIDF.csv'
# tf_file_list_out = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_TF_LIST.csv'

# Files questionnaire 2018 full set

# datafile         = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
# datafile         = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'
# labelfile        = 'url_list_2018_labels_all_DL_dimensions.xlsx'
# tf_file_out      = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_TF.csv'
# tfidf_file_out   = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_TFIDF.csv'
# tf_file_list_out = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_TF_LIST.csv'

# Files questionnaire 2020 full set

# datafile         = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
# datafile         = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'
# labelfile        = 'url_list_2020_labels_all_DL_dimensions.xlsx'
# tf_file_out      = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_TF.csv'
# tfidf_file_out   = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_TFIDF.csv'
# tf_file_list_out = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_TF_LIST.csv'

# Files questionnaire 2018 and 2020 full set

# datafile         = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
datafile         = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'
labelfile        = 'url_list_2018_2020_labels_all_DL_dimensions.xlsx'
# labelfile        = 'url_list_2018_2020_labels_all_DL_dimensions_final.xlsx'
# tf_file_out      = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_TF.csv'
# tfidf_file_out   = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_TFIDF.csv'
# tf_file_list_out = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_TF_LIST.csv'

full_path_file             = os.path.expanduser(datadir + datafile)
full_path_label_file       = os.path.expanduser(labeldir + labelfile)
# full_path_tf_file_out      = os.path.expanduser(datadir + tf_file_out)
# full_path_tfidf_file_out   = os.path.expanduser(datadir + tfidf_file_out)
# full_path_tf_file_list_out = os.path.expanduser(datadir + tf_file_list_out)

# Read scraped data

datascrape = pd.read_csv(full_path_file, sep = '\t')
datascrape['word_count'] = datascrape['text'].apply(lambda x: len(x.split()))

# Read labels

labels = pd.read_excel(full_path_label_file, sheet_name = 0)
labels.iloc[:, -10:] = labels.iloc[:, -10:].round(decimals = 0)
# labels.iloc[:, -10:] = labels.iloc[:, -10:].round(decimals = 2)

# Merge scraped data and labels
# Files questionnaire 2018 subset for testing
# Files questionnaire 2018 full set
# Files questionnaire 2020 full set

# datascrape = datascrape.merge(labels, left_on = 'ID', right_on = 'id', how = 'left', copy = False)

# Merge scraped data and labels
# Files questionnaire 2018 and 2020 full set

datascrape = datascrape.merge(labels, left_on = ['ID','dl_slot'], right_on = ['id','url'], how = 'left', copy = False)

# Derive term frequency and TF-IDF of the input

df_term_frequency_1 = hero.term_frequency(datascrape['text'], return_feature_names = False)
df_tfidf_1          = hero.tfidf(datascrape['text'], return_feature_names = False)

df_term_frequency_2 = hero.term_frequency(datascrape['text'], return_feature_names = True)
df_tfidf_2          = hero.tfidf(datascrape['text'], return_feature_names = True)

df_term_frequency_1_df = pd.DataFrame(df_term_frequency_1.to_list())
df_tfidf_1_df          = pd.DataFrame(df_tfidf_1.to_list())
df_term_frequency_2_df = pd.DataFrame(df_term_frequency_2[1])

# df_term_frequency_1.to_csv(full_path_tf_file_out, sep = ';', index = True)
# df_tfidf_1.to_csv(full_path_tfidf_file_out, sep = ';', index = True)
# df_term_frequency_2_df.to_csv(full_path_tf_file_list_out, sep = ';', index = True)

# Generate bag of words object

counter = sklearn.feature_extraction.text.CountVectorizer()
# counter = sklearn.feature_extraction.text.CountVectorizer(max_features = 1000)

# Get bag of words model as sparse matrix

bag_of_words = counter.fit_transform(datascrape['text'])
df_bag_of_words = pd.DataFrame(bag_of_words.toarray())

# Generate tf-idf object

# Median accuracy 0.45 (LogisticRegression)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer()

# Median accuracy 0.45 (MultinomialNB)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features = 1000)

# Median accuracy 0.47 (LinearSVC)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf = True)

# Median accuracy 0.45 (MultinomialNB)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df = 5)

# Median accuracy 0.45 (LogisticRegression)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(norm = 'l2')

# Median accuracy 0.47 (LinearSVC)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf = True
#                                                        ,norm = 'l2')

# Median accuracy 0.46 (MultinomialNB)
# features(211,7406)
# tfidf = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf = True
#                                                         ,min_df = 5
#                                                         ,norm = 'l2')

# Median accuracy 0.46 (LogisticRegression)
# features(211,9063)
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf = True
                                                        ,min_df = 0.015
                                                        ,max_df = 0.65
                                                        ,norm = 'l2'
                                                        ,ngram_range=(1,2))

# Get tf-idf matrix as sparse matrix

features = tfidf.fit_transform(datascrape['text']).toarray()
df_tfidf_sk = pd.DataFrame(features)
print(features.shape)

# Get the words corresponding to the vocab index

df_feat_names_sk = pd.DataFrame(tfidf.get_feature_names())

# Derive difference between results of texthero and sklearn libraries

df_diff = pd.concat([df_term_frequency_2_df,df_feat_names_sk]).drop_duplicates(keep=False)

# Subroutine to process models for every digital leadership dimension

def model_for_dl_dim(dim):
    
    print('****************************************************')
    print('*** ' + dim)
    print('****************************************************')

    # Plot distribution of dimensions (labels) of original dataset
    
    fig = plt.figure(figsize=(8,6))
    datascrape.groupby(dim).text.count().plot.bar(ylim=0, title = 'Original Dataset ' + dim)
    plt.show()
    
    # Plot distribution of word count by dimension and website of original dataset
    
    df_wc_max = datascrape['word_count'].max()
    
    fig, ax = plt.subplots(3, 2, figsize = (12,8))
    fig.tight_layout(pad=3.0)
    fig.suptitle('Verteilung der Anzahl Wörter pro ' + dim, fontsize=15)
    
    bins = 30
    
    ax[0, 0].hist(datascrape['word_count'].loc[datascrape[dim] == 1], bins = bins, color = 'lightblue')
    ax[0, 0].set_title(dim + ' 1', fontsize = 13)
    # ax[0, 0].set_xlim(0, 100000)
    ax[0, 0].set_ylabel('Anzahl Webseiten')
    ax[0, 0].yaxis.get_major_locator().set_params(integer=True)
    
    ax[0, 1].hist(datascrape['word_count'].loc[datascrape[dim] == 2], bins = bins, color = 'deepskyblue')
    ax[0, 1].set_title(dim + ' 2', fontsize = 13)
    # ax[0, 1].set_xlim(0, 100000)
    ax[0, 1].yaxis.get_major_locator().set_params(integer=True)
    
    ax[1, 0].hist(datascrape['word_count'].loc[datascrape[dim] == 3], bins = bins, color = 'dodgerblue')
    ax[1, 0].set_title(dim + ' 3', fontsize = 13)
    # ax[1, 0].set_xlim(0, 100000)
    ax[1, 0].set_ylabel('Anzahl Webseiten')
    ax[1, 0].yaxis.get_major_locator().set_params(integer=True)
    
    ax[1, 1].hist(datascrape['word_count'].loc[datascrape[dim] == 4], bins = bins, color = 'blue')
    ax[1, 1].set_title(dim + ' 4', fontsize = 13)
    # ax[1, 1].set_xlim(0, 100000)
    ax[1, 1].set_xlabel('Anzahl Wörter')
    ax[1, 1].yaxis.get_major_locator().set_params(integer=True)
    
    ax[2, 0].hist(datascrape['word_count'].loc[datascrape[dim] == 5], bins = bins, color = 'midnightblue')
    ax[2, 0].set_title(dim + ' 5', fontsize = 13)
    # ax[2, 0].set_xlim(0, 100000)
    ax[2, 0].set_xlabel('Anzahl Wörter')
    ax[2, 0].set_ylabel('Anzahl Webseiten')
    ax[2, 0].yaxis.get_major_locator().set_params(integer=True)
    
    ax[2, 1].set_visible(False)
    
    plt.show()
    
    # Find the terms that are the most correlated with each of the dimensions (labels)
    
    dim_dic = datascrape[dim].drop_duplicates().sort_values()
    # dim_dic = pd.Series([1,2,3,4,5], name = 'dim_dic')
    dim_dic.reset_index(drop = True, inplace = True)
    labels_s = datascrape[dim]
    
    N = 5
    
    for i in dim_dic.items():
        features_chi2 = chi2(features, labels_s == i[1])
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(i[1]))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
    
    # Model Selection and evaluate accuracy
    
    models = [RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
             ,LinearSVC()
             ,MultinomialNB()
             ,LogisticRegression(random_state = 0)
             ,SGDClassifier()]
    
    CV = 5
    cv_df = pd.DataFrame(index = range(CV * len(models)))
    entries = []
    
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels_s, scoring = 'accuracy', cv = CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns = ['model_name', 'fold_idx', 'accuracy'])
    
    print(cv_df.groupby('model_name').accuracy.mean())
    
    # Visualize model accuracy
    
    sns.boxplot(x = 'model_name', y = 'accuracy', data = cv_df)
    sns.stripplot(x = 'model_name', y = 'accuracy', data = cv_df, size = 8
                 ,jitter = True, edgecolor = 'gray', linewidth = 2)
    plt.title('Model Accuracy for ' + dim)
    plt.show()
    
    # Model evaluation with confusion matrix
    
    # model = LogisticRegression()
    model = MultinomialNB()
    
    # Balance the imbalanced dataset with RandomOverSampler
    
    ros = RandomOverSampler(random_state = 777)
    X_ROS, y_ROS = ros.fit_resample(features, labels_s)
    
    # Balance the imbalanced dataset with SMOTE (Synthetic Minority Over-Sampling Technique)
    
    if dim != 'error_culture':
        smote = SMOTE(random_state = 777, k_neighbors = 3)
        X_smote, y_smote = smote.fit_resample(features, labels_s)
    
    # Split data into training and test data
    
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features
                                                                                    ,labels_s, datascrape.index
                                                                                    ,test_size = 0.25, random_state = 0)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_ROS
                                                               ,y_ROS
                                                               ,test_size = 0.25, random_state = 0)
    
    if dim != 'error_culture':
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_smote
                                                                   ,y_smote
                                                                   ,test_size = 0.25, random_state = 0)
    
    # ros.sample_indices_
    
    # Derive and plot train/ test split of original dataset
    
    datascrape['data_type'] = ['not_set'] * datascrape.shape[0]
    datascrape.loc[indices_train, 'data_type'] = 'train'
    datascrape.loc[indices_test, 'data_type'] = 'test'
    print(datascrape.groupby([dim, 'data_type'])[[dim]].count())
    # datascrape.groupby([dim, 'data_type']).dim.count().plot.bar(ylim=0, title = 'Original Dataset')
    
    df_y_tr = y_train.value_counts().sort_index()
    df_y_tr = df_y_tr.rename('dim_train')
    df_y_te = y_test.value_counts().sort_index()
    df_y_te = df_y_te.rename('dim_test')
    df_y = pd.concat([df_y_tr, df_y_te], axis = 1)
    
    fig = plt.figure(figsize=(8,6))
    df_y.plot.bar(ylim=0, stacked = True, title = 'Original Dataset ' + dim)
    plt.show()
    
    # Plot distribution of dimensions (labels) of oversampled dataset (RandomOverSampler)
    
    fig = plt.figure(figsize=(8,6))
    y_ROS.value_counts().sort_index().plot.bar(ylim=0, title = 'Oversampled Dataset ' + dim)
    plt.show()
    
    # Derive and plot train/ test split of oversampled dataset (RandomOverSampler)
    
    df_y_tr_r = y_train_r.value_counts().sort_index()
    df_y_tr_r = df_y_tr_r.rename('dim_train')
    df_y_te_r = y_test_r.value_counts().sort_index()
    df_y_te_r = df_y_te_r.rename('dim_test')
    df_y_r = pd.concat([df_y_tr_r, df_y_te_r], axis = 1)
    
    fig = plt.figure(figsize=(8,6))
    df_y_r.plot.bar(ylim=0, stacked = True, title = 'Oversampled Dataset ' + dim)
    plt.show()
    
    # Model training with original dataset
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred, labels = dim_dic)
    fig, ax = plt.subplots(figsize = (8,6))
    sns.heatmap(conf_mat, annot = True, fmt = 'd',
                xticklabels = dim_dic.values, yticklabels = dim_dic.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Original Dataset ' + dim)
    plt.show()
    
    # Analyzing the misclassifications with original dataset

    for actual in dim_dic.index:
      for predicted in dim_dic.index:
        if predicted != actual and conf_mat[actual, predicted] >= 1:
            print("'{}' predicted as '{}' : {} example(s).".format(dim_dic[actual], dim_dic[predicted], conf_mat[actual, predicted]))
            print(datascrape.loc[indices_test[(y_test == dim_dic[actual].astype(int)) & (y_pred == dim_dic[predicted].astype(int))]][[dim, 'dl_slot']])
            print('')

    # Find the terms that are the most correlated with each of the dimensions (labels) using selected model
    
    model.fit(features, labels_s)
    
    N = 5
    
    for i in dim_dic.index:
      indices = np.argsort(model.coef_[i])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
      bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
      print("# '{}':".format(dim_dic[i]))
      print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
      print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
    
    # Prepare classification report for each class of original dataset
    
    print(metrics.classification_report(y_test, y_pred, target_names = datascrape[dim].astype(str).unique().sort()))
    cr = metrics.classification_report(y_test, y_pred, output_dict = True)
    df_cr = pd.DataFrame(cr).transpose()
    df_cr['dl_dim'] = dim
    
    # Prepare return variables
    
    df_pred = pd.DataFrame([[y_test.mean(),y_pred.mean(),dim]]
                          ,columns = ['dl_dim_act','dl_dim_pred','dl_dim'])
    
    # Model training with oversampled dataset (RandomOverSampler)
    
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    
    conf_mat = confusion_matrix(y_test_r, y_pred_r)
    fig, ax = plt.subplots(figsize = (8,6))
    sns.heatmap(conf_mat, annot = True, fmt = 'd',
                xticklabels = dim_dic.values, yticklabels = dim_dic.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Oversampled Dataset (RandomOverSampler) ' + dim)
    plt.show()
    
    # Prepare classification report for each class of oversampled dataset (RandomOverSampler)
    
    print(metrics.classification_report(y_test_r, y_pred_r, target_names = datascrape[dim].astype(str).unique().sort()))
    
    # Model training with oversampled dataset (SMOTE)
    
    if dim != 'error_culture':
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_test_s)
        
        conf_mat = confusion_matrix(y_test_s, y_pred_s)
        fig, ax = plt.subplots(figsize = (8,6))
        sns.heatmap(conf_mat, annot = True, fmt = 'd',
                    xticklabels = dim_dic.values, yticklabels = dim_dic.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Oversampled Dataset (SMOTE) ' + dim)
        plt.show()
    
    # Prepare classification report for each class of oversampled dataset (SMOTE)
    
    if dim != 'error_culture':
        print(metrics.classification_report(y_test_s, y_pred_s, target_names = datascrape[dim].astype(str).unique().sort()))

    return df_cr, df_pred

# Loop over all digital leadership dimensions

dl_dim_all = ['autonomy','transparency','goals_identification','digital_governance','digital_literacy'
              ,'error_culture','work_life_balance','agility','customer_centricity','internal_cooperation']
# dl_dim_all = 'autonomy','transparency'

df_dim_cr   = pd.DataFrame()
df_dim_pred = pd.DataFrame()

for dl_dim in dl_dim_all:
    df_dim_cr_tmp, df_dim_pred_tmp = model_for_dl_dim(dl_dim)
    
    if '1.0' not in df_dim_cr_tmp.index:
        df_dim_cr_tmp_missing = pd.DataFrame([[0, 0, 0, 0, dl_dim]]
                                             , index = ['1.0']
                                             , columns = ['precision', 'recall', 'f1-score'
                                                        , 'support', 'dl_dim'])
        df_dim_cr_tmp = df_dim_cr_tmp_missing.append(df_dim_cr_tmp)
    
    df_dim_cr = df_dim_cr.append(df_dim_cr_tmp)
    df_dim_pred = df_dim_pred.append(df_dim_pred_tmp)

df_dim_cr.reset_index(inplace = True)
df_dim_pred.reset_index(drop = True, inplace = True)

# Plot f1-scores and accuracy for all digital leadership dimensions

lab_1 = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == '1.0']
lab_2 = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == '2.0']
lab_3 = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == '3.0']
lab_4 = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == '4.0']
lab_5 = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == '5.0']

acc   = df_dim_cr['f1-score'].loc[df_dim_cr['index'] == 'accuracy']

x = np.arange(len(dl_dim_all))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*2, lab_1, width, label = 'class 1')
rects2 = ax.bar(x - width,   lab_2, width, label = 'class 2')
rects3 = ax.bar(x,           lab_3, width, label = 'class 3')
rects4 = ax.bar(x + width,   lab_4, width, label = 'class 4')
rects5 = ax.bar(x + width*2, lab_5, width, label = 'class 5')
line   = ax.plot(dl_dim_all, acc, label = 'accuracy')

ax.set_ylabel('Percentage')
ax.set_title('F1-Scores and Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(dl_dim_all, rotation = 90)
ax.legend()
fig.tight_layout()
plt.show()

# Plot spider diagram with actual and predicted values for all digital leadership dimensions

pred_act  = df_dim_pred['dl_dim_act'].tolist()
pred_pred = df_dim_pred['dl_dim_pred'].tolist()

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8), subplot_kw = dict(polar = True))

angles = [n / float(len(dl_dim_all)) * 2 * pi for n in range(len(dl_dim_all))]
angles += angles[:1]

plt.xticks(angles[:-1], dl_dim_all, color = 'grey', size = 12)
plt.yticks(np.arange(1, 6), ['1', '2', '3', '4', '5'], color = 'grey', size = 12)
plt.ylim(0, 5)
ax.set_rlabel_position(30)

pred_act += pred_act[:1]
ax.plot(angles, pred_act, linewidth = 1, linestyle = 'solid', label = 'Actuals')
ax.fill(angles, pred_act, 'skyblue', alpha = 0.4)

pred_pred += pred_pred[:1]
ax.plot(angles, pred_pred, linewidth = 1, linestyle = 'solid', label = 'Predicted')
ax.fill(angles, pred_pred, 'lightpink', alpha = 0.4)
 
plt.legend(loc = 'upper right', bbox_to_anchor = (0.1, 0.1))
plt.show()





