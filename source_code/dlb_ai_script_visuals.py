# -*- coding: utf-8 -*-
"""
Spyder Editor

Digital Leadership Barometer Script File

Visuals
"""

import pandas as pd
import texthero as hero

import matplotlib.pyplot as plt

import os

# Read input data

datadir = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/02_Data_Preprocessing/Output/'

# Files questionnaire 2018 subset for testing

# datafile = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de.csv'
# datafile = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_nocmp.csv'

# Files questionnaire 2018 full set

# datafile = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
# datafile = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

# Files questionnaire 2020 full set

# datafile = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
# datafile = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

# Files questionnaire 2018 and 2020 full set

# datafile = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de.csv'
datafile = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

full_path_file = os.path.expanduser(datadir + datafile)

datascrape = pd.read_csv(full_path_file, sep = '\t')

# Visualize top words with a bar chart

tw = hero.visualization.top_words(datascrape['text']).head(10)
plt.figure()
plt.bar(tw.index, tw)
plt.show()
print(tw.head())

# Visualize top words with a word cloud

wc = hero.visualization.wordcloud(datascrape['text'])

# Derive PCA value to use as visualization coordinates

df_scatter = pd.DataFrame(datascrape['text'].pipe(hero.tfidf).pipe(hero.pca), columns = ['pca'])
df_scatter = df_scatter['pca'].apply(pd.Series)

# Derive K-Means cluster to use for coloring

df_scatter['kmeans'] = (datascrape['text'].pipe(hero.tfidf).pipe(hero.kmeans))

# Generate scatter plot

plt.figure()
plt.scatter(df_scatter[0], df_scatter[1], c = df_scatter['kmeans'])
plt.show()



