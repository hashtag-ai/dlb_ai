# -*- coding: utf-8 -*-
"""
Spyder Editor

Digital Leadership Barometer Script File

Data Preprocessing
"""

import pandas as pd
import texthero as hero
from texthero import preprocessing

# import nltk
# nltk.download()
# from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

import spacy
from spacy_langdetect import LanguageDetector
nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')

# Experiment with langdetect, a re-implementation of Google’s language-detection library
# from langdetect import DetectorFactory, detect, detect_langs

import os

# Read input data

datadir_in  = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/02_Data_Preprocessing/Input/'
datadir_out = '~/OneDrive/Documents/MAS/MAS_Digital_Business/Thesis/Thesis_Phase_6/02_Data_Preprocessing/Output/'

# Files questionnaire 2018 subset for testing

# datafile_in  = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset.csv'
# datafile_out = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_subset_OUT_de_nocmp.csv'

# Files questionnaire 2018 full set

# datafile_in  = 'url_list_2018_scraped_texts_aggregated_merged_FINAL.csv'
# datafile_out = 'url_list_2018_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

# Files questionnaire 2020 full set

# datafile_in  = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL.csv'
# datafile_out = 'url_list_2020_run_5_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

# Files questionnaire 2018 and 2020 full set

datafile_in  = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL.csv'
datafile_out = 'url_list_2018_2020_scraped_texts_aggregated_merged_FINAL_OUT_de_nocmp.csv'

# List of named entities, company names, additional stop words etc

companies_in  = 'url_list_2018_2020_named_entities_companies.xlsx'

# Full paths and filenames

full_path_file_in  = os.path.expanduser(datadir_in  + datafile_in)
full_path_file_out = os.path.expanduser(datadir_out + datafile_out)
full_path_cmp_in   = os.path.expanduser(datadir_in  + companies_in)

# Read scraped data

datascrape = pd.read_csv(full_path_file_in, sep = '\t')

# Read list of named entities, company names, additional stop words etc

cmp_1 = pd.read_excel(full_path_cmp_in, sheet_name = 0)
cmp_2 = pd.read_excel(full_path_cmp_in, sheet_name = 1)

cmp_df = cmp_1.append(cmp_2)

# Remove HTML-tags

datascrape['text'] = datascrape['text'].str.replace('\[->\w+<-\]','')

# Remove non-German sentences

nlp_de.add_pipe(LanguageDetector(), name='language_detector', last=True)

# Error because of text length (1'301'677) -> increased from 1'000'000 to 2'000'000
print(nlp_de.max_length)
nlp_de.max_length = 2000000
print(nlp_de.max_length)

def detect_remove_lg(inp_text):
    doc = nlp_de(inp_text)
    text_new = ''
    for sent in doc.sents:
        if sent._.language['language'] == 'de':
            text_new = text_new + sent.text.strip()
            text_new = text_new + ' '
    text_ret = text_new.strip()
    return text_ret

datascrape['text'] = datascrape['text'].apply(detect_remove_lg)

# tdf[0] = tdf[0].apply(detect_remove_lg)

# tdf = pd.DataFrame(["This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin."])
# tdf = pd.DataFrame(["Das ist ein schöner text. This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin."])
# tdf = pd.DataFrame(["This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."])
# tdf = pd.DataFrame(["This is English text Er lebt mit seinen Eltern und seiner Schwester in Berlin Yo me divierto todos los días en el parque Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne"])
# tdf = pd.DataFrame(["This.is.English.text.Er.lebt.mit.seinen.Eltern.und.seiner.Schwester.in.Berlin.Yo.me.divierto.todos.los.días.en.el.parque.Je.m'appelle.Angélica.Summer,.j'ai.12.ans.et.je.suis.canadienne"])
# tdf = pd.DataFrame(["this. is. english. text. er. lebt. mit. seinen. eltern. und. seiner. schwester. in. berlin."])

# text_test = "this. is. english. text. er. lebt. mit. seinen. eltern. und. seiner. schwester. in. berlin."

# doc = nlp_de(tdf[0].to_string())
# doc = nlp_de(text_test)
# # document level language detection (average language)
# print(doc._.language)
# # sentence level language detection
# for sent in doc.sents:
#     # print(sent, sent._.language)
#     print(sent, sent._.language['language'])

# tdf[0] = tdf[0].apply(
#          lambda row: ' '.join([sent.text.strip() for sent in nlp_de(row) if sent._.language['language'] == 'de']))

# Experiment with langdetect, a re-implementation of Google’s language-detection library
# for word in text.split():
#     try:
#         lg = detect(word)
#     except:
#         lg = 'No lang'
#     print(word,lg)

# Create a custom cleaning pipeline

custom_pipeline_1 = [preprocessing.fillna
                    ,preprocessing.lowercase
                    ,preprocessing.remove_digits
                    ,preprocessing.remove_urls]

# Pass the custom cleaning pipeline to the pipeline argument

datascrape['text'] = hero.clean(datascrape['text'], pipeline = custom_pipeline_1)

# Remove stop words in different languages

stop_words_ge = stopwords.words('german')
stop_words_fr = stopwords.words('french')
stop_words_it = stopwords.words('italian')
stop_words_en = stopwords.words('english')
# print(stop_words_ge)
# print(stop_words_fr)
# print(stop_words_it)
# print(stop_words_en)

stop_words_all = stop_words_ge + stop_words_fr + stop_words_it + stop_words_en
# print(stop_words_all)

datascrape['text'] = hero.remove_stopwords(datascrape['text'], stop_words_all)

# Lemmatization of the input

datascrape['text'] = datascrape['text'].apply(
                        lambda row: ' '.join([w.lemma_ for w in nlp_de(row)]))

# Remove named entities

datascrape['text'] = datascrape['text'].apply(
                        lambda row: ' '.join([ent.text for ent in nlp_de(row) if not ent.ent_type_]))

# Stemming of the input

# datascrape['text'] = hero.stem(datascrape['text'], stem = 'snowball', language = 'german')

# Create a custom cleaning pipeline

custom_pipeline_2 = [preprocessing.remove_punctuation
                    ,preprocessing.remove_diacritics]

# Pass the custom cleaning pipeline to the pipeline argument

datascrape['text'] = hero.clean(datascrape['text'], pipeline = custom_pipeline_2)

# Remove listed words of named entities, company names, additional stop words etc

cmp_list = cmp_df['named_entity_company'].tolist()
datascrape['text'] = hero.remove_stopwords(datascrape['text'], cmp_list)

# Create a custom cleaning pipeline

custom_pipeline_3 = [preprocessing.remove_whitespace]

# Pass the custom cleaning pipeline to the pipeline argument

datascrape['text'] = hero.clean(datascrape['text'], pipeline = custom_pipeline_3)

# Write output data

datascrape.to_csv(full_path_file_out, sep = '\t', index = False)



