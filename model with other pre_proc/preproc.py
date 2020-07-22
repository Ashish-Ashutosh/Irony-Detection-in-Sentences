#!/usr/bin/env python3

'''

Preprocessing - for files with target label and without target label

'''

import logging
import re
import itertools
import json
from autocorrect import spell
from os.path import abspath, exists
import csv
import sys
import numpy as np
import pandas as pd
from wordsegment import load, segment
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
logging.basicConfig(level=logging.INFO)


def preprocess(fp):
    '''

    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''

    
    #y = []
    corpus = []
    load()
    stopWords = set(stopwords.words('english'))
    data = {'Tweet Text': []};
    #data = {'index Label' :[], 'Tweet Text': []};
    # Open/Create a file to append data
    #csvFile = open('preprocessed_sarcasm_corpus.csv', 'a')

    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                #label = int(line.split("\t")[1])
                tweet = line.split("\t")[0]

                # remove url
                tweet = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', '', tweet, flags=re.MULTILINE)

                # remove mentions
                remove_friendtag = re.compile(r'(?:@[\w_]+)') 
                tweet = remove_friendtag.sub('',tweet)

                # remove certain hashtags
                #remove_sarcasm = re.compile(re.escape('sarcasm'),re.IGNORECASE)
                #remove_sarcastic = re.compile(re.escape('sarcastic'),re.IGNORECASE)
                #remove_irony = re.compile(re.escape('irony'),re.IGNORECASE) 
                #tweet = remove_sarcasm.sub('',tweet)
                #tweet = remove_sarcastic.sub('',tweet)
                #tweet = remove_irony.sub('',tweet)

                # Standarising words
                tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

                # contractions applied
                words = tweet.split()
                tweet = [apoDict[word] if word in apoDict else word for word in words]
                tweet = " ".join(tweet)

                hashWords =  re.findall(r'(?:^|\s)#{1}(\w+)', tweet)
                # replace #word with word
                #hashtags = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
                tweet = re.sub(r'(?:^|\s)#{1}(\w+)', r' \1', tweet)

                # word segmentation
                token_list =  word_tokenize(tweet)
                segmented_word = []
                for i in token_list:
                    if i in hashWords:
                        seg = segment(i)
                        segmented_word.extend(seg)
                    else:
                        segmented_word.append(i)

                tweet = ' '.join(segmented_word)


                # remove special symbols
                tweet = re.sub(r'[^a-zA-Z0-9 ]',r' ',re.sub(r'[$|.|@]',r' ',tweet))

                # remove stop words
                #token_list = word_tokenize(tweet)
                #wordsFiltered = []
                #for i in token_list:
                #    if i not in stopWords:
                #        wordsFiltered.append(i)
                #tweet = ' '.join(wordsFiltered)

                # remove extra whitspaces
                tweet = re.sub('[\s]+', ' ', tweet)

                # convert to lowercase
                #tweet = tweet.lower()


                # Spelling correction
                spell_list = word_tokenize(tweet)
                filterlist = []
                for i in spell_list:
                    correct_spell = spell(i)
                    filterlist.append(correct_spell)
                tweet = ' '.join(filterlist)
               

                # lemma
                tweet = word_tokenize(tweet)
                lemma_tweet = []
                for tweet_word in tweet:
                    lemma_tweet.append(WordNetLemmatizer().lemmatize(tweet_word,'v'))

                tweet = ' '.join(lemma_tweet)
                print(tweet)

                #y.append(label)
                corpus.append(tweet)

                #creating a dictionary
                #data['Tweet Text'].append(tweet)
                #data['index Label'].append(label)
    # #creating a dataframe
    # df = pd.DataFrame(data)
    # #to order the columns in the csv file (while copying from dataframes to a CSV file)
    # df_reorder = df[['Tweet Text']]
    # #df_reorder = df[['index Label', 'Tweet Text']]
    # #writing dataframe to a csv file
    # df_reorder.to_csv('preprocessed_sarcasm_corpus.csv', encoding='utf-8', index= False)

    #to create a npy file directly use the below code and comment from line 133
    corpus = list(set(corpus))
    corpus = np.array(corpus)
    return corpus
    #return corpus, y

# Contractions
def loadAppostophesDict(fp):
    apoDict = json.load(open(fp))
    return apoDict

if __name__ == "__main__":
    #DATASET_FP = list(csv.reader(open('sarcasm_corpus.csv', 'rU'), delimiter='\n'))
    DATASET_FP = "./non_sarcasm_corpus.csv"
    #DATASET_FP = "./sarcasm_corpus.csv"
    APPOSTOPHES_FP = "./appos.txt"
    apoDict = loadAppostophesDict(APPOSTOPHES_FP)
    corpus = preprocess(DATASET_FP)
    np.save('non_sarcasm_preprocessed_npy', corpus)
    #corpus, y = preprocess(DATASET_FP)
