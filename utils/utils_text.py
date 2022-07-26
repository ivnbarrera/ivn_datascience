from time import sleep
import datetime as dt
import pandas as pd
import numpy as np
import math
import itertools
import re
from itertools import compress

from nltk.corpus import wordnet
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

tokenizer = RegexpTokenizer(r"\w+")
sp_checker = SpellChecker(language='en')
wnl = WordNetLemmatizer()
ps = PorterStemmer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def remove_stop_words(text, stopwords, lemmatize=True):
    if lemmatize:
        text = " ".join([wnl.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])
    pattern = re.compile(r'\b(' + r'|'.join(stopwords)+ r')\b\s*')
    return pattern.sub('', text)

def process_phone(phone, country_code='1'):
    """get only numbers of phones"""
    if pd.isnull(phone):
        return phone
    elif not isinstance(phone, str):
        return phone
    else:
        temp = re.findall(r'\d+', phone)
        new_phone = ''.join(temp)
        if len(new_phone) == 11 and new_phone[0]==country_code:
            new_phone = new_phone[1:]
        return new_phone

def preprocess_text(text, lower=True,
                    punctuation=False,
                    spelling=False,
                    lemmatization=False,
                    stemming=False,
                    remove_stop=False, stopwords=None):
    if text:
        if lower:
            text = text.lower()
        text = re.sub('  +', ' ', text)
        text = re.sub('\n', ' ', text)
        text = text.strip().strip('"').strip("'").lower().strip()
        words = nltk.word_tokenize(text)
        # remove punctuation
        if punctuation:
            words = tokenizer.tokenize(text)
        # spelling correction
        if spelling:
            wlist = []
            for word in words:
                correct_word = sp_checker.correction(word)
                wlist.append(correct_word)
            words = wlist

        if lemmatization:
            words = [wnl.lemmatize(w, get_wordnet_pos(w)) for w in words]

        if stemming:
            words = [ps.stem(w) for w in words]
        if remove_stop:
            if not stopwords:
                stopwords = nltk.corpus.stopwords.words('english')
            concat = remove_stop_words(" ".join(words), stopwords, lemmatize=False)
            words = nltk.word_tokenize(concat)


        text = " ".join(words)

    return text

def preprocess_text_noTags_noDigits(text):
    """ pre process any text"""
    # lowercase
    if text:
        text=text.lower()

        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.0001, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.0001, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(20, 8), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()