#!/usr/bin/env python
# coding: utf-8

# In[423]:


from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

# song metia file read
song_meta = pd.read_json('../res/song_meta.json', typ = 'frame')
song_meta_indexed = song_meta.set_index('id')

# train file reaad
train = pd.read_json('../res/train.json', typ = 'frame')


# make sentence
def songs_to_words(songs):
    word = []
    for song in songs:
        word.append(str(song))
    return word

train['song_words'] = train['songs'].apply(songs_to_words)
sentences =  train['song_words'].tolist()


# w2v model training
model = Word2Vec(sentences, min_count=1, size=200, window=200)


# model save
model.save('song2vec.model')
