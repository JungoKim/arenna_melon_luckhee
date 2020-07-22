#!/usr/bin/env python
# coding: utf-8

# In[1563]:


from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re
import math
import numpy as np
import pandas as pd
import io
import distutils.dir_util

from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors


# read song meta file
song_meta = pd.read_json('../res/song_meta.json', typ = 'frame')
song_meta_indexed = song_meta.set_index('id')


# make song-artist dataframe
song_artist = []
def get_song_artist(song, artist_list):
    for artist in artist_list:
        if artist is not 'Various Artists':
            song_artist.append([song, artist])

result = song_meta.apply(lambda x : get_song_artist(x.id, x.artist_name_basket), axis=1)
song_artist_df = pd.DataFrame(song_artist, columns=['song', 'artist'])
song_artist_df = song_artist_df.set_index('song')


# make song-album dataframe
song_album_df = song_meta[['id', 'album_id']]
song_album_df.columns = ['song', 'album']
song_album_df =  song_album_df.set_index('song')


# make song-gn dataframe
song_gn = []
def get_song_gn(song, gn_list):
    if len(gn_list) == 0:
        song_gn.append([song, 'nogn'])
    else:
        for gn in gn_list:
            song_gn.append([song, gn])

result = song_meta.apply(lambda x : get_song_gn(x.id, x.song_gn_dtl_gnr_basket), axis=1)
song_gn_df = pd.DataFrame(song_gn, columns=['song', 'gn'])
song_gn_df = song_gn_df.set_index('song')


# read train, val, test file
train = pd.read_json('../res/train.json', typ = 'frame')
val = pd.read_json('../res/val.json', typ = 'frame')
test = pd.read_json('../res/test.json', typ = 'frame')



# calculate song lenth and tag length
train['songs_len'] = train['songs'].apply(lambda x : len(x))
train['tags_len'] = train['tags'].apply(lambda x : len(x))
val['songs_len'] = val['songs'].apply(lambda x : len(x))
val['tags_len'] = val['tags'].apply(lambda x : len(x))
test['songs_len'] = test['songs'].apply(lambda x : len(x))
test['tags_len'] = test['tags'].apply(lambda x : len(x))


# load w2v model
model = Word2Vec.load("song2vec.model")


# make song-song co occur dataframe
song_song_list = []
def co_occur_song(song_list):
    for idx1,song1 in enumerate(song_list):
        for idx2,song2 in enumerate(song_list[idx1:idx1+30]):
            if song1 is not song2:
                song_song_list.append([song1, song2])
                song_song_list.append([song2, song1])
            
result = train['songs'].apply(co_occur_song)
co_song_df = pd.DataFrame(song_song_list, columns=['sa', 'sb'])
co_song_count_df = co_song_df.groupby(['sa', 'sb']).size().reset_index().set_index('sa')
co_song_count_df.columns = ['sb', 'count']
co_song_count_df['count_scale'] = co_song_count_df['count'].apply(lambda x : math.log(x+1))

# make popular song dataframe
pp_song_count_df = co_song_df.groupby('sa').size().sort_values(ascending=False).reset_index().set_index('sa')


# make tag-song co occur dataframe
tag_song_list = []
def co_occur_tag_song(song_list, tag_list):
    tags = list(set(tag_list))
    for tag in tags:
        for song in song_list:
            tag_song_list.append([tag, song])

result = train.apply(lambda x : co_occur_tag_song(x.songs, x.tags), axis=1)
co_tag_song_df = pd.DataFrame(tag_song_list, columns=['tag', 'song'])
co_tag_song_df_indexed =  co_tag_song_df.set_index('tag')
co_tag_song_count_df = co_tag_song_df.groupby(['tag', 'song']).size().reset_index()
co_tag_song_count_df.columns = ['tag', 'song', 'count']
co_tag_song_count_df['count_scale'] = co_tag_song_count_df['count'].apply(lambda x : math.log(x+1))
co_tag_song_count_df_indexed = co_tag_song_count_df.set_index('song')
song_has_tag_set = set(co_tag_song_count_df_indexed.index.tolist())


# make tag-tag co occur dataframe
tag_tag_list = []
def co_occur_tag(tag_list):
    for tag1 in tag_list:
        for tag2 in tag_list:
            if tag1 is not tag2:
                tag_tag_list.append([tag1, tag2])

result = train['tags'].apply(co_occur_tag)
co_tag_df = pd.DataFrame(tag_tag_list, columns=['ta', 'tb'])
co_tag_count_df = co_tag_df.groupby(['ta', 'tb']).size().reset_index().set_index('ta')
co_tag_count_df.columns = ['tb', 'count']
co_tag_count_df['count_scale'] = co_tag_count_df['count'].apply(lambda x : math.log(x+1))

# make popular tag dataframe
pp_tag_count_df = co_tag_df.groupby('ta').size().sort_values(ascending=False).reset_index().set_index('ta')
tag_has_tag_set = set(co_tag_count_df.index.tolist())



# tag reco algorithm 1 : tag_co_occur
def get_reco_tag_co_occur(tag_list, size=10):
    if len(tag_list) == 0:
        return []
    
    tag_set = tag_has_tag_set & set(tag_list)
    co_tag = co_tag_count_df.loc[tag_set]
    co_tag_count = co_tag.groupby('tb')['count_scale'].sum().sort_values(ascending=False).reset_index()
    #print(co_tag_count)
    return co_tag_count[~co_tag_count['tb'].isin(tag_list)].head(size)

def reco_tag_co_occur(tag_list, size=10):
    if len(tag_list) == 0:
        return []
    reco_top = get_reco_tag_co_occur(tag_list)
    if len(reco_top) == 0:
        return []
    return reco_top['tb'].tolist()

# tag reco algorithm 2 : tag_song_co_occur
def get_reco_tag_song_co_occur(song_list, ignore_list=[], size=10):
    if len(song_list) == 0:
        return []
    
    song_set = song_has_tag_set & set(song_list)
    co_tag = co_tag_song_count_df_indexed.loc[song_set]
    co_tag_count = co_tag.groupby('tag')['count_scale'].sum().sort_values(ascending=False).reset_index()
    #print(co_tag_count.shape)
    #print(co_tag_count)
    return co_tag_count[~co_tag_count['tag'].isin(ignore_list)].head(size)


def reco_tag_song_co_occur(song_list, ignore_list=[], size=10):
    if len(song_list) == 0:
        return []
    reco_top = get_reco_tag_song_co_occur(song_list, ignore_list)
    if len(reco_top) == 0:
        return []
    return reco_top['tag'].tolist()

# tag reco algorithm 3 : tag_song_mix
def reco_tag_song_mix(tag_list, song_list, size=10):
    reco_tag = get_reco_tag_co_occur(tag_list, 20)
    if len(reco_tag) == 0:
        return []
    reco_tag.columns = ['tag', 'count_tag']
    
    reco_song = get_reco_tag_song_co_occur(song_list, tag_list, 20)
    
    if len(reco_song) == 0:
        return reco_tag['tag'].tolist()
    
    reco_song.columns = ['tag', 'count_song']     
    reco = pd.merge(reco_tag, reco_song, how='outer', on='tag')
    reco = reco.fillna(0)
    reco['count'] = reco.apply(lambda x : x.count_tag + x.count_song * 0.1 , axis=1)
    reco = reco.sort_values('count', ascending=False)  
    #print(reco)
    return reco.head(size)['tag'].tolist()


# reco tag final 
pp_tag_count_top = pp_tag_count_df.head(100)
reco_tag_count = 0
def tag_reco_final(song_list, tag_list):
    global reco_tag_count
    if reco_tag_count % 1000 == 0:
        print(f'===== tag reco working at {reco_tag_count},  {datetime.now()} =====')
        
    song_len = len(song_list)
    tag_len = len(tag_list)
    reco = []
    if song_len > 0 and tag_len > 0:
        reco = reco_tag_song_mix(tag_list, song_list)
    elif song_len > 0 and tag_len == 0:
        reco = reco_tag_song_co_occur(song_list)
    elif song_len == 0 and tag_len > 0:
        reco = reco_tag_co_occur(tag_list)
    
    extra = 10 - len(reco)
    if extra > 0:
        reco.extend(pp_tag_count_top[~pp_tag_count_top.index.isin(reco+tag_list)].head(extra).index.tolist())
    reco_tag_count = reco_tag_count + 1
    return reco[:10]

test['tag_reco'] = test.apply(lambda x : tag_reco_final(x.songs, x.tags), axis=1)
test[['id', 'tag_reco']].to_csv('tag_reco_result.csv', encoding='utf-8-sig')




# reco song algorithm 1 : song_co_occur
def get_reco_song_co_occur(song_list, ignore_list=[], size=100):
    reco_list = []
    for song in song_list:
        if song in co_song_count_df.index:
            reco_s = co_song_count_df.loc[song]['sb'].tolist()
            reco_c = co_song_count_df.loc[song]['count_scale'].tolist()
            if type(reco_s) is not list:
                reco_list.append([reco_s, reco_c])
            else:
                for idx, reco in enumerate(reco_s):
                    reco_list.append([reco_s[idx], reco_c[idx]])
    reco_df = pd.DataFrame(reco_list, columns=['song', 'count_scale'])
    #print(reco_df.head(20))
    reco_count_df = reco_df.groupby(['song'])['count_scale'].sum().sort_values(ascending=False).reset_index()
    #print(reco_count_df.head(20))
    reco_top = reco_count_df[~reco_count_df['song'].isin(song_list+ignore_list)].head(size)
    #print(reco_top.head(20))
    return reco_top


def reco_song_co_occur(song_list, size=100):
    if len(song_list) == 0:
        return []
    reco_top = get_reco_song_co_occur(song_list)
    if len(reco_top) == 0:
        return []
    return reco_top['song'].tolist()


# reco song algorithm 2 : song_w2v
def get_reco_song_w2v(song_list, ignore_list=[], size=100):
    reco_num_per_song = int(size / len(song_list)) + 1
    reco_num_per_song = max(reco_num_per_song, 10) 
    reco_list = []
    
    for idx, song in enumerate(song_list):
        if str(song) in model.wv.vocab:
            reco_list.extend(model.wv.most_similar(str(song), topn=reco_num_per_song))
    #print(reco_list)
        
    reco_df = pd.DataFrame(reco_list, columns=['song', 'dist'])
    reco_count_df = reco_df.groupby('song')['dist'].sum().sort_values(ascending=False).reset_index()
    reco_count_df['song_id'] = reco_count_df['song'].apply(lambda x : int(x))
    reco_top = reco_count_df[~reco_count_df['song_id'].isin(song_list+ignore_list)].head(size)

    #print(reco_top[['song_id', 'dist']])
    return reco_top[['song_id', 'dist']]


def reco_song_w2v(song_list, ignore_list=[], size=100):
    if len(song_list) == 0:
        return []
    reco_top = get_reco_song_w2v(song_list, ignore_list, size)
    if len(reco_top) == 0:
        return []
    return reco_top['song_id'].tolist()


# reco song algorithm 3 : song_tag_co_occur
def get_reco_song_tag_co_occur(tag_list, ignore_list=[], size=100):
    tag_song_df = co_tag_song_count_df[co_tag_song_count_df['tag'].isin(tag_list)]
    #print(tag_song_df)
    if tag_song_df.shape[0] == 0:
        return []
    tag_song_count_df = tag_song_df.groupby('song')['count_scale'].sum().reset_index()
    #print(tag_song_count_df)
    tag_song_count_df_remove_ignore = tag_song_count_df[~tag_song_count_df['song'].isin(ignore_list)]
   
    tag_song_count_top = tag_song_count_df_remove_ignore.sort_values('count_scale', ascending=False).head(size)
    #print(tag_song_count_top.head(20))
    return tag_song_count_top


def reco_song_tag_co_occur(tag_list, ignore_list=[], size=100):
    tags = list(set(tag_list))
    if len(tags) == 0:
        return []
    reco_top = get_reco_song_tag_co_occur(tags, ignore_list, size)
    if len(reco_top) == 0:
        return []
    return reco_top['song'].tolist()


# reco song algorithm 4 : song_co_occur + album, artist, gn weight
def reco_song_co_occur_album_artist_count(song_list):
    #print(song_list)
    reco_ss = get_reco_song_co_occur(song_list, [], 200)
    
    if len(reco_ss) == 0:
        return []
    
    reco_ss.columns = ['song', 'count_song']
    reco_ss = pd.merge(reco_ss, song_album_df, how='left', left_on='song', right_on='song')
    #print(reco_ss)
    
    song_album_count = song_album_df.loc[song_list].groupby('album').size().reset_index()
    song_album_count.columns = ['album', 'count']
    #print(song_album_count)
    song_album_count = song_album_count.set_index('album')
       
    def get_album_count(album):
        count = 1
        if album in song_album_count.index:
            #print(song_album_count.loc[album]['count'])
            count = count + song_album_count.loc[album]['count']
        return math.log(count+1)
    reco_ss['count_album'] = reco_ss['album'].apply(get_album_count)
   

    reco_ss = pd.merge(reco_ss, song_meta[['id', 'artist_name_basket']], how='left', left_on='song', right_on='id')
    #print(reco_ss)
    
    song_artist_count = song_artist_df.loc[song_list].groupby('artist').size().reset_index()
    song_artist_count.columns = ['artist', 'count']
    #print(song_artist_count)
    song_artist_count = song_artist_count.set_index('artist')
       
    def get_artist_count(artist_list):    
        count = 1
        if len(artist_list) == 0:
            return count
        for artist in artist_list:
            if artist in song_artist_count.index:
                #print(song_artist_count.loc[artist]['count'])
                count = count + song_artist_count.loc[artist]['count']
        return math.log(max(int(count / len(artist_list)), 1)+1)
    reco_ss['count_artist'] = reco_ss['artist_name_basket'].apply(get_artist_count)
    
    
    reco_ss = pd.merge(reco_ss, song_meta[['id', 'song_gn_dtl_gnr_basket']], how='left', left_on='id', right_on='id')
    #print(reco_ss)
    song_gn_count = song_gn_df.loc[song_list].groupby('gn').size().reset_index()
    song_gn_count.columns = ['gn', 'count']
    #print(song_artist_count)
    song_gn_count = song_gn_count.set_index('gn')
    
    def get_gn_count(gn_list):
        count = 1
        if len(gn_list) == 0:
            return count
        for gn in gn_list:
            if gn in song_gn_count.index:
                #print(song_gn_count.loc[gn]['count'])
                count = count + song_gn_count.loc[gn]['count']
        return math.log(max(int(count / len(gn_list)), 1))
    reco_ss['count_gn'] = reco_ss['song_gn_dtl_gnr_basket'].apply(get_gn_count)

    reco = reco_ss
    reco['count'] = reco.apply(lambda x : x.count_song + x.count_album * 5 + x.count_artist * 5 + x.count_gn, axis=1)
    reco = reco.sort_values('count', ascending=False)  
    #print(reco.head(10))
    return reco.head(100)['song'].tolist()



# reco song final
pp_song_count_df_top200 = pp_song_count_df.head(200)
reco_song_count = 0
def song_reco_final(song_list, tag_list):
    global reco_song_count
    song_len = len(song_list)
    tag_len = len(tag_list)

    reco = []
    extra = 100
    
    if reco_song_count % 1000 == 0:
        print(f'===== song reco working at {reco_song_count},  {datetime.now()} =====')
    
    if song_len > 0 and tag_len > 0:
        reco = reco_song_co_occur_album_artist_count(song_list)
        extra = 100 - len(reco)
        if extra > 0:
            reco.extend(reco_song_w2v(song_list, reco, extra))
        extra = 100 - len(reco)
        if extra > 0:
            reco.extend(reco_song_tag_co_occur(tag_list, song_list+reco, extra))
    elif song_len > 0 and tag_len == 0:
        reco = reco_song_co_occur_album_artist_count(song_list)
        extra = 100 - len(reco)
        if extra > 0:
            reco.extend(reco_song_w2v(song_list, reco, extra))
    elif song_len == 0 and tag_len > 0:
        reco = reco_song_tag_co_occur(tag_list)
    
    extra = 100 - len(reco)
    if extra > 0:
        reco.extend(pp_song_count_df_top200[~pp_song_count_df_top200.index.isin(song_list+reco)].head(extra).index.tolist())
    reco_song_count = reco_song_count + 1
    return reco[:100]


test['song_reco'] = test.apply(lambda x : song_reco_final(x.songs, x.tags), axis=1)
test[['id', 'song_reco']].to_csv('song_reco_result.csv')


# make result file
song_reco = pd.read_csv('song_reco_result.csv')
song_reco['song_reco'] = song_reco['song_reco'].apply(lambda x : eval(x))
test['song_reco'] = song_reco['song_reco']

tag_reco = pd.read_csv('tag_reco_result.csv')
tag_reco['tag_reco'] = tag_reco['tag_reco'].apply(lambda x : eval(x))
test['tag_reco'] = tag_reco['tag_reco']


answers = []

def make_answer(pid, songs, tags):
    answers.append({
        "id": pid,
        "songs": songs,
        "tags": tags
    }) 

result = test.apply(lambda x : make_answer(x.id, x.song_reco, x.tag_reco), axis=1)


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

write_json(answers, 'results.json')

print("result file created...")
