#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import json
import re
import string
from collections import Counter

from nltk.corpus import stopwords


# In[ ]:


def to_dict(string):
    """Transforma una cadena de caracteres con forma de diccionario a diccionario"""
    if string != "[]":
        string = json.loads(string.replace("'", "\""))
        return ",".join([s["screen_name"] for s in string])
    return ""

def to_list(list_):
    """Transforma una cadena de caracteres con forma de lista a lista"""
    if list_ != "[]":
        list_ = list_[1:-1]
        list_ = list_.split(",")
        return ",".join([s.strip().strip("'") for s in list_])
    return ""

def normalize(s):
    """Reemplaza las letras con tildes y retorna la cadena de caracteres en minuscula"""
    replacements = (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"))
    for a, b in replacements:
        s = s.lower()
        s = s.replace(a, b)
    return s

def deEmojify(text):
    """Quita los emojis de los tweets"""
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r"", text)

def cleanTxt(text):
    """Elimina mentions, hiperlinks, quita el simbolo "#" y el "RT""""
    text = re.sub(r"@[a-zA-Z0-9]+", "", text) #Removes @mentions
    text = re.sub(r"#", "", text) #Removing the "#" symbol
    text = re.sub(r"RT[\s]+", "", text) #Removing RT
    text = re.sub(r"https?:\/\/\S+", "", text) #Remove the hyperlink
    return text

def replace_punct(s):
    """Elimina los signos de puntuacion"""
    for i in string.punctuation:
        if i in s:
            s = s.replace(i, "").strip()
    return s

def replace_num(s):
    """Remueve los numeros de los tweets"""
    for i in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        s = s.replace(i, "")
    return s

def tokenizador(text):
    """Tokeniza el texto del tweet"""
    important_words = []
    for word in text.split(" "):
        if word not in stopwords.words("spanish"):
            if word != "":
                important_words.append(word)
    return " ".join(important_words).strip()

def foo(text):
    """Elimina mas signos de puntuacion"""
    forbidden = ("?", "¿", "¡", "!", ",", ".", ";", ":", "-", "'", "+", "$", "/", "*",'«','»', "~", "(", ")")
    aux = ""
    for v in text:
        if not v in forbidden:
            aux += v
    return aux

def quita_palabras_pequeñas(text):
    """Quita palabras de longitud menor a 4 del texto del tweet"""
    return " ".join([word for word in text.split(" ") if len(word) >= 5])            


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv(\'C:/Users/Daniel/Desktop/csv/dia 25/no trends/tweets_25_notendencias_raw.csv\')\ndf.drop([\'Unnamed: 0\',\'Unnamed: 0.1\'], axis = 1, inplace = True)\n\ndf_summary = pd.read_csv("C:/Users/Daniel/Desktop/por hora/lista_tendencias_25_por_hora.csv", sep = ";")')


# # PREPROCESAMIENTO

# In[ ]:


# 1. Hace drop a las columnas de ids, husos horarios, url y traducciones
# 2. Filtra los tweets por idioma ("es")

columns_to_drop = ["conversation_id", "cashtags", "timezone", "user_id", "name", "near", "geo", "source",
                   "user_rt_id", "user_rt", "retweet_id", "retweet_date", "translate", "trans_src",
                   "trans_dest", "place", "quote_url", "thumbnail", "created_at", "id", "link"]

df.drop(columns_to_drop, axis = 1, inplace = True)

df = df[df.language == "es"]

df.drop("language", axis = 1, inplace = True)

df = df.reset_index(drop = True)


# In[ ]:


# Transforma la columna "reply_to" a diccionario
# Elimina las filas donde no es posible

reply_to_rows = []
for num, row in enumerate(df.reply_to):
    try:
        to_dict(row)
    except:
        reply_to_rows.append(num)
        
df.drop(reply_to_rows, inplace = True)

df.reply_to = df.reply_to.apply(to_dict)

df = df.reset_index(drop = True)


# In[ ]:


# Transforma la columna "mentions" a diccionario
# Elimina las filas donde no es posible

mention_rows = []
for num, row in enumerate(df.mentions):
    try:
        to_dict(row)
    except:
        mention_rows.append(num)
        
df.drop(mention_rows, inplace = True)

df.mentions = df.mentions.apply(to_dict)

df = df.reset_index(drop = True)


# In[ ]:


# Transforma la columna "hashtags" a lista
# Elimina las filas donde no es posible

hashtags_rows = []
for num, row in enumerate(df.hashtags):
    try:
        to_list(row)
    except:
        hashtags_rows.append(num)
        
df.drop(hashtags_rows, inplace = True)

df.hashtags = df.hashtags.apply(to_list)

df = df.reset_index(drop = True)


# In[ ]:


# A las columnas "photos", "retweet" y "url" las cambiamos por valores de 0 y 1
# 0 si no hay photo, url o si el tweet no es retweet
# 1 si hay photo, url o si el tweet es retweet

df.photos = df.photos.apply(lambda x : 1 if x != "[]" else 0)
df.retweet = df.retweet.apply(lambda x : 1 if x == "True" else 0)
df.urls = df.urls.apply(lambda x : 1 if x != "[]" else 0)


# In[ ]:


# Columnas de tiempo

df["month"] = df.date.apply(lambda x : int(x[5 : 7]))
df["day"] = df.date.apply(lambda x : int(x[-2:]))

df["hour"] = df.time.apply(lambda x : int(x[:2]))
df["minute"] = df.time.apply(lambda x : int(x[3:5]))
df["second"] = df.time.apply(lambda x : int(x[6:]))


# In[ ]:


# Columnas de interaccion:
# "mentions_count" : cuenta cuantas mentions hay en el tweet
# "reply_to_count" : cuenta a cuantas personas le hace respuesta el tweet
# "hashtags_count" : cuenta cuantos hashtags hay en el tweet

# "interaccion" : es la summa de las 3 columnas anteriores

df["mentions_count"] = [len(mention.split(",")) if type(mention) == str else 0 for mention in df.mentions]
df["reply_to_count"] = [len(reply.split(","))   if type(reply)   == str else 0 for reply   in df.reply_to]
df["hashtags_count"] = [len(hashtag.split(",")) if type(hashtag) == str else 0 for hashtag in df.hashtags]

df["interaccion"] = [rt + re + lk for rt, re, lk in zip(df.retweets_count, df.replies_count, df.likes_count)]


# In[ ]:


# Elimina las filas donde la fecha es NaN

indices_todrop = list()
for num, time in enumerate(df.time):
    if type(time) != str:
        indices_todrop.append(num)
        
df.drop(indices_todrop, inplace = True)

df = df.reset_index(drop = True)


# In[ ]:


# Filtro por el dia 24 o 25

FECHA = 25

df = df[df.day == FECHA]
df = df.reset_index(drop = True)
print(df.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Eliminio las filas que no tengan texto en el tweet\n\ntweet_na = []\nfor num, tweet in enumerate(df.tweet):\n    if type(tweet) != str:\n        tweet_na.append(num)\ndf.drop(tweet_na, inplace = True)\ndf = df.reset_index(drop = True)')


# In[ ]:


#df.to_csv("tweets_24_notendencias_preprocesado.csv", sep = ";", index = False)


# # TERMINA PREPROCESAMIENTO

# # LISTA DE PALABRAS Y HASHTAGS TENDENCIAS (PARA FILTRAR)

# In[ ]:


# Cargo las tendencias de ese dia

tendencias = []
with open("C:/Users/Daniel/Desktop/csv/dia 25/trends/dia 25 tendencias.txt", "r", encoding = "UTF-8") as f:
    tendencias.extend(f.readlines())
    
tendencias = [t[:-1].strip("\t") for num, t in enumerate(tendencias) if num != len(tendencias) - 1]

df_tendencias = pd.DataFrame(tendencias, columns = ["trends"])
df_tendencias = df_tendencias.trends.unique()
df_tendencias = pd.DataFrame(df_tendencias, columns = ["trends"])
solo_tendencias = list(df_tendencias.trends.unique())


# In[ ]:


# Lista de palabras tendencias y hashtags tendencias

hashtags_tendencias = [t for t in solo_tendencias if t[0] == "#"]
hashtags_tendencias_sin_numeral = [t.strip("#").lower() for t in solo_tendencias if t[0] == "#"]

palabras_tendencias = [t.strip("\t") for t in solo_tendencias if t[0] != "#"]
palabras_tendencias_lower = [t.strip("\t").lower() for t in solo_tendencias if t[0] != "#"]

print("hashtags_tendencias:", len(hashtags_tendencias))
print("hashtags_tendencias_sin_numeral:", len(hashtags_tendencias_sin_numeral))

print("palabras_tendencias:", len(palabras_tendencias))
print("palabras_tendencias_lower:", len(palabras_tendencias_lower))


# # FUNCIONES ESPECIALES

# In[ ]:


def f_hashtags_no_tendencias(df_aux):
    """Retorna un diccionario con los hashtags no tendencias que mas se repiten"""
    # Cuento cuantos hashtags hay en el df y me quedo con los mas repetidos
    hashtags_no_tendencias = list()
    for h in df_aux.hashtags:
        for hashtag in h.split(","):
            if hashtag not in hashtags_tendencias and hashtag != "":
                hashtags_no_tendencias.append(hashtag)

    hashtags_no_tendencias = Counter(hashtags_no_tendencias).most_common()
    hashtags_no_tendencias = {h[0] : h[1] for h in hashtags_no_tendencias}

    #print("Numero de hashtasg no tendencia:", len(hashtags_no_tendencias))

    return hashtags_no_tendencias


# In[ ]:


def elimina_hashtags_tendencias(df_aux, hashtags_tendencias_sin_numeral):
    """Elimina las filas que tengan hashtags tendencias"""
    # Saco los indices de las filas que tengan hashtags tendencias

    hashtags_indices = list()
    for num, h in enumerate(df_aux.hashtags):
        for hashtag in h.split(","):
            if hashtag.lower() in hashtags_tendencias_sin_numeral:
                hashtags_indices.append(num)

    #print("Cantidad de tweets con hashtags tendencias:", len(hashtags_indices))

    df_aux.drop(hashtags_indices, inplace = True)

    df_aux = df_aux.reset_index(drop = True)
    return df_aux


# In[ ]:


def elimina_palabras_tendencias(df_aux, palabras_tendencias_lower):
    """Elimina las filas que tengan palabras tendencias"""
    # Voy a quitar los tweets que tengan palabras claves tendencias

    palabras_indices = list()
    for num, tweet in enumerate(df_aux.tweet):
        for palabra in palabras_tendencias_lower:
            if tweet.lower().find(palabra) != -1:
                palabras_indices.append(num)

    #print(len(palabras_indices))

    df_aux.drop(palabras_indices, inplace = True)
    
    df_aux = df_aux.reset_index(drop = True)
    return df_aux


# In[ ]:


def limpieza(df_aux):
    """Realiza toda las limpieza de texto"""
    # Ahora voy a limpiar los tweets, para poder ver que palabras claves no tendencia se repiten mas

    df_aux.tweet = df_aux.tweet.apply(normalize)
    df_aux.tweet = df_aux.tweet.apply(deEmojify)
    df_aux.tweet = df_aux.tweet.apply(cleanTxt)
    df_aux.tweet = df_aux.tweet.apply(replace_punct)
    df_aux.tweet = df_aux.tweet.apply(replace_num)
    df_aux.tweet = df_aux.tweet.apply(quita_palabras_pequeñas)
    df_aux.tweet = df_aux.tweet.apply(tokenizador)
    df_aux.tweet = df_aux.tweet.apply(foo)
    return df_aux


# In[ ]:


def elimina_tweets_vacios(df_aux):
    """ELimina los tweets que no tengan texto despues de aplicar las limpiezas"""
    # Dropeo las filas de tweets que tengan texto ""

    tweet_vacios = []

    for num, tweet in enumerate(df_aux.tweet):
        if tweet == "":
            tweet_vacios.append(num)

    #print(len(tweet_vacios))        

    df_aux.drop(tweet_vacios, inplace = True)

    df_aux = df_aux.reset_index(drop = True)

    return df_aux


# In[ ]:


def f_palabras_no_tendencias(df_aux):
    """Retorna un diccionario con las palabras no tendencia que mas se repiten"""

    # Cuanto cuantos palabras hay en el df y me quedo con los mas repetidos

    palabras_no_tendencias = list()
    for p in df_aux.tweet:
        for palabra in p.split(" "):
            palabras_no_tendencias.append(palabra)

    palabras_no_tendencias = Counter(palabras_no_tendencias).most_common()
    palabras_no_tendencias = {h[0] : h[1] for h in palabras_no_tendencias}

    #print(len(palabras_no_tendencias))
    return palabras_no_tendencias


# In[ ]:


def get_df_h(df_aux, hashtags_no_tendencias):
    """Retorna un dataframe de solo hashtags, con una columna donde aparece la no tendencia"""
    df_h = df_aux[df_aux.hashtags != ""]

    df_h = df_h.reset_index(drop = True)

    df_h["trends"] = [[h if h in hashtags_no_tendencias else 0 for h in hashtag.split(",")] for hashtag in df_h.hashtags]

    df_h.trends = df_h.trends.apply(lambda x : [h for h in x if h != 0])

    indices_drop = list()
    for num, t in enumerate(df_h.trends):
        if t == []:
            indices_drop.append(num)

    df_h.drop(indices_drop, inplace = True)

    df_h = df_h.reset_index(drop = True)


    indices_para_clonar = list()
    for num, t in enumerate(df_h.trends):
        if len(t) > 1:
            indices_para_clonar.append(num)


    dic_indices = {indice : [len(trends), trends] for indice, trends in zip(indices_para_clonar, df_h.loc[indices_para_clonar].trends)}

    df_v = pd.DataFrame(columns = df_h.columns)

    for key in dic_indices.keys():
        for time in range(dic_indices[key][0]):
            df_d = pd.DataFrame(df_h.loc[key]).T
            df_d.drop(df_d.columns[-1], axis = 1, inplace = True)
            df_d["trends"] = dic_indices[key][1][time]
            df_v = pd.concat([df_v, df_d])



    df_h.drop(indices_para_clonar, inplace = True)

    df_h = df_h.reset_index(drop = True)

    df_h.trends = df_h.trends.apply(lambda x : x[0]) 

    df_h = pd.concat([df_h, df_v])
    df_h.trends = df_h.trends.apply(lambda x : "#" + x)

    #df_h.to_csv("H_6.csv", sep = ";", index = False)

    return df_h


# In[ ]:


def get_df_p(df_aux, palabras_no_tendencias):
    """Retorna un dataframe de solo palabras, con una columna donde aparece la no tendencia"""
    df_p = df_aux[df_aux.hashtags == ""]
    df_p = df_p.reset_index(drop = True)

    df_p["trends"] = [[p for p in palabra.split(" ") if p in palabras_no_tendencias] for palabra in df_p.tweet]

    indices_drop = list()
    for num, trend in enumerate(df_p.trends):
        if trend == []:
            indices_drop.append(num)

    df_p.drop(indices_drop, inplace = True)
    df_p = df_p.reset_index(drop = True)


    indices_multi = []
    for num, t in enumerate(df_p.trends):
        if len(t) >= 2:
            indices_multi.append(num)


    df_dup = df_p.iloc[indices_multi, :]
    df_dup = df_dup.reset_index(drop = True)


    indices_dup = df_dup.index.tolist()
    dic_indices = {indice : [len(trends), trends] for indice, trends in zip(indices_dup, df_dup.trends)}

    vacio = list()
    for key, value in dic_indices.items():
        prueba = np.tile([list(df_dup.iloc[key])], (value[0], 1))
        vacio.extend(prueba)

    df_multi = pd.DataFrame(vacio, columns = df_dup.columns)


    palabras = list()
    for i in range(len(df_dup.trends)):
        words = df_dup.trends[i]
        for j in range(len(words)):
            word = words[j]
            palabras.append(word)

    df_multi['trends'] = palabras


    df_uni = df_p[~(df_p.index.isin(indices_multi))]
    df_uni.trends = df_uni.trends.apply(lambda x : x[0])

    df_palabras = pd.concat([df_multi, df_uni])

    return df_palabras


# In[ ]:


def get_df_no_trends(df_h, df_p, start, num):
    """Retorna la concatenacion de los otros dos dataframes, ademas de una lista con la hora
       donde esa palabra o hashtag no tendencia se repite mas"""
    df_concat = pd.concat([df_h, df_p])
    
    df_summary = pd.DataFrame(df_concat.trends.value_counts()).reset_index()
    df_summary.columns = ["trend", "total_tweet"]
    
    df_summary["total_interaction"] = [df_concat[df_concat.trends == trend].interaccion.sum() for trend in df_summary.trend]
    
    df_summary = df_summary.sort_values("total_interaction", ascending = False).iloc[: num, :]
    
    no_trends = df_summary.trend.tolist()
    
    return df_concat, [[nt, start] for nt in no_trends]


# # TERMINA FUNCIONES ESPECIALES

# # BUCLE

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nto_df = list() # Lista que guarda la palabra o hashtag no tendencia en la hora que mas se repite\ndf_target = pd.DataFrame(columns = df.columns) #Dataframe con todos los dataframes concatenados (palabras y hashtags)\n\nstarts = [i for i in range(24)]\nfor start in starts:\n    \n    num = df_summary[df_summary.hour == start].shape[0] #Numero de no tendencia que va a coleccionar\n    \n    df_aux = df[df.hour == start] #Filtro por hora\n    df_aux = df_aux.reset_index(drop = True)\n     \n    # Limpieza\n    hashtags_no_tendencias = f_hashtags_no_tendencias(df_aux)\n    df_aux                 = elimina_hashtags_tendencias(df_aux, hashtags_tendencias_sin_numeral)\n    df_aux                 = elimina_palabras_tendencias(df_aux, palabras_tendencias_lower)\n    df_aux                 = limpieza(df_aux)\n    df_aux                 = elimina_tweets_vacios(df_aux)\n    palabras_no_tendencias = f_palabras_no_tendencias(df_aux)\n    \n    df_aux.hashtags = df_aux.hashtags.apply(str)\n    \n    df_h = get_df_h(df_aux, hashtags_no_tendencias)\n    df_p = get_df_p(df_aux, palabras_no_tendencias)\n    \n    df_concat, no_trends = get_df_no_trends(df_h, df_p, start, num)\n    \n    df_target = pd.concat([df_target, df_concat])\n    to_df.extend(no_trends)\n    \n    print(start)')


# In[ ]:


#df_target.to_csv("tweets_24_notedencias_preprocesado_labels.csv", sep = ";", index = False)


# In[ ]:


df_st = pd.DataFrame(to_df, columns = ["trend", "start_lifetime"])
#df_st.to_csv("tweets_24_start_lifetime_notendencias.csv", sep = ";", index = False)

