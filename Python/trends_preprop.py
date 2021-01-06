#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import json
import re
import string

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
    """Quita palabras de longitud menor a 3 del texto del tweet"""
    return " ".join([word for word in text.split(" ") if len(word) > 2])  


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv("C:/Users/Daniel/Desktop/csv/dia 24/trends/tweets_tendencias_24.csv")\ndf.head()')


# In[ ]:


# 1. Hace drop a las columnas de ids, husos horarios, url y traducciones
# 2. Filtra los tweets por idioma ("es")

columns_to_drop = ["conversation_id", "cashtags", "timezone", "user_id", "name", "near", "geo", "source",
                   "user_rt_id", "user_rt", "retweet_id", "retweet_date", "translate", "trans_src",
                   "trans_dest", "place", "quote_url", "thumbnail", "created_at", "id", "link"]

df.drop(columns_to_drop, axis = 1, inplace = True)

df = df[df.language == "es"]

df.drop("language", axis = 1, inplace = True)

df = df.reset_index().drop("index", axis = 1)


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

df = df.reset_index().drop("index", axis = 1)


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

df = df.reset_index().drop("index", axis = 1)


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

df = df.reset_index().drop("index", axis = 1)


# In[ ]:


# A las columnas "photos", "retweet" y "url" las cambiamos por valores de 0 y 1
# 0 si no hay photo, url o si el tweet no es retweet
# 1 si hay photo, url o si el tweet es retweet

df.photos = df.photos.apply(lambda x : 1 if x != "[]" else 0)
df.retweet = df.retweet.apply(lambda x : 1 if x == "True" else 0)
df.urls = df.urls.apply(lambda x : 1 if x != "[]" else 0)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Aplica las funciones de limpieza de texto\n\ndf.tweet = df.tweet.apply(normalize)\ndf.tweet = df.tweet.apply(deEmojify)\ndf.tweet = df.tweet.apply(cleanTxt)\ndf.tweet = df.tweet.apply(replace_punct)\ndf.tweet = df.tweet.apply(replace_num)\n\ndf.tweet = df.tweet.apply(tokenizador)\ndf.tweet = df.tweet.apply(foo)\ndf.tweet = df.tweet.apply(quita_palabras_pequeñas)')


# In[ ]:


# Columnas de tiempo

df["month"] = df.date.apply(lambda x : x[5 : 7])
df["day"] = df.date.apply(lambda x : x[-2:])

df["hour"] = df.time.apply(lambda x : x[:2])
df["minute"] = df.time.apply(lambda x : x[3:5])
df["second"] = df.time.apply(lambda x : x[6:])


# In[ ]:


# Columnas de interaccion:
# "mentions_count" : cuenta cuantas mentions hay en el tweet
# "reply_to_count" : cuenta a cuantas personas le hace respuesta el tweet
# "hashtags_count" : cuenta cuantos hashtags hay en el tweet

# "interaccion" : es la summa de las 3 columnas anteriores

df["mentions_count"] = [len(mention.split(",")) if type(mention) == str else 0 for mention in df.mentions]

df["reply_to_count"] = [len(reply.split(",")) if type(reply) == str else 0 for reply in df.reply_to]

df["hashtags_count"] =  [len(hashtag.split(",")) if type(hashtag) == str else 0 for hashtag in df.hashtags]

df["interaccion"] = [rt + re + lk for rt, re, lk in zip(df.retweets_count, df.replies_count, df.likes_count)]


# In[ ]:


df.head()


# In[ ]:


#df.to_csv("tweets_24_tendencia_preprocesado.csv", sep = ";", index = False)

