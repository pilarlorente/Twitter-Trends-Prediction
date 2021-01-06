#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from scipy.ndimage.interpolation import shift


# In[ ]:


get_ipython().run_cell_magic('time', '', '# df con los datos de las tendencias\ndf = pd.read_csv("C:/Users/Daniel/Desktop/por hora/tweets_24_notedencias_preprocesado_labels_hora.csv", sep = ";")\ndf = df.reset_index(drop = True)\n\n# df que tiene la hora de inicio de las tendencias\ndf_start_lifetime = pd.read_csv("C:/Users/Daniel/Desktop/por hora/tweets_24_start_lifetime_notendencias_hora.csv", sep = ";")')


# In[ ]:


df["trend"] = df.trends.apply(lambda x : x)
df.drop("trends", axis = 1, inplace = True)


# In[ ]:


# df_summary
# Es el df donde vamos a guardar la informacion de los tweets
# Primero agregamos los trends del dataframe df 
# Segundo agregamos la hora donde empieza a ser tendencia (merge)
# Vamos a estudiar el comportamiento de las tendencias 6 horas antes de que lo sean
# Por lo tanto vamos a quitar las tendencias entre las horas 0 y 5 ya que no tienen 6 horas de informacion

df_summary = df_start_lifetime.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nstarts = [i for i in range(24)]\n\nto_df = list()\nfor start in starts:\n    # Creo una lista con las horas desde 6 horas antes de start\n    hours = [start - i for i in range(6, -1, -1)]\n    \n    # Filtro el df_summary por las tendencias que comiencen en la hora "start"\n    df_aux = df_summary[df_summary.start_lifetime == start]\n    df_aux = df_aux.reset_index(drop = True)\n    \n    list_aux = list()\n    for trend in df_aux.trend:\n        \n        lista_filtro = list()\n        \n        # Calcula todas las variables de agregacion\n        df_filtro = df[(df.trend == trend) & (df.hour.isin(hours))]\n        \n        total_tweets      = df_filtro.shape[0]\n\n        total_hashtags    = df_filtro.hashtags_count.sum()\n\n        total_mentions    = df_filtro.mentions_count.sum()\n\n        total_reply_to    = df_filtro.reply_to_count.sum()\n\n        total_url         = df_filtro.urls.sum()\n\n        total_photo       = df_filtro.photos.sum()\n\n        total_retweets    = df_filtro.retweets_count.sum()\n\n        total_likes       = df_filtro.likes_count.sum()\n\n        total_replies     = df_filtro.replies_count.sum()\n\n        total_interaction = df_filtro.interaccion.sum()\n        \n        lista_filtro.extend([total_tweets, total_hashtags, total_mentions, total_reply_to, total_url,\n                             total_photo, total_retweets, total_likes, total_replies, total_interaction])\n        \n        # Crea las variables dependientes del tiempo\n        tweet_counts = [df[(df.trend == trend) & (df.hour == hour)].shape[0] for hour in hours]\n\n        tweet_counts = list(tweet_counts + shift(tweet_counts, 1))\n        vel          = list(tweet_counts - shift(tweet_counts, 1))\n        acc          = list(vel          - shift(vel         , 1))\n        \n        user_count = [df[(df.trend == trend) & (df.hour == hour)].username.unique().shape[0] for hour in hours]\n        user_count = list(user_count + shift(user_count, 1))\n\n        \n        lista_filtro.extend(tweet_counts + vel + acc + user_count)\n        \n        list_aux.append(lista_filtro)\n        \n    to_df.extend(list_aux)\n    print(start)')


# In[ ]:


columns_labels_agg = ["total_tweets", "total_hashtags", "total_mentions", "total_reply_to", "total_url",
                      "total_photo", "total_retweets", "total_likes", "total_replies", "total_interaction"]

count_labels      = ["tweet_count_{}{}".format(i, i + 1) for i in range(7)]
vel_labels        = ["tweet_vel_{}{}".format(i, i + 1)   for i in range(7)]
acc_labels        = ["tweet_acc_{}{}".format(i, i + 1)   for i in range(7)]
user_count_labels = ["user_count_{}{}".format(i, i + 1)  for i in range(7)]

columns_labels_time = count_labels + vel_labels + acc_labels + user_count_labels

columns_labels = columns_labels_agg + columns_labels_time


# In[ ]:


df_features = pd.DataFrame(to_df, columns = columns_labels)


# In[ ]:


df_summary = pd.concat([df_summary, df_features], axis = 1)


# In[ ]:


df_summary


# In[ ]:


#df_summary.to_csv("tweets_24_notendencias_variables_hora.csv", sep = ";", index = False)

