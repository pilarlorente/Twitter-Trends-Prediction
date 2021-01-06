#!/usr/bin/env python
# coding: utf-8

# Short code to merge all the CSVs extracted from Twint per trend.

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[3]:


df = pd.DataFrame()


# In[4]:


os.chdir("....")
len(os.listdir())


# In[5]:


for csv in os.listdir():
    df1 = pd.read_csv(csv)
    df1["trend"] = [csv.split(".")[0] for i in range(df1.shape[0])]
    df = pd.concat([df, df1])


# In[6]:


os.chdir("C:/Users/Daniel/Desktop/copia tendencia daniel")
len(os.listdir())


# In[8]:


for csv in os.listdir():
    df1 = pd.read_csv(csv)
    df1["trend"] = [csv.split(".")[0] for i in range(df1.shape[0])]
    df = pd.concat([df, df1])


# In[9]:


os.chdir("C:/Users/Daniel/Desktop/copia2 tendencia daniel")
len(os.listdir())


# In[10]:


for csv in os.listdir():
    df1 = pd.read_csv(csv)
    df1["trend"] = [csv.split(".")[0] for i in range(df1.shape[0])]
    df = pd.concat([df, df1])


# In[11]:


os.chdir("C:/Users/Daniel/Desktop/ddd")
len(os.listdir())


# In[12]:


for csv in os.listdir():
    df1 = pd.read_csv(csv)
    df1["trend"] = [csv.split(".")[0] for i in range(df1.shape[0])]
    df = pd.concat([df, df1])


# In[13]:


df.drop_duplicates(inplace = True)


# In[14]:


df.to_csv("dia_24_tendencias.csv", sep = ";", index = False)

