#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


tendencias_24 = list()
with open("C:/Users/Daniel/Desktop/csv/dia 24/trends/tendencias por hora 24.txt", "r") as f:
    tendencias_24.extend(f.readlines())
    
    
tendencias_25 = list()
with open("C:/Users/Daniel/Desktop/csv/dia 25/trends/tendencias por hora 25.txt", "r") as f:
    tendencias_25.extend(f.readlines())


# In[12]:


tendencias_24 = [t.strip("\n, \t, ") for t in tendencias_24]

tendencias_25 = [t.strip("\n, \t, ") for t in tendencias_25]


# In[13]:


horas = ["0{}:00".format(i) if len(str(i)) < 2 else "{}:00".format(i) for i in range(24)]


# In[14]:


slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_24.index(horas[i]), tendencias_24.index(horas[i + 1]) - 1])
slicing.append([tendencias_24.index(horas[i + 1]), -1])


dict_24 = dict()
for num, slice_ in enumerate(slicing):
    if num != len(slicing) - 1:
        for t in tendencias_24[slice_[0] + 1: slice_[1]]:
            h = tendencias_24[slice_[0] : slice_[1]][0]
            if t not in dict_24:
                dict_24[t] = h
    else:
        for t in tendencias_24[slice_[0] + 1:]:
            h = tendencias_24[slice_[0] :][0]
            if t not in dict_24:
                dict_24[t] = h


# In[15]:


slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_25.index(horas[i]), tendencias_25.index(horas[i + 1]) - 1])
slicing.append([tendencias_25.index(horas[i + 1]), -1])


dict_25 = dict()
for num, slice_ in enumerate(slicing):
    if num != len(slicing) - 1:
        for t in tendencias_25[slice_[0] + 1: slice_[1]]:
            h = tendencias_25[slice_[0] : slice_[1]][0]
            if t not in dict_25:
                dict_25[t] = h
    else:
        for t in tendencias_25[slice_[0] + 1:]:
            h = tendencias_25[slice_[0] :][0]
            if t not in dict_25:
                dict_25[t] = h


# In[16]:


df_24 = pd.DataFrame([[key, value] for key, value in dict_24.items()], columns = ["trend", "start_lifetime"])
df_25 = pd.DataFrame([[key, value] for key, value in dict_25.items()], columns = ["trend", "start_lifetime"])


# In[8]:


#df_24.to_csv("trends_24_start_lifetime.csv", sep = ";", index = False)
#df_25.to_csv("trends_25_start_lifetime.csv", sep = ";", index = False)


# In[17]:


slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_24.index(horas[i]), tendencias_24.index(horas[i + 1]) - 1])
slicing.append([tendencias_24.index(horas[i + 1]), -1])

slicing = slicing[::-1]

dict_24_inv = dict()
for num, slice_ in enumerate(slicing):
    if num == 0:
        for t in tendencias_24[slice_[0] + 1:]:
            h = tendencias_24[slice_[0] :][0]
            if t not in dict_24_inv:
                dict_24_inv[t] = h    
    else:
        for t in tendencias_24[slice_[0] + 1: slice_[1]]:
            h = tendencias_24[slice_[0] : slice_[1]][0]
            if t not in dict_24_inv:
                dict_24_inv[t] = h


# In[10]:


slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_25.index(horas[i]), tendencias_25.index(horas[i + 1]) - 1])
slicing.append([tendencias_25.index(horas[i + 1]), -1])

slicing = slicing[::-1]

dict_25_inv = dict()
for num, slice_ in enumerate(slicing):
    if num == 0:
        for t in tendencias_25[slice_[0] + 1:]:
            h = tendencias_25[slice_[0] :][0]
            if t not in dict_25_inv:
                dict_25_inv[t] = h    
    else:
        for t in tendencias_25[slice_[0] + 1: slice_[1]]:
            h = tendencias_25[slice_[0] : slice_[1]][0]
            if t not in dict_25_inv:
                dict_25_inv[t] = h


# In[11]:


diff_24 = list()
for key1, value1 in dict_24.items():
    for key2, value2 in dict_24_inv.items():
        if key1 == key2:
            diff_24.append([key1, int(value2[:2]) - int(value1[:2])])


# In[12]:


diff_25 = list()
for key1, value1 in dict_25.items():
    for key2, value2 in dict_25_inv.items():
        if key1 == key2:
            diff_25.append([key1, int(value2[:2]) - int(value1[:2])])


# In[13]:


df_diff_24 = pd.DataFrame(diff_24, columns = ["trend", "lifetime"])
df_diff_25 = pd.DataFrame(diff_25, columns = ["trend", "lifetime"])


# In[14]:


#df_diff_24.to_csv("trends_24_lifetime.csv", sep = ";", index = False)
#df_diff_25.to_csv("trends_25_lifetime.csv", sep = ";", index = False)


# In[ ]:





# In[ ]:





# In[18]:


slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_24.index(horas[i]), tendencias_24.index(horas[i + 1]) - 1])
slicing.append([tendencias_24.index(horas[i + 1]), -1])

lista24 = list()
for slice_ in slicing:
    for trend in tendencias_24[slice_[0] : slice_[1]][1:]:
        lista24.append([trend, int(tendencias_24[slice_[0] : slice_[1]][0][:2])])

slicing = list()
for i in range(len(horas) - 1):
    slicing.append([tendencias_25.index(horas[i]), tendencias_25.index(horas[i + 1]) - 1])
slicing.append([tendencias_25.index(horas[i + 1]), -1])

lista25 = list()
for slice_ in slicing:
    for trend in tendencias_25[slice_[0] : slice_[1]][1:]:
        lista25.append([trend, int(tendencias_25[slice_[0] : slice_[1]][0][:2])])


# In[19]:


df_24 = pd.DataFrame(lista24, columns = ["trend", "hour"])
df_25 = pd.DataFrame(lista25, columns = ["trend", "hour"])


# In[20]:


df_24.to_csv("lista_tendencias_24_por_hora.csv", sep = ";", index = False)
df_25.to_csv("lista_tendencias_25_por_hora.csv", sep = ";", index = False)


# In[21]:


df_24


# In[22]:


df_25


# In[ ]:




