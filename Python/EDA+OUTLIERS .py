#!/usr/bin/env python
# coding: utf-8

# This notebook must be apply to every csv: Tweets_trends_preprop and tweets_notrends_preprop, due to trends and no trends has different 
# characteristics.
# 
# As we are working with two days, there are four datasets: tweets_24_trends_preprop, tweets_24_notrends_preprop, tweets_25_trends_preprop and tweets_25_notrends_preprop.
# 
# Day 24 will be our train data and day 25 the test one.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv("/.../tweets_24_trends_feautures.csv", sep = ";")


# In[14]:


#TIME FEAUTURES
df_t = df.iloc[:, 12:]


# In[15]:


df_t.columns


# In[16]:


#tweet count
df_tc = df_t.iloc[:, 0:7]
df_tc.columns


# In[17]:


#user count
df_uc = df_t.iloc[:, 21:28]
df_uc.columns


# In[18]:


#vel
df_v = df_t.iloc[:, 7:14]
df_v.columns


# In[19]:


#acc
df_a = df_t.iloc[:, 14:21]
df_a.columns


# # TIME FEAUTURES

# In[20]:


# Initialize the figure
plt.style.use("seaborn-darkgrid")
# create a color palette
palette = plt.get_cmap("Set1")


# In[21]:


df_t.T


# In[24]:


# multiple line plot
plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 0:16]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    plt.plot(range(7), df_t.T.iloc[0:7, column], marker = "", linewidth = 2, alpha = 1, label = column, color="blue") #tweet_count
    plt.plot(range(7), df_t.T.iloc[21:28, column], marker = "", linewidth = 2, alpha = 1, label = column, color="green") #user_count

    plt.legend(["tweet_count", "user_count"])
    plt.title(df.trend[num - 1])
    #same limits
    plt.xlim(0, 7)
    plt.ylim(-max(df.total_tweets)//7, max(df.total_tweets)//7)

#General title
plt.suptitle("TWEET COUNT & USER COUNT PER HOUR\nTrends from 0 to 15", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# We can see clearly the above graphs the difference between a paid hashtag and an organic hashtag. For example, #SuperligaOrangeLOL12 has not any tweet on the timeline. However, Twitter pins it as a trend.

# In[95]:


plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 0:16]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    #plt.plot(range(24), df_t.T.iloc[0:24, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "blue")
    plt.plot(range(7), df_t.T.iloc[7:14, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "red") #velocity
    plt.plot(range(7), df_t.T.iloc[14:21, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "orange") #acceleration
    plt.legend(["velocity", "acceleration"])
    plt.title(df.trend[num -1])
    #same limits
    plt.xlim(0, 7)
    plt.ylim((-max(df_t.iloc[:, 7:14].sum()))//10, (max(df_t.iloc[:, 7:14].sum()))//10)

#General title
plt.suptitle("VEC & ACC PER HOUR\nTrends from 0 to 15", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# In[28]:


# multiple line plot
plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 16:32]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    plt.plot(range(7), df_t.T.iloc[0:7, column], marker = "", linewidth = 2, alpha = 1, label = column, color="blue") #tweet_count
    plt.plot(range(7), df_t.T.iloc[21:28, column], marker = "", linewidth = 2, alpha = 1, label = column, color="green") #user_count

    plt.legend(["total_tweet", "user_count"])
    plt.title(df.trend[num +15])
    #same limits
    plt.xlim(0, 7)
    plt.ylim(-max(df.total_tweets)//10, max(df.total_tweets)//10)

#General title
plt.suptitle("TWEET COUNT & USER COUNT PER HOUR\nTrends from 16 to 31", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# In[97]:


plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 16:32]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    #plt.plot(range(24), df_t.T.iloc[0:24, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "blue")
    plt.plot(range(7), df_t.T.iloc[7:14, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "red") #velocity
    plt.plot(range(7), df_t.T.iloc[14:21, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "orange") #acceleration
    plt.legend(["velocity", "acceleration"])
    plt.title(df.trend[num +15])
    #same limits
    plt.xlim(0, 7)
    plt.ylim(-(max(df_t.iloc[:, 7:14].sum())//10), (max(df_t.iloc[:, 7:14].sum())//10))

#General title
plt.suptitle("VEC & ACC PER HOUR\nTrends from 16 to 31", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# In[26]:


# multiple line plot
plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 32:48]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    plt.plot(range(7), df_t.T.iloc[0:7, column], marker = "", linewidth = 2, alpha = 1, label = column, color="blue") #tweet_count
    plt.plot(range(7), df_t.T.iloc[21:28, column], marker = "", linewidth = 2, alpha = 1, label = column, color="green") #user_count

    plt.legend(["total_tweet", "user_count"])
    plt.title(df.trend[num +31])
    #same limits
    plt.xlim(0, 7)
    plt.ylim(-max(df.total_tweets)//4, max(df.total_tweets)//4)

#General title
plt.suptitle("TWEET COUNT & USER COUNT PER HOUR\nTrends from 32 to 47", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# In[27]:


plt.subplots(figsize = (15, 10))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)
num = 0
for column in df_t.T.iloc[0:7, 32:48]:
    num += 1
    #Find the right spot on the plot
    plt.subplot(4, 4, num)
 # Plot the lineplot and title
    #plt.plot(range(24), df_t.T.iloc[0:24, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "blue")
    plt.plot(range(7), df_t.T.iloc[7:14, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "red") #velocity
    plt.plot(range(7), df_t.T.iloc[14:21, column], marker = "", linewidth = 2, alpha = 1, label = column, color = "orange") #acceleration
    plt.legend(["velocity", "acceleration"])
    plt.title(df.trend[num +31])
    #same limits
    plt.xlim(0, 7)
    plt.ylim(-(max(df_t.iloc[:, 7:14].sum())//4), (max(df_t.iloc[:, 7:14].sum())//4))

#General title
plt.suptitle("VEC & ACC PER HOUR\nTrends from 32 to 47", fontsize = 22, fontweight = 0,
             color = "black", style = "italic", y = 1.02)


# Interacción

# In[104]:


plt.subplots(figsize = (30, 15))
g = sns.lineplot(df.trend, df.total_interaction[0:100])
g.set_xticklabels(df.trend[:],rotation = 90)


# # CLUSTERING
# 
# As a first aproximation, we are going to apply some clustering algorithms to identify classes and outliers.

# **KMEANS**

# In[105]:


from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist 


# In[106]:


X = df.iloc[:, 1:].values


# In[107]:


from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
X.shape


# INERCIAS

# In[108]:


inercias = [] 
for k in range(1, 10): 
    kmeans = KMeans(k)
    kmeans.fit(X)     
    inercias.append(kmeans.inertia_) 
inercias
plt.plot(range(1, 10), inercias, "bx-") 
plt.xlabel("Ks") 
plt.ylabel("Inercia") 
plt.show() #buscar el codo, es decir, donde varía la inercia


# In[109]:


kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
kmeans.labels_


# In[110]:


sns.scatterplot(df["total_tweets"].values, df["total_hashtags"].values, hue = kmeans.labels_, palette = "Set2")
plt.show()


# In[111]:


sns.scatterplot(df["total_tweets"].values, df["total_interaction"].values, hue = kmeans.labels_, palette = "Set2")
plt.show()


# **HIERARCHICAL**
# AGGLOMERATIVE

# In[112]:


from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics.cluster import homogeneity_score


# In[113]:


linkage = ["ward", "complete", "average", "single"]
for link in linkage:
    agglom = AgglomerativeClustering(n_clusters = 3, linkage = link)
    agglom.fit(X)
    print("Linkage tipo", link)
    print(agglom.labels_)
    print("****")
    print(homogeneity_score(agglom.labels_, kmeans.labels_))
    print("*****")


# **HIERARCHICAL**
# DENDROGRAMA

# In[114]:


from scipy.spatial import distance_matrix 
from scipy.cluster import hierarchy 
import pylab


# In[115]:


dist_matrix = distance_matrix(X,X)


# In[116]:


link=["single", "complete", "average", "weighted", "centroid", "median", "ward"]
for l in link:
    Z = hierarchy.linkage(dist_matrix, l)
    pylab.figure(figsize=(10, 10))
    pylab.title(l)
    dendro = hierarchy.dendrogram(Z)


# **DBSCAN**

# In[117]:


from sklearn.cluster import DBSCAN 


# DISTANCIAS

# In[118]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors = 2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis = 0)
distances = distances[:, 1]
plt.plot(distances)


# In[119]:


epsilon = 0.4
minimumSamples = 7
db = DBSCAN(eps = epsilon, min_samples = minimumSamples).fit(X)
labels = db.labels_
labels


# # OUTLIERS

# In[120]:


df.total_interaction.std()


# In[121]:


df.total_interaction.mean()


# In[122]:


df.total_interaction.mode()


# In[123]:


for col in df.columns[2:12]:
    plt.subplots()
    sns.distplot(df[col])
    plt.axvline(0              , 0, 1, color = "red")
    plt.axvline(df[col].std()  , 0, 1, color = "red")
    plt.axvline(df[col].std()*2, 0, 1, color = "yellow")
    plt.axvline(df[col].std()*3, 0, 1, color = "red")
    plt.show()


# As we can see, data from the second standard deviation is considered outlier.

# In[124]:


df[df.total_tweets > df.total_tweets.std()*2]


# In[31]:


#tweets with high interaction
df[df.total_interaction > df.total_interaction.std()*2]


# In[30]:


#correlation
plt.subplots(figsize = (7, 7))
sns.heatmap(df.iloc[:, 2:12].corr(), annot = True)


# In[127]:


#a few tweets and high interaction
df[(df.total_tweets < df.total_tweets.std()) & (df.total_interaction > df.total_interaction.std()*2)]


# In[128]:


#lot of tweets y low interaction
df[(df.total_tweets > df.total_tweets.std()*2) & (df.total_interaction < df.total_interaction.std())]


# In[129]:


#lot of tweets and high interaction
df[(df.total_tweets > df.total_tweets.std()*2) & (df.total_interaction > df.total_interaction.std()*2)]


# In[29]:


#boxplot
for col in df.columns[2:12]:
    plt.subplots()
    sns.boxplot(df[col], showfliers = True)


# In[131]:


Q1 = df['total_interaction'].quantile(0.25)
Q3 = df['total_interaction'].quantile(0.75)
IQR = Q3 - Q1    #IQR rango intercuartil


# In[31]:


#outliers trends
filter = (df['total_interaction'] >= Q1) & (df['total_interaction'] <= Q3 + 2.5 *IQR)
df_normal=df.loc[filter]


# In[52]:


df_normal=df_normal.reset_index(drop=True)


# In[53]:


import os
os.chdir('...')
df_normal.to_csv('Q1Q3_tweets_day*_trends_feautures.csv', sep=';')


# To remove no-trends outliers, we must not to quit data below Q1, as we are working with NO TRENDS tweets.

# In[79]:


#outliers no trends
filter2 = (df['total_interaction'] <= Q3)
df_up=df.loc[filte<r2]


# In[78]:


df_up.sort_values('total_interaction', ascending=False)['total_interaction']


# In[80]:


df_up=df_up.reset_index(drop=True)


# In[81]:


import os
os.chdir('...')
df_up.to_csv('Q3_tweets_day*_notrends_feautures.csv', sep=';')

