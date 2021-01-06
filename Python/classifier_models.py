#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing
from sklearn.preprocessing import StandardScaler

#Regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn import svm
import xgboost as xgb

#Report
from sklearn.metrics import classification_report

#Validation
from sklearn.model_selection import GridSearchCV

#Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_confusion_matrix


# In[35]:


#Cargamos y leemos el csv

df1 = pd.read_csv("Q3_tweets_24_notrends_feautures.csv", sep = ";")
df2 = pd.read_csv("Q3_tweets_25_notrends_feautures.csv", sep = ";")
df3 = pd.read_csv("Q1Q3_tweets_24_trends_feautures.csv", sep = ";")
df4 = pd.read_csv("Q1Q3_tweets_25_trends_feautures.csv", sep = ";")

df1["target"] = 0
df2["target"] = 0
df3["target"] = 1
df4["target"] = 1

del(df1["Unnamed: 0"])
del(df2["Unnamed: 0"])
del(df3["Unnamed: 0"])
del(df4["Unnamed: 0"])


# In[36]:


df_train = pd.concat([df1, df3])
df_test = pd.concat([df2, df4])

df_train.drop("start_lifetime", axis = 1, inplace = True)
df_test.drop("start_lifetime", axis = 1, inplace = True)


# In[37]:


#train
X = np.asarray(df_train.iloc[:,1:-2])
y = np.asarray(df_train.target)

#test
X_test = df_test.iloc[:, 1:-2]
y_test = df_test.target


# In[38]:


df_train.iloc[:, 1:-2]


# In[39]:


df_train


# In[40]:


#Standard data
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)


# In[41]:


df_metrics = pd.DataFrame(columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])


# # RandomForestClassifier()

# In[10]:


#Calculamos los mejores parametros para el modelo
clf = RandomForestClassifier()
clfparam_grid = {"bootstrap"         : [True, False],
                 "max_depth"         : [10, 20, 30, 40, 50, 60, 70, None],
                 "max_features"      : ["auto", "sqrt"],
                 "min_samples_leaf"  : [1, 2, 4],
                 "min_samples_split" : [2, 5, 10],
                 "n_estimators"      : [200, 400, 600]}

clf_search = GridSearchCV(clf, param_grid = clfparam_grid, cv = 3, verbose = 2, n_jobs = -1)
 
model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "RandomForestClassifier",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])


# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# # LogisticRegression()

# In[11]:


#Calculamos los mejores parametros para el modelo
clf = LogisticRegression()
clfparam_grid = {"C"        : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 "penalty"  : ["l1", "l2"],
                 "max_iter" : list(range(100,800,100)),
                 "solver"   : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}

clf_search = GridSearchCV(clf, param_grid = clfparam_grid, refit = True, verbose = 3, cv = 5, n_jobs= -1)

# print("Mean Accuracy: %.3f" % clf_search.best_score_)
# print("Config: %s" % clf_search.best_params_)


model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "LogisticRegression",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])


# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# # GaussianNB()

# In[12]:


#Calculamos los mejores parametros para el modelo
clf = GaussianNB()
clfparam_grid = {"var_smoothing" : np.logspace(0, -9, num = 100)}
clf_search = GridSearchCV(clf, param_grid = clfparam_grid, cv = 5, verbose = 1, n_jobs= -1)


model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "GaussianNB",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])


# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# # KNeighborsClassifier()
# 

# In[13]:


#Calculamos los mejores parametros para el modelo
clf = KNeighborsClassifier()
clfparam_grid = {"n_neighbors": [3,4,5,6,10],
                 "weights"    : ["uniform", "distance"],
                 "metric"     : ["euclidean", "manhattan"]}

clf_search = GridSearchCV(clf, param_grid = clfparam_grid, verbose = 1, cv = 3, n_jobs = -1 )

model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "KNeighborsClassifier",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])


# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# # DecisionTreeClassifier()

# In[14]:


#Calculamos los mejores parametros para el modelo
clf = DecisionTreeClassifier()
clfparam_grid = {"criterion" : ["gini", "entropy"],
                 "max_depth" : [2,4,6,8,10,12]}

clf_search = GridSearchCV(clf, param_grid = clfparam_grid, refit = True, verbose = 3, cv=5, n_jobs= -1)

 
model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "DecisionTreeClassifier",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])

# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# In[18]:


importances = svm.feature_importances_


# # SVM

# In[31]:


#Calculamos los mejores parametros para el modelo
clf = svm.SVC()
clfparam_grid = {"C"           : [5, 10, 15],
                 "gamma"       : ["auto", 0.001, 0.00001, 0.000001],
                 "kernel"      : ["rbf", "poly", "linear"],
                 "shrinking"   : [True, False],
                 "probability" : [True, False]}

clf_search = GridSearchCV(clf, param_grid = clfparam_grid, refit = True, verbose = 3, cv = 5, n_jobs = -1)

model_result = clf_search.fit(X, y)
best_model = model_result.best_estimator_
final_model = best_model.fit(X,y)
yhat = final_model.predict(X_test)

print("Accuracy_score:", accuracy_score(y_test, yhat))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
print("------------------------------------------------------")
print(classification_report(y_test, yhat))

probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

row = {"Model"     : "SVM (Classifier)",
       "Accuracy"  : round(accuracy_score(y_test, yhat), 3),
       "F1-Score"  : round(f1_score(y_test,yhat), 3),
       "Precision" : round(precision_score(y_test, yhat), 3),
       "Recall"    : round(recall_score(y_test, yhat), 3),
       "AUC"       : round(roc_auc, 3)}

df_metrics = pd.concat([df_metrics, pd.DataFrame(row, index = [0])])

# Confussion Matrix
disp = plot_confusion_matrix(final_model, X_test, y_test,
                             display_labels=["NT", "T"],
                             cmap = plt.cm.Blues,
                             normalize = "true")
plt.show()

# Curva ROC
probs = final_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print("Area bajo la curva: ",auc(fpr, tpr))
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.ylabel("Sensibilidad")
plt.xlabel("1-Especificidad")
plt.show()


# In[16]:


df_metrics = df_metrics.sort_values("AUC", ascending = False)


# In[27]:


df_metrics.reset_index().drop('index', axis = 1)


# In[18]:


df_metrics.to_csv("metrics_models.csv", sep = ";", index = False)


# In[ ]:




