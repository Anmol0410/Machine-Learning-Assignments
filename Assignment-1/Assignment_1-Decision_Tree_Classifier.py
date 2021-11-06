#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


# # Iris Dataset

# In[3]:


df1 = pd.read_csv('data/iris.data', header=None)
df1.head()


# In[4]:


features = df1.iloc[:, :-1]
labels = df1[4]


# In[5]:


le = LabelEncoder().fit(labels)


# In[6]:


le.classes_


# In[7]:


labels = le.transform(labels)
labels


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=7)


# In[34]:


dtclf = DecisionTreeClassifier()
dtclf.fit(X_train, y_train)


# In[35]:


dtclf.get_params()


# In[36]:


print(f"Accuracy Score of Training Set:  {accuracy_score(y_train, dtclf.predict(X_train))}")

y_pred_dtclf = dtclf.predict(X_test)
print(f"Accuracy Score of Test Set: {accuracy_score(y_test, y_pred_dtclf)}")

f1 = f1_score(y_test, y_pred_dtclf, average='weighted')
print(f"F1 Score of Test Set: {f1}")
      
print("Classification Report")    
print(classification_report(y_test, y_pred_dtclf))


# In[42]:


cv_sets = ShuffleSplit(n_splits=5, test_size=.2, random_state=8)
param_grid = {
    "max_depth" : [1,3,5,7,9,11,12],
    "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
}

grid_search = GridSearchCV(estimator=dtclf,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets)


# In[43]:


grid_search.fit(X_train, y_train)


# In[44]:


grid_search.best_params_


# In[45]:


grid_search.best_score_


# In[46]:


bestNB = grid_search.best_estimator_
print(f"Accuracy Score of Training Set:  {accuracy_score(y_train, bestNB.predict(X_train))}")

y_pred_bestNB = bestNB.predict(X_test)
print(f"Accuracy Score of Test Set: {accuracy_score(y_test, y_pred_bestNB)}")

f1 = f1_score(y_test, y_pred_bestNB, average='micro')
print(f"F1 Score of Test Set: {f1}")
      
print("Classification Report")    
print(classification_report(y_test, y_pred_bestNB))


# # Diabetes Dataset

# In[48]:


diabetes = pd.read_csv('data/diabetes.tab.txt', delimiter = "\t")


# In[49]:


diabetes


# In[50]:


features = diabetes.iloc[:, :-1]
labels = diabetes['Y']


# In[52]:


scaler = StandardScaler().fit(features)


# In[53]:


X_scaled = scaler.transform(features)


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=8) 


# In[55]:


dtclf1 = DecisionTreeClassifier()


# In[56]:


dtclf1.fit(X_train, y_train)


# In[58]:


print(f"Accuracy Score of Training Set:  {accuracy_score(y_train, dtclf1.predict(X_train))}")

y_pred_dtclf = dtclf1.predict(X_test)
print(f"Accuracy Score of Test Set: {accuracy_score(y_test, y_pred_dtclf)}")

f1 = f1_score(y_test, y_pred_dtclf, average='weighted')
print(f"F1 Score of Test Set: {f1}")
      
print("Classification Report")    
print(classification_report(y_test, y_pred_dtclf))


# # Breast Cancer Dataset

# In[60]:


data = pd.read_csv("data/breast-cancer-wisconsin.data", header=None)


# In[61]:


data = data[data[6] != '?']


# In[62]:


X = data.iloc[:, 1: -1]
y = data[10]


# In[63]:


y = y.replace(2, 0)
y = y.replace(4, 1)


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) 


# In[67]:


dtclf2 = DecisionTreeClassifier()
dtclf2.fit(X_train, y_train)


# In[68]:


print(f"Accuracy Score of Training Set:  {accuracy_score(y_train, dtclf2.predict(X_train))}")

y_pred_dtclf = dtclf2.predict(X_test)
print(f"Accuracy Score of Test Set: {accuracy_score(y_test, y_pred_dtclf)}")

f1 = f1_score(y_test, y_pred_dtclf, average='weighted')
print(f"F1 Score of Test Set: {f1}")
      
print("Classification Report")    
print(classification_report(y_test, y_pred_dtclf))


# In[78]:


cv_sets = ShuffleSplit(n_splits=5, test_size=.2, random_state=8)
param_grid = {
    "max_depth" : [1,3,5,7,9,11,12],
    "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
}

grid_search = GridSearchCV(estimator=dtclf2,
                            param_grid=param_grid,
                            scoring='f1',
                            cv=cv_sets)


# In[79]:


grid_search.fit(X_train, y_train)


# In[80]:


grid_search.best_params_


# In[81]:


bestNB = grid_search.best_estimator_
print(f"Accuracy Score of Training Set:  {accuracy_score(y_train, bestNB.predict(X_train))}")

y_pred_bestNB = bestNB.predict(X_test)
print(f"Accuracy Score of Test Set: {accuracy_score(y_test, y_pred_bestNB)}")

f1 = f1_score(y_test, y_pred_bestNB, average='micro')
print(f"F1 Score of Test Set: {f1}")
      
print("Classification Report")    
print(classification_report(y_test, y_pred_bestNB))

