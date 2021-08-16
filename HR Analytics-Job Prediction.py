#!/usr/bin/env python
# coding: utf-8

# In[636]:


import pandas as pd
DF=pd.read_csv("HR_comma_sep.csv")


# In[637]:


DF.head(10) 


# In[638]:


DF.info()


# In[639]:


DF.isnull().sum() 


# In[640]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set() 
from matplotlib import pyplot as plt


# In[641]:


sns.countplot(data=DF, x='Ayrildi');


# In[642]:


sns.countplot(data= DF,x="Maas");


# In[643]:


plt.figure(figsize=(5,5))
plt.title('Maaşa türüne göre Ayrılma Sayıları')
sns.countplot(data=DF, x='Maas', hue='Ayrildi');


# In[644]:


plt.figure(figsize=(10,10))
sns.countplot(data=DF,x="Departman"); 


# In[645]:


plt.figure(figsize=(8,8))
plt.title('Maaşa türüne göre Ayrılma Sayıları')
sns.countplot(data=DF, x='Departman', hue='Ayrildi');


# In[646]:


DF.corr()['Ayrildi'].sort_values()[:-1]


# In[647]:


plt.figure(figsize=(20,10))
plt.title('Memnuniyet Seviyesinin ve şirkette harcanan zamanın ayrılmaya etkisi')
sns.scatterplot(data=DF, x='Memnuniyet_Seviyesi',y='Sirkette_Harcanan_Zaman', hue='Ayrildi',s=100);


# In[648]:


DF


# In[649]:


Data = DF[['Memnuniyet_Seviyesi','Ortalama_Aylik_Saat','Sirkette_Harcanan_Zaman','Maas']]
Data.head()


# In[650]:


dummies = pd.get_dummies(Data.Maas)


# In[651]:


data_dummies = pd.concat([Data,dummies],axis='columns')


# In[652]:


data_dummies.head()


# In[653]:


data_dummies.drop('Maas',axis='columns',inplace=True)
data_dummies.head()


# In[654]:


X = data_dummies
X.head()


# In[666]:


y = DF.Ayrildi


# # LogisticRegression Algoritması

# In[667]:



from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
LogisticRegression()
Tahminler=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(confusion_matrix(y_test,Tahminler))
print(classification_report(y_test,Tahminler))


# In[668]:


Tahminler[3]


# # RandomForest Algoritması

# In[669]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
RF_Model=RandomForestClassifier(n_estimators=100)
RF_Model.fit(X_train,y_train)
Tahmin=RF_Model.predict(X_test)
from sklearn import metrics
print("Doğruluk=",metrics.accuracy_score(y_test,Tahmin))


# In[670]:


Tahmin[1:20]


# # Karar Ağacı Algoritması

# In[671]:


from sklearn.tree import DecisionTreeRegressor


# In[672]:


from sklearn.tree import DecisionTreeClassifier


# In[673]:


DT_Model=DecisionTreeClassifier(random_state=10)
DT_Model.fit(X_train,y_train)
Tahmin=DT_Model.predict(X_test)


# In[674]:


DT_Model.score(X_train,y_train)


# In[675]:


DT_Model.score(X_test,y_test)


# In[665]:


Tahmin[1:20]

