#!/usr/bin/env python
# coding: utf-8

# # The Titanic Classification
# 
# 
# ## About Dataset:
# 
# Without a doubt, one of the most well-known shipwrecks in history is the sinking of the Titanic.
# 
# The presumably "unsinkable" RMS Titanic sank on the 15th of April, 1912, during her first voyage after hitting an iceberg. Unfortunately, there weren't enough lifeboats on board to accommodate everyone and 1502 out of 2224 passengers and staff died.
# 
# Even while survival required a certain amount of luck, it appears that some groups of people had a higher chance of living than others.
# 
# We're creating a predictive model that addresses the following question: "What kinds of people were more likely to survive?" utilizing traveller information, such as name, age, gender, socioeconomic status, etc.

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading the dataset
titanic = pd.read_csv("Titanic-Dataset.csv")
titanic


# In[3]:


#Reading the first five rows
titanic.head()


# In[4]:


#Reading the last five rows
titanic.tail()


# In[6]:


# To shows the number of rows and columns in the dataset
print("Total no of rows and columns: ")
titanic.shape


# In[10]:


#checking for columns
print("Column names are: ")
titanic.columns


# ## PassengerId - PassengerId. Survival - Survival in Titanic. pclass - Passenger class. Name - Name of the passenger. Sex - Sex. Age - Age in year. SibSp - Siblings/spouses aboard the Titanic. Parch - Parents/children aboard the Titanic. Ticket - ticket number. Fare - Passenger fare. Cabin - cabin number. Embarked - Port of embarkation (C-Cherbourg, Q- Queenstown, S- Southampton, O- others)

# # Data Preprocessing and Cleaning

# In[11]:


# check the data types
titanic.dtypes


# In[12]:


# check the duplicated value
titanic.duplicated().sum()


# In[13]:


#Checking the null values
titanic.isnull()


# In[14]:


sns.heatmap(titanic.isnull(),yticklabels= False, cbar=False,cmap="mako")


# In[15]:


#checking for number of null values 
nval = titanic.isnull().sum().sort_values(ascending=False)
nval = nval[nval>0]
nval


# In[16]:


#checking the what percentage missing columns value
titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)


# # # 1. There are high percentage of NaN values are present that is more then 75% so we will just drop the cabin column. 2. In the age column there are less than 20% NaN values are there so we will impute the mean age. 3. Only 2 NaN values are there in embarked column so we will impute the most frequent embarkation place.

# In[17]:


#drop the cabin column
titanic.drop(columns =  'Cabin', axis = 1, inplace=True)
titanic.columns


# In[18]:


sns.heatmap(titanic.isnull(),yticklabels= False, cbar=False,cmap="mako")


# In[19]:


# Filling Null Values in Age column with mean values of age column
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

#Filling the Null values in Embarked column with mode of embarked column

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[21]:


#To check the null values in heatmap
sns.heatmap(titanic.isnull(),yticklabels= False, cbar=False,cmap="mako")


# In[22]:


# Finding no. of unique values in each column of dataset
titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# In[23]:


titanic['Survived'].unique()


# In[24]:


titanic['Sex'].unique()


# In[25]:


titanic['Pclass'].unique()


# In[26]:


titanic['SibSp'].unique()


# In[28]:


titanic['Parch'].unique()


# In[29]:


titanic['Embarked'].unique()


# In[30]:


# we dont required the some columns as per our objective so drop it.
titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[31]:


sns.heatmap(titanic.isnull(),yticklabels= False, cbar=False,cmap="mako")


# In[32]:


# Information about the whole dataset
titanic.info()


# In[33]:


#Showing the information about non categorical column
titanic.describe()


# In[34]:


# showing info. about categorical columns
titanic.describe(include='O')


# # Data Visualization

# In[35]:


#Sex column
titanic['Sex'].value_counts()


# In[36]:


#Plotting count column for sex column
sns.countplot(x=titanic['Sex'])
plt.show()


# In[37]:


#plotting survived column
sns.set_style('whitegrid')
sns.countplot(x='Survived',data= titanic) 


# In[38]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue= 'Sex', data= titanic,palette= 'Spectral')


# In[39]:


#plotting passenger class survive
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue= 'Pclass', data= titanic,palette= 'rainbow')


# In[40]:


# Showing Distribution of Pclass Sex wise
sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])


# In[41]:


# age distribution
titanic['Age'].hist(bins= 30,color= 'cyan',ec='black')


# In[42]:


sns.kdeplot(x=titanic['Age'])
plt.show()


# In[43]:


sns.countplot(x='SibSp',data=titanic)


# In[44]:


# Showing Distribution of SibSp Survived Wise
sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[45]:


# Plotting Histplot for Dataset
titanic.hist(figsize=(10,10))
plt.show()


# In[46]:


# Plotting pairplot
sns.pairplot(titanic)
plt.show()


# # Label encoding

# In[47]:


from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for column in ['Sex','Embarked']:
    titanic[column] = le.fit_transform(titanic[column])

titanic.head()

# Sex Column

# 0 represents female
# 1 represents Male

# Embarked Column

# 0 represents C
# 1 represents Q
# 2 represents S


# In[48]:


# importing libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# # Selecting independent and dependent Features

# In[49]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[50]:


x.head(5)


# In[51]:


y.head(5)


# # Train Test Split

# In[52]:


print(891*0.10)


# In[53]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Creating Functions to compute Confusion Matrix, Classification Report and to generate Training and the Testing Score(Accuracy)

# In[54]:


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# # 1. Logistic Regression

# In[55]:


# Building the logistic Regression Model
lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[56]:


# Computing Training and Testing score
mscore(lr)


# In[57]:


# Generating Prediction
ypred_lr = lr.predict(x_test)
print(ypred_lr)


# In[58]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# # 2. Random Forest Classifier

# In[59]:


# Building the RandomForest Classifier Model
rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[60]:


# Computing Training and Testing score
mscore(rfc)


# In[61]:


# Generating Prediction
ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)


# In[62]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# # 3. Decision Tree Classifier

# In[63]:


# Building the DecisionTree Classifier Model
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[64]:


# Computing Training and Testing score
mscore(dt)


# In[65]:


# Generating Prediction
ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[66]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# # 4. K Neighbors Classifier

# In[67]:


# Building the knnClassifier Model
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[68]:


# Computing Training and Testing score
mscore(knn)


# In[69]:


# Generating Prediction
ypred_knn = knn.predict(x_test)
print(ypred_knn)


# In[70]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# # SVC

# In[71]:


# Building Support Vector Classifier Model
svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[72]:


# Computing Training and Testing score
mscore(svc)


# In[73]:


# Generating Prediction
ypred_svc = svc.predict(x_test)
print(ypred_svc)


# In[74]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# # 6. Ada Boost Classifier

# In[75]:


# Builing the Adaboost model
ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[76]:


# Computing the Training and Testing Score
mscore(ada_boost)


# In[77]:


# Generating the predictions
ypred_ada_boost = ada_boost.predict(x_test)
print(ypred_ada_boost)


# In[78]:


# Evaluate the model - confusion matrix, classification Report, Accuracy Score
cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[79]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','KNeighborsClassifier','SVC','Ada Boost Classifier'],
    'Score': [acc_lr,acc_rfc,acc_dt,acc_knn,acc_svc,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[80]:


sns.set_style("whitegrid")
plt.figure(figsize=(18,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette="magma" )
plt.show()


# Random forest and decision tree has highest accuracy.

# In[ ]:




