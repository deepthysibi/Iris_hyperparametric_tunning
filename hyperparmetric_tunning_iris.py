#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


iris=pd.read_csv("iris.csv")
df=iris.copy()
df.head()

df.drop('Id',axis=1,inplace=True)

#Summary of dataset
df.shape

df.info()

df.describe()

df.Species.value_counts()

df.isnull().sum()

#visualisations

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.boxplot(df['SepalLengthCm'],color='pink')

plt.subplot(2,2,2)
sns.boxplot(df['SepalWidthCm'],color='pink')

plt.subplot(2,2,3)
sns.boxplot(df['PetalLengthCm'],color='pink')

plt.subplot(2,2,4)
sns.boxplot(df['PetalWidthCm'],color='pink')

#outlier in sepal width remove it
df['SepalWidthCm']=df['SepalWidthCm'].clip(lower=df['SepalWidthCm'].quantile(0.05),
                    upper=df['SepalWidthCm'].quantile(0.95))

sns.boxplot(df['SepalWidthCm'])

df['SepalWidthCm'].shape

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(df['SepalLengthCm'],color='lightblue')

plt.subplot(2,2,2)
sns.distplot(df['SepalWidthCm'],color='lightblue')

plt.subplot(2,2,3)
sns.distplot(df['PetalLengthCm'],color='lightblue')

plt.subplot(2,2,4)
sns.distplot(df['PetalWidthCm'],color='lightblue')

# boxplot on each feature split out by species
df.boxplot(by="Species",figsize=(10,10))

# violinplots on petal-length for each species
sns.violinplot(data=df,x="Species", y="PetalLengthCm")

sns.pairplot(df,hue='Species')#diagonal kind=kde

# we can see that the species setosa is separataed from the other two across all feature combinations
sns.pairplot(df,hue='Species',diag_kind='hist')

#finding best model using gridsearch cv
#here we are checking logistic_regression,svm,randomforestclassifier

x=df.drop('Species',axis=1)
y=df['Species']

#seting models as parameters 
model_para={'svm':{'model':SVC(gamma='auto'),'params':{'C':[1,10,20],'kernel':['rbf','linear']}},
           'random_forest':{'model':RandomForestClassifier(),'params':{'n_estimators':[1,5,10]}},
           'logistic_reg':{"model":LogisticRegression(),'params':{'C':[1,5,10]}}}

score=[]
models=['svm','random_forest','logistic_reg']
for model_name in models:
    mp=model_para[model_name]
    gds=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=True)
    gds.fit(x,y)
    score.append({'model':model_name,'best_score':gds.best_score_,'best_params':gds.best_params_})

print(score)

#svm and logistic regression gives best performance
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=24)
model=SVC(C=1,gamma='auto')
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

ypred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
con=confusion_matrix(ypred,y_test)
print(con)
