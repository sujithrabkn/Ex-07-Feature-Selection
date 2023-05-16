# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## CODE
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)  

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

    cols.remove(feature_with_p_max)
    
else:

    break
    selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()
```
## OUPUT
![7 1](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/cc0b83c4-3a6f-49fa-acd5-426ee8bb9060)

![7 2](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/3468e95c-43c3-4092-8f3b-1f79f40f8c90)

![7 3](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/157e37fa-0029-4d33-8b51-fb65ead73395)

![7 4](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/8072cfe5-9d80-468a-9438-d251c6b2d4b9)

![7 5](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/d585ae4c-6889-4069-9fe1-611b3698fbc3)

![7 6](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/f7a0f9c0-bcdd-465d-ae83-fb865a178615)

![7 7](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/a9ada2b0-013a-49b5-8b7d-241f71226f50)

![7 8](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/3e3526d2-e724-4849-bdb6-bd97044cdf77)

![7 9](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/84432dda-1a4b-4064-aa8d-6e97d0eca8b9)

![7 10](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/e0b795c8-baef-4b00-9a06-99bbf590fb47)

![7 11](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/4eeac9a0-a171-4112-9180-81c0395f07c9)

![7 12](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/712fa921-caa0-4a13-9444-0da641be872a)

![7 13](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/e0f21ccf-3b87-44fa-b82e-f6061fe0ff80)

![7 14](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/18de18f5-79ff-4a7e-b3ea-002345500290)

![7 15](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/66cd649a-8a11-4384-ade5-4382ee428a60)

![7 16](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/f685861f-bd6b-4663-bff5-94032f388151)

![7 17](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/1ef88b6c-920c-4974-ab64-b16484f3d588)

![7 18](https://github.com/sujithrabkn/Ex-07-Feature-Selection/assets/119477857/19920a1b-d07e-4649-9217-3adc70f7ce26)

## RESULT
The various feature selection techniques are performed on a dataset and saved the data to a file.
