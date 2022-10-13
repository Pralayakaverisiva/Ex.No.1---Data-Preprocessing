# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
```
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train
```
##PROGRAM:
```
Register Number: 212219220027
Name:C.Lalitha parameswari
import pandas as pd
df=pd.read_csv("/content/Churn_Modelling.csv")
df.head()
df.isnull().sum()
df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
```

##OUTPUT:

## Dataset:

![image](https://user-images.githubusercontent.com/103946827/191925684-9e54fb38-5cab-4167-b40a-de85f61e131b.png)

## Discribing data:

![image](https://user-images.githubusercontent.com/103946827/191925837-f5c96de0-5e97-45a8-922a-7881ca5b9a7c.png)

## Normalized data:

![image](https://user-images.githubusercontent.com/103946827/191925966-7473c3a5-3437-4cf6-8f8a-7bd8dce4bd7a.png)


## X_train and Y_train values:

![image](https://user-images.githubusercontent.com/103946827/191926179-cb81c2dc-aba1-4f48-8ddd-1680de2498e1.png)


## X and Y values:

![image](https://user-images.githubusercontent.com/103946827/191926285-ed534f6e-666f-4d17-bd5d-b4ba8571916e.png)


## X_test and Y_test values:

![image](https://user-images.githubusercontent.com/103946827/191926429-6ee92b9f-5130-4da1-bda5-06f610a87b50.png)




##RESULT
Thus,the program has been done successfully and verified the output.
