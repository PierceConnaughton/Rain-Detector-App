# Import the required Libraries
# For Linear Algerbra
import numpy as np

# For Data Processing
import pandas as pd

#Load the data set
df = pd.read_csv('E:/College/Semester 7/AI/Rain Detector App/weatherAUS.csv')

# When we checked the null values in our data we could see that the first 4 values had over 40% null values so we removed them from our database.  
# We will also remove date and location as they are not necessary
# We will also remove the ‘RISK_MM’ variable because we want to predict ‘RainTomorrow’ and RISK_MM (amount of rain the next day) can leak some info to our model. 

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date'],axis=1)
print(df.shape)

# Removing all null values from data
df = df.dropna(how='any')
print(df.shape)

# Here we are removing all the outliners from our data
# Outliners is a point that significantly differs from other observations.
# They usually occur due to miscaluclations while collecting data
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 3).all(axis=1)]
print(df.shape)
 
# Next we Change yes and no to 1 and 0 respectvely for RainToday and RainTomorrow variable
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#Using SelectKBest to get the top features!
# from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
# selector = SelectKBest(chi2, k=3)
# selector.fit(X, y)
# X_new = selector.transform(X)
# print(X.columns[selector.get_support(indices=True)])

# The important features are put in a data frame
df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
 
# To simplify computations we will use only one feature (Humidity3pm) to build the model
 
X = df[['Humidity3pm']]
y = df[['RainTomorrow']]

# We’ll be building classification models, by using the following algorithms:

# 1.Logistic Regression
# 2.Random Forest
# 3.Decision Tree

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
 
# # Calculating the accuracy and the time taken by the classifier
# t0=time.time()

# # Data Splicing
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# clf_logreg = LogisticRegression(random_state=0)

# # Building the model using the training data set
# clf_logreg.fit(X_train,y_train)
 
# # Evaluating the model using testing data set
# y_pred = clf_logreg.predict(X_test)
# score = accuracy_score(y_test,y_pred)
 
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Logistic Regression:',score)
# print('Time taken using Logistic Regression:' , time.time()-t0)


# Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
 
# # Calculating the accuracy and the time taken by the classifier
# t0=time.time()

# # Data Splicing
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)

# # Building the model using the training data set
# clf_rf.fit(X_train,y_train)
 
# # Evaluating the model using testing data set
# y_pred = clf_rf.predict(X_test)
# score = accuracy_score(y_test,y_pred)
 
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Random Forest Classifier:',score)
# print('Time taken using Random Forest Classifier:' , time.time()-t0)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
 
#Calculating the accuracy and the time taken by the classifier
t0=time.time()
#Data Splicing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_dt = DecisionTreeClassifier(random_state=0)
#Building the model using the training data set
clf_dt.fit(X_train,y_train)
 
#Evaluating the model using testing data set
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)
 
#Printing the accuracy and the time taken by the classifier
print('Accuracy using Decision Tree Classifier:',score)
print('Time taken using Decision Tree Classifier:' , time.time()-t0)
 
 
