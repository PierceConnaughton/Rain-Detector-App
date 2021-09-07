# Import the required Libraries
# For Linear Algerbra
import numpy as np

# For Data Processing
import pandas as pd

#Load the data set
df = pd.read_csv('E:/College/Semester 7/AI/Rain Detector App/weatherAUS.csv')

# Checking for null values
print(df.count().sort_values())

# When we checked the null values in our data we could see that the first 4 values had over 40% null values so we removed them from our database.  
# We will also remove date and location as they are not necessary
# We will also remove the ‘RISK_MM’ variable because we want to predict ‘RainTomorrow’ and RISK_MM (amount of rain the next day) can leak some info to our model. 

# df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
# print(df.shape)

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

# Next we normalise our data in order to avoid being baised
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df.iloc[4:10]