# Import the required Libraries
# For Linear Algerbra
import numpy as np

# For Data Processing
import pandas as pd

#Load the data set
df = pd.read_csv('E:/College/Semester 7/AI/Rain Detector App/weatherAUS.csv')

#Display the shape of the data set
print('Size of weather data frame is :',df.shape)

#Display data rows from 0 to 5
print(df[0:5])