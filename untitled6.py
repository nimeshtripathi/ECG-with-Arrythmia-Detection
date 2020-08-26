import numpy as np 
import pandas as pd 
  
# import the data required 
data = pd.read_csv("ar.csv") 
print(data) 
from sklearn.preprocessing import OneHotEncoder 
  
# creating one hot encoder object with categorical feature 0 
# indicating the first column 
onehotencoder = OneHotEncoder(categorical_features = [0]) 
data = onehotencoder.fit_transform(data).toarray() 