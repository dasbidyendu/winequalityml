import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import pickle as pickle

wine_df = pd.read_csv('winequality-white.csv', delimiter=';')


feature = wine_df.drop('quality',axis=1)
label = wine_df['quality']
X_train,X_test,y_train,y_test = train_test_split(feature,label,test_size=.9)

model = LinearRegression()
print(X_train,y_train)
model.fit(X_train,y_train)
print(model.score(X_test,y_test)*100)

y_pred = model.predict(X_test)
print(y_pred)                    
print(X_test.describe())
filename = 'savedmodel.sav'
pickle.dump(model,open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))




