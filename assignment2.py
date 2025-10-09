import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
testData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

 # Upper case before split, lower case after
Y = data['meal']
# make sure you drop a column with the axis=1 argument
X = data.drop(['meal', 'id', 'DateTime'], axis=1)

#x, xt, y, yt = train_test_split(X, Y, test_size=0.2, random_state=42)

#Since we actually ahve test data, no split, instead train on everything and test on given data
Yt = data['meal']
Xt = data.drop(['meal', 'id', 'DateTime'], axis=1)

model = XGBClassifier(n_estimators=1000, max_depth=15, learning_rate=1, objective='binary:logistic')

model.fit(X, Y)

pred = model.predict(Xt)

print(accuracy_score(Yt, pred)*100)
#Lower learning rate allows us to reduce the pace at which the model makes a new assumption based on the last tree