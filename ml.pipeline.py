import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()

X = training_data.iloc[:,:-1].values

Y = training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_1 = sc.fit_transform(X_train)
X_test_1 = sc.transform(X_test)

## Building a Classification model (KNN)
from sklearn.neighbors import KNeighborsClassifier
# minkowski is for euclidean distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)

# Model training

classifier.fit(X_train_1, Y_train)

Y_pred = classifier.predict(X_test_1)
Y_prob = classifier.predict_proba(X_test_1)[:,1]

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test,Y_pred))

new_prediction = classifier.predict(sc.transform(np.array([[40,60000]])))
new_proba = classifier.predict_proba(sc.transform(np.array([[40,52000]])))[:,1]


import pickle
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file,'wb')) 
# wb - write in binary mode

scaler_file = "sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))
