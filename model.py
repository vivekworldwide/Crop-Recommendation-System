import pickle
import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Use absolute path to ensure the file is found regardless of where the script is run from
file_path = os.path.join(os.path.dirname(__file__), 'Crop_Recommendation.csv')
data = pd.read_csv(file_path)

X,Y= data.iloc[:, :-1], data["Crop"]

# Changed X_scaled to X as X_scaled was not defined
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier()
model.fit(X_train, y_train)


pickle.dump(model, open('model.pkl', 'wb'))