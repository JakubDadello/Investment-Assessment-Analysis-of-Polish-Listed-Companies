import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
from src.preprocessing import preprocessor, X, Y
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 


# splitting input and output data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# definition of a logistic regression model using a sequential processing chain (transforming pipeline)               
chain = Pipeline(steps = [                      
    ("preprocessing", preprocessor),            
    ("rf", RandomForestClassifier())                 
] )

# parameters for GridSearchCV
param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__criterion": ["gini", "entropy"], 
    "rf__max_depth": [None, 5, 10],
}

# definition of the GridSearchCV object
grid_search = GridSearchCV(chain, param_grid, scoring='accuracy', cv = 5)

# training a random forest model
grid_search.fit(X_train, y_train)

# model prediction 
y_pred = grid_search.predict(X_test)

# evaluation of model fit using the accuracy metric
accuracy_values = accuracy_score (y_test, y_pred)

# evaluation of model fit using the precision metric
precision_values = precision_score (y_test, y_pred, average='macro')

# evaluation of model fit using the recall metric
recall_values = recall_score (y_test, y_pred, average = 'macro')

# evaluation of model fit using the confusion matrix 
confusionmatrix_values = confusion_matrix(y_test, y_pred, labels = ['low', 'middle', 'high'])
