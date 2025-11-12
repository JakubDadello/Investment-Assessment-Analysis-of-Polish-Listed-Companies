import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# all_data object contains all variables (input features X and output target Y)
all_data = pd.read_csv("../data/initial_labeling_data.csv")

# X contains all input features 
X = all_data.iloc[:, 2:-1]

# Y contains all output features 
Y = all_data.iloc[:, -1:]

# split dataset X into numerical and categorical features
numeric_data = ['net_income', 'net_cash_flow', 'roe', 'roa', 'ebitda', 'cumulation'] 
categorical_data = ['sector']

# numerical data preprocessing: missing value imputation and standardization
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# categorical data preprocessing: missing value imputation and OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


#definition of an object performing comprehensive preprocessing transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_data),
    ('cat', categorical_transformer, categorical_data)
])
