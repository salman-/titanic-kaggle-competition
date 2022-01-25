import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

dt = pd.read_csv("./datasets/train.csv")
print(dt.columns)

categorical_features = ['Sex', 'Embarked', 'Pclass']
numerical_features = ['Age', 'SibSp', 'Fare', 'Parch']

numerical_pipeline = Pipeline(steps=[
    ('numerical_impute', SimpleImputer(missing_values=np.nan, strategy="median")),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('categorical_impute', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ('one_hot', OneHotEncoder())
])

column_transformer = ColumnTransformer(transformers=[
    ('categorical_pipeline', categorical_pipeline, categorical_features),
    ('numerical_pipeline', numerical_pipeline, numerical_features)], remainder="drop")

x = column_transformer.fit_transform(dt)

print(x)
