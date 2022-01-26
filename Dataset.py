import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from CustomColumnTransformer import CustomColumnTransformer

class Dataset:

    def __init__(self):

        dt = pd.read_csv("./datasets/train.csv")
        categorical_features = ['Sex', 'Embarked', 'Pclass']
        numerical_features = ['Age', 'SibSp', 'Fare', 'Parch']

        cabin_pipeline = Pipeline(steps=[
            ('rename', CustomColumnTransformer(CustomColumnTransformer.rename_cabin)),
            ('impute_cabin_using_pclass',CustomColumnTransformer(CustomColumnTransformer.impute_cabin)),
            ('one_hot_encoder',OneHotEncoder())
        ])

        SimpleImputer.get_feature_names_out = (lambda self, names=None: self.feature_names_in_)

        numerical_pipeline = Pipeline(steps=[
            ('numerical_impute', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scale', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('categorical_impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('one_hot', OneHotEncoder())
        ])

        column_transformer = ColumnTransformer(transformers=[
            ('cabin_pipeline', cabin_pipeline, ['Cabin']),
            ('categorical_pipeline', categorical_pipeline, categorical_features),
            ('numerical_pipeline', numerical_pipeline, numerical_features)],
            remainder="drop")

        self.x = column_transformer.fit_transform(dt)

        print("Column names: ", column_transformer.get_feature_names_out())

