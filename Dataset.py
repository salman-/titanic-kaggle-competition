import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from CustomColumnTransformer import CustomColumnTransformer


class Dataset:

    def __init__(self):
        self.dt = pd.read_csv("./datasets/train.csv")
        self.test_dt = pd.read_csv('./datasets/test.csv')

        categorical_features = ['Sex', 'Embarked', 'Pclass']
        numerical_features = ['Age', 'SibSp', 'Fare', 'Parch']

        cabin_pipeline = Pipeline(steps=[
            ('rename', CustomColumnTransformer(CustomColumnTransformer.rename_cabin)),
            ('impute_cabin_using_pclass', CustomColumnTransformer(CustomColumnTransformer.impute_cabin)),
            ('one_hot_encoder', OneHotEncoder())
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

        self.column_transformer = ColumnTransformer(transformers=[
            ('cabin_pipeline', cabin_pipeline, ['Cabin']),
            ('categorical_pipeline', categorical_pipeline, categorical_features),
            ('numerical_pipeline', numerical_pipeline, numerical_features)],
            remainder="drop")

        self.x = self.column_transformer.fit_transform(self.dt)
        self.test_dt = self.column_transformer.fit_transform(self.test_dt)
        print("Column names: ", self.column_transformer.get_feature_names_out())

    def get_train_test_dataset(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.dt["Survived"], test_size=0.8, shuffle=True)
        return x_train, x_test, y_train, y_test
