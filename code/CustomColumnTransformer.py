import numpy as np

class CustomColumnTransformer:

    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    # if cabin nr is C85 then rename it to C
    @staticmethod
    def rename_cabin(X):
        X['Cabin'] = X['Cabin'].map(lambda x: x[0] if not (x is np.nan) else "UNKNOWN")
        return X

    @staticmethod
    def impute_cabin(X):
        X['Cabin'] = X['Cabin'].map(
            lambda x: CustomColumnTransformer.get_deck_nr(x) if x == "UNKNOWN" or x == "T" else x)
        return X

    @staticmethod
    def get_deck_nr(pclass):
        if pclass == 1:
            return np.random.choice(['A', 'B', 'C', 'D', 'E'])
        elif pclass == 2:
            return np.random.choice(['D', 'E'])
        else:
            return np.random.choice(['F', 'G'])

    def get_feature_names_out(self, X):
        return ['Cabin']
