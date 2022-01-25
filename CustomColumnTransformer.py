class CustomColumnTransformer:

    def __init__(self, func):
        self.func = func

    def fit(self):
        return self

    def transform(self,X):
        return self.func(X)