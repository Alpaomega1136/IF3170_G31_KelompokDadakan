class NaiveBayesPreprocessor:
    def fit(self, X, y=None):
        self.columns_ =list(X.columns)
        
        return self

    def transform(self, X):
        if not hasattr(self,"columns_"):
            raise ValueError("Fit the preprocessor before transform")
        missing = set(self.columns_)-set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return X[self.columns_].copy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
