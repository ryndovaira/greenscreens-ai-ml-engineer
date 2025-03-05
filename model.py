class Model:
    def __init__(self):
        self.mean_rate = None

    def fit(self, x, y):
        self.mean_rate = y.mean()
        return self

    def predict(self, x):
        return [self.mean_rate] * len(x)
