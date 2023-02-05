import pandas as pd

class ModelInterface:

    def train(self, train_df: pd.DataFrame):
        pass

    def predict(self, n_days: int) -> list:
        pass
