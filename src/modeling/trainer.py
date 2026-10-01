import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RandomForestTrainer:

    def __init__(self, num_trees=100, max_depth=8):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.model = None

    def train(self, X_train, y_train):
        self.model = RandomForestRegressor(
            n_estimators=self.num_trees,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X, y, dataset_name):
        predictions = self.model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        print(dataset_name + " -> RMSE: " + str(rmse) + " | MAE: " + str(mae) + " | R2: " + str(r2))
        return {"rmse": rmse, "mae": mae, "r2": r2}
