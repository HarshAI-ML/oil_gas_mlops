class Predictor:

    def __init__(self, spark, model, test_table, label_column, exclude_columns, id_columns):
        self.spark = spark
        self.model = model
        self.test_table = test_table
        self.label_column = label_column
        self.exclude_columns = exclude_columns
        self.id_columns = id_columns

    def load_test_data(self):
        return self.spark.table(self.test_table).toPandas()

    def get_feature_columns(self, test_pandas_df):
        feature_columns = [
            c for c in test_pandas_df.columns
            if c != self.label_column and c not in self.exclude_columns
        ]

        trained_columns = list(self.model.feature_names_in_)
        missing = set(trained_columns) - set(feature_columns)
        extra = set(feature_columns) - set(trained_columns)

        if missing or extra:
            raise ValueError(
                "Test columns do not match training columns. "
                "Missing: " + str(missing) + " | Extra: " + str(extra)
            )

        return trained_columns

    def predict(self):
        test_pandas_df = self.load_test_data()
        feature_columns = self.get_feature_columns(test_pandas_df)

        for c in feature_columns + [self.label_column]:
            test_pandas_df[c] = test_pandas_df[c].astype(float)

        X_test = test_pandas_df[feature_columns]

        result_df = test_pandas_df[self.id_columns].copy()
        result_df["actual_demand"] = test_pandas_df[self.label_column]
        result_df["predicted_demand"] = self.model.predict(X_test)
        result_df["absolute_error"] = (result_df["actual_demand"] - result_df["predicted_demand"]).abs()

        return result_df