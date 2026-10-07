class ModelDataPreparer:

    def __init__(self, spark, train_table, validation_table, test_table,
                 label_column, exclude_columns):
        self.spark = spark
        self.train_table = train_table
        self.validation_table = validation_table
        self.test_table = test_table
        self.label_column = label_column
        self.exclude_columns = exclude_columns
        self.feature_columns = None

    def get_feature_columns(self, df):
        # X = every column except the label and the excluded columns
        self.feature_columns = [
            c for c in df.columns
            if c != self.label_column and c not in self.exclude_columns
        ]

    def to_pandas_xy(self, spark_df):
        pandas_df = spark_df.select(self.feature_columns + [self.label_column]).toPandas()

        # Decimal -> float (avoids the JSON serialization error)
        for c in pandas_df.columns:
            pandas_df[c] = pandas_df[c].astype(float)

        X = pandas_df[self.feature_columns]
        y = pandas_df[self.label_column]
        return X, y

    def prepare(self):
        train_df = self.spark.table(self.train_table)
        validation_df = self.spark.table(self.validation_table)
        test_df = self.spark.table(self.test_table)

        self.get_feature_columns(train_df)
        print("Feature columns (X): " + str(self.feature_columns))
        print("Label column (Y): " + self.label_column)

        X_train, y_train = self.to_pandas_xy(train_df)
        X_validation, y_validation = self.to_pandas_xy(validation_df)
        X_test, y_test = self.to_pandas_xy(test_df)

        return X_train, y_train, X_validation, y_validation, X_test, y_test
