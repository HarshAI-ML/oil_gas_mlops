# from pyspark.ml.feature import StringIndexer, OneHotEncoder
# from pyspark.ml import Pipeline
# from pyspark.ml.functions import vector_to_array
# from pyspark.sql.functions import col


# class CategoricalEncoder:

#     def __init__(self, categorical_columns):
#         self.categorical_columns = categorical_columns
#         self.ohe_columns = [c + "_ohe" for c in categorical_columns]
#         self.fitted_pipeline = None
#         self.category_labels = {}

#     def build_pipeline(self):
#         indexers = [
#             StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep")
#             for c in self.categorical_columns
#         ]
#         encoders = [
#             OneHotEncoder(inputCol=c + "_index", outputCol=c + "_ohe")
#             for c in self.categorical_columns
#         ]
#         return Pipeline(stages=indexers + encoders)

#     def fit(self, train_df):
#         pipeline = self.build_pipeline()
#         self.fitted_pipeline = pipeline.fit(train_df)

#         for c in self.categorical_columns:
#             indexer_model = self.fitted_pipeline.stages[self.categorical_columns.index(c)]
#             self.category_labels[c] = indexer_model.labels

#         print("Encoder fitted on training data only")

#     def transform(self, df):
#         if self.fitted_pipeline is None:
#             raise ValueError("Call fit() before transform().")

#         transformed_df = self.fitted_pipeline.transform(df)

#         for c in self.categorical_columns:
#             ohe_col = c + "_ohe"
#             array_col = vector_to_array(col(ohe_col))
#             labels = self.category_labels[c]

#             for i, label in enumerate(labels):
#                 safe_label = label.replace(" ", "_")
#                 transformed_df = transformed_df.withColumn(
#                     c + "_" + safe_label,
#                     array_col[i]
#                 )

#             transformed_df = transformed_df.drop(ohe_col)

#         return transformed_df

import pandas as pd


class CategoricalEncoder:

    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.known_categories = {}

    def fit(self, train_df):
        train_pandas = train_df.select(self.categorical_columns).toPandas()
        for c in self.categorical_columns:
            self.known_categories[c] = sorted(train_pandas[c].dropna().unique().tolist())
        print("Encoder fitted on training data only")

    def transform(self, df):
        pandas_df = df.toPandas()

        for c in self.categorical_columns:
            categories = self.known_categories[c]
            for category in categories:
                safe_label = str(category).replace(" ", "_")
                pandas_df[c + "_" + safe_label] = (pandas_df[c] == category).astype(float)

        spark = df.sparkSession
        return spark.createDataFrame(pandas_df)