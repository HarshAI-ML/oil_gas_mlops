from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


class TrainingSetBuilder:

    def __init__(self, spark, labels_table, feature_store_table, key_columns, label_column):
        self.spark = spark
        self.labels_table = labels_table
        self.feature_store_table = feature_store_table
        self.key_columns = key_columns
        self.label_column = label_column
        self.fe = FeatureEngineeringClient()
        self.training_set = None

    def build(self):
        base_df = self.spark.table(self.labels_table)

        feature_store_df = self.spark.table(self.feature_store_table)
        feature_names = [
            c for c in feature_store_df.columns
            if c not in self.key_columns
        ]

        feature_lookups = [
            FeatureLookup(
                table_name=self.feature_store_table,
                lookup_key=self.key_columns,
                feature_names=feature_names
            )
        ]

        self.training_set = self.fe.create_training_set(
            df=base_df,
            feature_lookups=feature_lookups,
            label=self.label_column
        )

        all_data_df = self.training_set.load_df()
        return all_data_df
