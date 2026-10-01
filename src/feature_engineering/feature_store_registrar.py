from databricks.feature_engineering import FeatureEngineeringClient


class FeatureStoreRegistrar:

    def __init__(self, spark, features_table, feature_store_table, key_columns, description):
        self.spark = spark
        self.features_table = features_table
        self.feature_store_table = feature_store_table
        self.key_columns = key_columns
        self.description = description
        self.fe = FeatureEngineeringClient()

    def register(self):
        df = self.spark.table(self.features_table)

        self.fe.create_table(
            name=self.feature_store_table,
            primary_keys=self.key_columns,
            df=df,
            description=self.description
        )
        print("Feature store table created: " + self.feature_store_table)