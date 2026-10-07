from pyspark.sql.functions import col


class SilverTransformation:

    def __init__(self, spark, source_df, config_table, source_table_name):
        self.spark = spark
        self.df = source_df
        self.config_table = config_table
        self.source_table_name = source_table_name
        self.schema_map = None

    def load_schema_map(self):
        config_df = (
            self.spark.table(self.config_table)
            .filter(col("table_name") == self.source_table_name)
        )
        rows = config_df.select("column_name", "target_type").collect()
        self.schema_map = {row["column_name"]: row["target_type"] for row in rows}

    def apply_casting(self):
        for column_name, target_type in self.schema_map.items():
            if column_name in self.df.columns:
                self.df = self.df.withColumn(column_name, col(column_name).cast(target_type))

    def run(self):
        self.load_schema_map()
        self.apply_casting()
        return self.df
