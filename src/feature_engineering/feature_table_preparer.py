from pyspark.sql.functions import col


class FeatureTablePreparer:

    def __init__(self, spark, source_table, target_table, key_columns):
        self.spark = spark
        self.source_table = source_table
        self.target_table = target_table
        self.key_columns = key_columns
        self.df = None

    def read_source(self):
        self.df = self.spark.table(self.source_table)
        self.df = self.df.withColumn("transaction_date", col("transaction_date").cast("date"))

    def validate_key_not_null(self):
        for key_col in self.key_columns:
            null_count = self.df.filter(col(key_col).isNull()).count()
            if null_count > 0:
                raise ValueError("Key column '" + key_col + "' has " + str(null_count) + " null values.")
        print("Key columns have no nulls: " + str(self.key_columns))

    def validate_key_uniqueness(self):
        total_rows = self.df.count()
        distinct_key_rows = self.df.select(*self.key_columns).distinct().count()

        if total_rows != distinct_key_rows:
            raise ValueError(
                "Key columns " + str(self.key_columns) + " are NOT unique. "
                "Total rows: " + str(total_rows) + ", distinct keys: " + str(distinct_key_rows)
            )
        print("Key uniqueness validated: " + str(total_rows) + " rows, all unique")

    def write_table(self):
        self.df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(self.target_table)
        print("Prepared table written: " + self.target_table)

    def run(self):
        self.read_source()
        self.validate_key_not_null()
        self.validate_key_uniqueness()
        self.write_table()