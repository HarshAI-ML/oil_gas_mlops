class PreprocessedDataWriter:

    def __init__(self, spark, schema_name):
        self.spark = spark
        self.schema_name = schema_name

    def write(self, df, table_name):
        full_table_name = self.schema_name + "." + table_name
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(full_table_name)
        print("Written: " + full_table_name)