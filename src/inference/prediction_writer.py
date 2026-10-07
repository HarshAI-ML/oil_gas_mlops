from pyspark.sql.functions import current_timestamp, lit


class PredictionWriter:

    def __init__(self, spark, output_table):
        self.spark = spark
        self.output_table = output_table

    def write(self, result_pandas_df, model_name, model_version):
        result_spark_df = self.spark.createDataFrame(result_pandas_df)

        result_spark_df = (
            result_spark_df
            .withColumn("model_name", lit(model_name))
            .withColumn("model_version", lit(str(model_version)))
            .withColumn("prediction_ts", current_timestamp())
        )

        result_spark_df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(self.output_table)

        print("Predictions saved to: " + self.output_table)
