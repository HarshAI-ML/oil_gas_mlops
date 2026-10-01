from pyspark.sql.functions import current_timestamp

from data_engineering.silver_transform import SilverTransformation
from data_engineering.silver_dq import DataQualityChecker


def run_silver(spark, silver_config):
    # Step 1: Load bronze dataset
    bronze_df = spark.table(silver_config["input_table"])

    # Step 2: Drop unwanted columns
    bronze_df = bronze_df.drop(*silver_config["columns_to_drop"])

    # Step 3: Type casting
    transformer = SilverTransformation(
        spark=spark,
        source_df=bronze_df,
        config_table=silver_config["config_transformation_table"],
        source_table_name=silver_config["source_table_name"]
    )
    silver_df = transformer.run()

    # Step 4: DQ checks
    dq_checker = DataQualityChecker(
        spark=spark,
        source_df=silver_df,
        config_table=silver_config["config_dq_table"],
        source_table_name=silver_config["source_table_name"]
    )
    clean_df = dq_checker.run()

    # Step 5: Add silver-layer timestamp
    clean_df = clean_df.withColumn("silver_processed_ts", current_timestamp())

    # Step 6: Write to silver schema
    clean_df.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable(silver_config["output_table"])

    print("Silver table created: " + silver_config["output_table"])