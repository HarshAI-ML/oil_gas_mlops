from pyspark.sql import Window
from pyspark.sql.functions import col, lag, avg, dayofweek, month, sum as spark_sum, count


class GoldAggregation:

    def __init__(self, spark, silver_table, gold_table):
        self.spark = spark
        self.silver_table = silver_table
        self.gold_table = gold_table
        self.df = None
        self.gold_df = None

    def read_silver(self):
        self.df = self.spark.table(self.silver_table)

    def aggregate(self):
        self.gold_df = (
            self.df.groupBy("transaction_date", "product_name", "destination_city")
            .agg(
                avg("supplier_reliability_score").alias("avg_supplier_reliability_score"),
                spark_sum("ordered_quantity").alias("total_ordered_quantity"),
                spark_sum("demand_quantity").alias("total_demand_quantity"),
                spark_sum("available_inventory").alias("total_available_inventory"),
                avg("unit_price_usd").alias("avg_unit_price_usd"),
                spark_sum("product_cost_usd").alias("total_product_cost_usd"),
                spark_sum("transportation_cost_usd").alias("total_transportation_cost_usd"),
                spark_sum("total_cost_usd").alias("total_cost_usd"),
                spark_sum("expected_lead_time_days").alias("total_expected_lead_time_days"),
                spark_sum("actual_lead_time_days").alias("total_actual_lead_time_days"),
                spark_sum("delay_days").alias("total_delay_days"),
                spark_sum(col("is_delayed").cast("int")).alias("total_is_delayed"),
                spark_sum(col("is_stockout").cast("int")).alias("total_is_stockout"),
                avg("quality_score").alias("avg_quality_score"),
                count("*").alias("transaction_count")
            )
        )

    def write_gold(self):
        self.gold_df.write.format("delta") \
            .mode("overwrite") \
            .saveAsTable(self.gold_table)
        print("Gold table created: " + self.gold_table)

    def run(self):
        self.read_silver()
        self.aggregate()
        self.write_gold()


class GoldFeatureEngineering:

    def __init__(self, spark, gold_table, feature_table):
        self.spark = spark
        self.gold_table = gold_table
        self.feature_table = feature_table
        self.df = None
        self.feature_df = None

    def read_gold(self):
        self.df = self.spark.table(self.gold_table)

    def add_calendar_features(self):
        self.df = (
            self.df
            .withColumn("day_of_week", dayofweek(col("transaction_date")))
            .withColumn("month", month(col("transaction_date")))
        )

    def add_lag_and_rolling_features(self):
        window_spec = Window.partitionBy("product_name", "destination_city").orderBy("transaction_date")

        rolling_window_spec = (
            Window.partitionBy("product_name", "destination_city")
            .orderBy("transaction_date")
            .rowsBetween(-6, 0)
        )

        self.df = (
            self.df
            .withColumn("demand_lag_1", lag("total_demand_quantity", 1).over(window_spec))
            .withColumn("demand_lag_7", lag("total_demand_quantity", 7).over(window_spec))
            .withColumn("rolling_avg_7", avg("total_demand_quantity").over(rolling_window_spec))
        )

    def select_final_columns(self):
        self.feature_df = self.df.select(
            "transaction_date",
            "product_name",
            "destination_city",
            "total_demand_quantity",
            "avg_unit_price_usd",
            "total_available_inventory",
            "day_of_week",
            "month",
            "demand_lag_1",
            "demand_lag_7",
            "rolling_avg_7"
        )

    def write_feature_table(self):
        self.feature_df.write.format("delta") \
            .mode("overwrite") \
            .saveAsTable(self.feature_table)
        print("Feature table created: " + self.feature_table)

    def run(self):
        self.read_gold()
        self.add_calendar_features()
        self.add_lag_and_rolling_features()
        self.select_final_columns()
        self.write_feature_table()