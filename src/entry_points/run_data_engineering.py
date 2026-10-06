import sys, os

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

sys.path.append(os.path.join(current_dir, ".."))
from pyspark.sql import SparkSession
#test line
# from ./utils.config_loader import ConfigLoader
# from ../utils.config_loader import ConfigLoader
# from ../data_engineering.ingestion import ExcelToBronzeIngestion
# from ../data_engineering.silver_main import run_silver
# from ../data_engineering.gold import GoldAggregation, GoldFeatureEngineering

from utils.config_loader import ConfigLoader
from data_engineering.ingestion import ExcelToBronzeIngestion
from data_engineering.silver_main import run_silver
from data_engineering.gold import GoldAggregation, GoldFeatureEngineering

#test line 
def main(config_path):
    spark = SparkSession.builder.getOrCreate()
    config = ConfigLoader(config_path).load()

    # ---- Bronze ----
    bronze_cfg = config["bronze"]
    bronze = ExcelToBronzeIngestion(
        spark=spark,
        source_path=bronze_cfg["source_path"],
        sheet_name=bronze_cfg["sheet_name"],
        target_catalog=bronze_cfg["target_catalog"],
        target_schema=bronze_cfg["target_schema"],
        target_table=bronze_cfg["target_table"],
        write_mode=bronze_cfg["write_mode"]
    )
    bronze.run()

    # ---- Silver ----
    run_silver(spark, config["silver"])

    # ---- Gold: aggregation ----
    gold_cfg = config["gold"]
    gold_agg = GoldAggregation(
        spark=spark,
        silver_table=gold_cfg["silver_table"],
        gold_table=gold_cfg["gold_table"]
    )
    gold_agg.run()

    # ---- Gold: feature engineering ----
    gold_fe = GoldFeatureEngineering(
        spark=spark,
        gold_table=gold_cfg["gold_table"],
        feature_table=gold_cfg["feature_table"]
    )
    gold_fe.run()

    print("Data engineering complete.")


if __name__ == "__main__":
    import sys
    config_path = "/Workspace/Users/hsshinnde0702@gmail.com/Oil_gas_mlops/oil_gas_mlops/config/config.yml"
    main(config_path)