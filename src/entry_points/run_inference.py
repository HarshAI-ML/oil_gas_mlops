import os
import sys

import mlflow
from pyspark.sql import SparkSession

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()


sys.path.append(os.path.join(current_dir, ".."))
from inference.champion_loader import ChampionLoader
from inference.prediction_writer import PredictionWriter
from inference.predictor import Predictor
from utils.config_loader import ConfigLoader


def main(config_path):
    spark = SparkSession.builder.getOrCreate()
    config = ConfigLoader(config_path).load()

    inf_cfg = config["inference"]

    mlflow.set_registry_uri("databricks-uc")

    # ---- Step 1: load the champion model ----
    loader = ChampionLoader(model_name=inf_cfg["model_name"])
    model_version = loader.get_version()
    model = loader.load_model()
    print("Loaded champion: version " + str(model_version))

    # ---- Step 2: predict on test data ----
    predictor = Predictor(
        spark=spark,
        model=model,
        test_table=inf_cfg["test_table"],
        label_column=inf_cfg["label_column"],
        exclude_columns=inf_cfg["exclude_columns"],
        id_columns=inf_cfg["id_columns"]
    )
    results_df = predictor.predict()

    # ---- Step 3: save predictions ----
    writer = PredictionWriter(spark=spark, output_table=inf_cfg["output_table"])
    writer.write(results_df, model_name=inf_cfg["model_name"], model_version=model_version)

    print("Inference complete.")


if __name__ == "__main__":
    import sys
    config_path = "/Workspace/Users/hsshinnde0702@gmail.com/Oil_gas_mlops/oil_gas_mlops/config/config.yml"
    main(config_path)
