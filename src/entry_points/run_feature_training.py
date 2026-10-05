from datetime import datetime

from pyspark.sql import SparkSession

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import sys, os

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, ".."))


from utils.config_loader import ConfigLoader
from feature_engineering.feature_table_preparer import FeatureTablePreparer
from feature_engineering.feature_label_splitter import FeatureLabelSplitter
from feature_engineering.feature_store_registrar import FeatureStoreRegistrar
from feature_engineering.training_set_builder import TrainingSetBuilder
from feature_engineering.splitter import TimeSeriesSplitter
from feature_engineering.encoder import CategoricalEncoder
from feature_engineering.preprocessed_writer import PreprocessedDataWriter
from modeling.data_preparer import ModelDataPreparer
from modeling.trainer import RandomForestTrainer
from modeling.registrar import ModelRegistrar
from modeling.champion_manager import ChampionChallengerManager



def main(config_path):
    spark = SparkSession.builder.getOrCreate()
    config = ConfigLoader(config_path).load()

    fs_cfg = config["feature_store"]
    split_cfg = config["split"]
    prep_cfg = config["preprocessing"]
    train_cfg = config["training"]

        # ---- Step 0: prepare feature table, split X/Y, register feature store ----
    preparer = FeatureTablePreparer(
        spark=spark,
        source_table=fs_cfg["gold_feature_table"],
        target_table=fs_cfg["v1_feature_table"],
        key_columns=fs_cfg["key_columns"]
    )
    preparer.run()

    splitter_fe = FeatureLabelSplitter(
        spark=spark,
        source_table=fs_cfg["v1_feature_table"],
        key_columns=fs_cfg["key_columns"],
        label_column=fs_cfg["label_column"],
        features_table=fs_cfg["features_table"],
        labels_table=fs_cfg["labels_table"]
    )
    splitter_fe.run()

    registrar_fs = FeatureStoreRegistrar(
        spark=spark,
        features_table=fs_cfg["features_table"],
        feature_store_table=fs_cfg["feature_store_table"],
        key_columns=fs_cfg["key_columns"],
        description=fs_cfg["description"]
    )
    registrar_fs.register()

    # ---- Step 1: build training set from feature store + labels ----
    builder = TrainingSetBuilder(
        spark=spark,
        labels_table=fs_cfg["labels_table"],
        feature_store_table=fs_cfg["feature_store_table"],
        key_columns=fs_cfg["key_columns"],
        label_column=fs_cfg["label_column"]
    )
    all_data_df = builder.build()
    print("Total rows: " + str(all_data_df.count()))

    # ---- Step 2: chronological split ----
    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    train_val_df, test_df, _ = splitter.split(all_data_df, split_ratio=split_cfg["train_ratio"])
    train_df, validation_df, _ = splitter.split(train_val_df, split_ratio=split_cfg["validation_ratio"])

    print("Train rows: " + str(train_df.count()))
    print("Validation rows: " + str(validation_df.count()))
    print("Test rows: " + str(test_df.count()))

    # ---- Step 3: one-hot encode (fit on train only) ----
    encoder = CategoricalEncoder(categorical_columns=prep_cfg["categorical_columns"])
    encoder.fit(train_df)

    train_encoded_df = encoder.transform(train_df)
    validation_encoded_df = encoder.transform(validation_df)
    test_encoded_df = encoder.transform(test_df)

    writer = PreprocessedDataWriter(spark=spark, schema_name=prep_cfg["ml_schema"])
    writer.write(train_encoded_df, "train_data")
    writer.write(validation_encoded_df, "validation_data")
    writer.write(test_encoded_df, "test_data")

    # ---- Step 4: prepare X, y for modeling ----
    preparer = ModelDataPreparer(
        spark=spark,
        train_table=prep_cfg["train_table"],
        validation_table=prep_cfg["validation_table"],
        test_table=prep_cfg["test_table"],
        label_column=fs_cfg["label_column"],
        exclude_columns=train_cfg["exclude_columns"]
    )
    X_train, y_train, X_validation, y_validation, X_test, y_test = preparer.prepare()

    # ---- Step 5: MLflow setup ----
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(train_cfg["experiment_path"])
    run_name = "rf_supply_chain_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Step 6: train, evaluate, log, register, promote ----
    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_param("model_type", "RandomForestRegressor (sklearn)")
        mlflow.log_param("num_trees", train_cfg["num_trees"])
        mlflow.log_param("max_depth", train_cfg["max_depth"])
        mlflow.log_param("label_column", fs_cfg["label_column"])
        mlflow.log_param("num_features", len(preparer.feature_columns))

        mlflow.set_tag("project", "oil_gas_supply_chain")
        mlflow.set_tag("problem_type", "time_series_regression")
        mlflow.set_tag("stage", "development")

        trainer = RandomForestTrainer(num_trees=train_cfg["num_trees"], max_depth=train_cfg["max_depth"])
        model = trainer.train(X_train, y_train)

        train_metrics = trainer.evaluate(X_train, y_train, "Train")
        validation_metrics = trainer.evaluate(X_validation, y_validation, "Validation")
        test_metrics = trainer.evaluate(X_test, y_test, "Test")

        mlflow.log_metric("train_rmse", train_metrics["rmse"])
        mlflow.log_metric("train_mae", train_metrics["mae"])
        mlflow.log_metric("train_r2", train_metrics["r2"])
        mlflow.log_metric("validation_rmse", validation_metrics["rmse"])
        mlflow.log_metric("validation_mae", validation_metrics["mae"])
        mlflow.log_metric("validation_r2", validation_metrics["r2"])
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_r2", test_metrics["r2"])

        signature_input = X_train.head(10)
        signature_output = model.predict(signature_input)
        signature = infer_signature(signature_input, signature_output)

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=signature_input,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        registrar = ModelRegistrar(
            model_name=train_cfg["model_name"],
            max_rmse=train_cfg["max_rmse"],
            min_r2=train_cfg["min_r2"]
        )
        new_version = registrar.register_if_good(
            model_uri=model_info.model_uri,
            rmse=validation_metrics["rmse"],
            r2=validation_metrics["r2"]
        )

        if new_version is not None:
            cc_manager = ChampionChallengerManager(model_name=train_cfg["model_name"])
            cc_manager.promote_challenger_if_better(
                new_version=new_version,
                new_rmse=validation_metrics["rmse"]
            )

        print("Run ID: " + run.info.run_id)

    print("Feature engineering and model training complete.")


if __name__ == "__main__":
    import sys
    config_path = "/Workspace/Users/hsshinnde0702@gmail.com/Oil_gas_mlops/oil_gas_mlops/config/config.yml"
    main(config_path)