"""
Two-sample Kolmogorov-Smirnov (KS) drift check: training data vs test data.

For every numeric feature it produces two tables that are ready for a dashboard:

1. ks_test_results
   One row per feature: KS statistic, p-value, alpha, critical value, decision,
   and the x value where the two cumulative curves are furthest apart.

2. ks_cdf_points  (long format)
   feature | dataset_type ('Train' / 'Test') | x | cumulative_proportion
   Plot x on the horizontal axis and cumulative_proportion on the vertical axis,
   one line per dataset_type, with a filter on feature.

How to read the result
- The KS statistic is the largest vertical gap between the two cumulative curves
  (0 = identical distributions, 1 = completely different).
- If p_value < alpha (0.05), the difference is statistically significant and the
  feature is flagged as drifted.

Null handling
- Lag features are null for the first rows of each series, so nulls are dropped
  per feature before testing. n_train / n_test show how many values were used.
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_TABLE = "oil_gas_mlops.ml_data.train_data"
TEST_TABLE = "oil_gas_mlops.ml_data.test_data"

RESULTS_TABLE = "oil_gas_mlops.ml_data.ks_test_results"
CDF_TABLE = "oil_gas_mlops.ml_data.ks_cdf_points"

FEATURES = [
    "avg_unit_price_usd",
    "total_available_inventory",
    "demand_lag_1",
    "demand_lag_7",
    "rolling_avg_7",
    "total_demand_quantity",  # the label, so you also see target drift
]

ALPHA = 0.05
MAX_CDF_POINTS = 300  # max x values per feature in the CDF table (keeps the dashboard light)


class KSDriftTest:

    def __init__(self, spark, train_table, test_table, features,
                 alpha=0.05, max_cdf_points=300):
        self.spark = spark
        self.train_table = train_table
        self.test_table = test_table
        self.features = features
        self.alpha = alpha
        self.max_cdf_points = max_cdf_points

    # ------------------------------------------------------------------
    # Small helpers (pure numpy / pandas, no Spark)
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(series):
        # Decimal / string -> float, drop nulls and non-finite values, sort ascending
        values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
        values = values[np.isfinite(values)]
        return np.sort(values)

    @staticmethod
    def _ecdf(sorted_values, x):
        # Cumulative proportion: share of values that are <= x
        return np.searchsorted(sorted_values, x, side="right") / len(sorted_values)

    def _critical_value(self, n_train, n_test):
        # Large-sample critical value of D at this alpha:
        # reject "same distribution" when the KS statistic is above this value
        c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2.0))
        return float(c_alpha * np.sqrt((n_train + n_test) / (n_train * n_test)))

    def _cdf_rows(self, feature, train_values, test_values, ks_location):
        # Both curves are evaluated on the SAME x values, so the dashboard can
        # draw them on one chart. Long datasets are thinned to max_cdf_points,
        # and the x where the gap is largest is always kept.
        pooled_x = np.unique(np.concatenate([train_values, test_values]))

        if len(pooled_x) > self.max_cdf_points:
            # one slot is reserved for ks_location, so the total never exceeds max_cdf_points
            positions = np.linspace(0, len(pooled_x) - 1, self.max_cdf_points - 1).round().astype(int)
            grid = np.unique(np.append(pooled_x[positions], ks_location))
        else:
            grid = pooled_x

        train_rows = pd.DataFrame({
            "feature": feature,
            "dataset_type": "Train",
            "x": grid,
            "cumulative_proportion": np.round(self._ecdf(train_values, grid), 6),
        })
        test_rows = pd.DataFrame({
            "feature": feature,
            "dataset_type": "Test",
            "x": grid,
            "cumulative_proportion": np.round(self._ecdf(test_values, grid), 6),
        })
        return pd.concat([train_rows, test_rows], ignore_index=True)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def load_data(self):
        train_pdf = self.spark.table(self.train_table).select(*self.features).toPandas()
        test_pdf = self.spark.table(self.test_table).select(*self.features).toPandas()
        print("Train rows: " + str(len(train_pdf)) + " | Test rows: " + str(len(test_pdf)))
        return train_pdf, test_pdf

    def compute(self, train_pdf, test_pdf):
        result_rows = []
        cdf_frames = []

        for feature in self.features:
            train_values = self._clean(train_pdf[feature])
            test_values = self._clean(test_pdf[feature])
            n_train = len(train_values)
            n_test = len(test_values)

            if n_train == 0 or n_test == 0:
                print("Skipping " + feature + ": no non-null values in train or test")
                continue

            ks = stats.ks_2samp(train_values, test_values, alternative="two-sided")

            # x where the two cumulative curves are furthest apart
            pooled_x = np.unique(np.concatenate([train_values, test_values]))
            gap = np.abs(self._ecdf(train_values, pooled_x) - self._ecdf(test_values, pooled_x))
            ks_location = float(pooled_x[np.argmax(gap)])

            drift_detected = bool(ks.pvalue < self.alpha)

            result_rows.append({
                "feature": feature,
                "ks_statistic": round(float(ks.statistic), 6),
                "p_value": float(ks.pvalue),
                "alpha": self.alpha,
                "critical_value": round(self._critical_value(n_train, n_test), 6),
                "drift_detected": drift_detected,
                "decision": ("Drift detected (p < " + str(self.alpha) + ")" if drift_detected
                             else "No significant drift"),
                "ks_location": ks_location,
                "n_train": int(n_train),
                "n_test": int(n_test),
            })
            cdf_frames.append(self._cdf_rows(feature, train_values, test_values, ks_location))

        results_pdf = pd.DataFrame(result_rows)
        cdf_pdf = pd.concat(cdf_frames, ignore_index=True) if cdf_frames else pd.DataFrame()
        return results_pdf, cdf_pdf

    def write(self, results_pdf, cdf_pdf, results_table, cdf_table):
        from pyspark.sql.functions import current_timestamp

        for pdf, table in [(results_pdf, results_table), (cdf_pdf, cdf_table)]:
            (
                self.spark.createDataFrame(pdf)
                .withColumn("computed_at", current_timestamp())
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(table)
            )
            print("Written: " + table)

    def run(self, results_table, cdf_table):
        train_pdf, test_pdf = self.load_data()
        results_pdf, cdf_pdf = self.compute(train_pdf, test_pdf)
        self.write(results_pdf, cdf_pdf, results_table, cdf_table)
        return results_pdf, cdf_pdf


def main():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    ks_test = KSDriftTest(
        spark=spark,
        train_table=TRAIN_TABLE,
        test_table=TEST_TABLE,
        features=FEATURES,
        alpha=ALPHA,
        max_cdf_points=MAX_CDF_POINTS,
    )
    results_pdf, _ = ks_test.run(results_table=RESULTS_TABLE, cdf_table=CDF_TABLE)

    print(results_pdf[["feature", "ks_statistic", "p_value", "drift_detected"]].to_string(index=False))


if __name__ == "__main__":
    main()
