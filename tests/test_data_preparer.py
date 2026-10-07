from modeling.data_preparer import ModelDataPreparer


class FakeColumns:
    """
    Mimics the one thing get_feature_columns() actually reads off a Spark
    DataFrame — its .columns attribute — so that test doesn't need a real
    DataFrame (or Spark session) just to check a list comprehension.
    """
    def __init__(self, columns):
        self.columns = columns


def _make_preparer(label_column="target", exclude_columns=None, spark=None):
    return ModelDataPreparer(
        spark=spark,
        train_table="t",
        validation_table="v",
        test_table="te",
        label_column=label_column,
        exclude_columns=exclude_columns or [],
    )


def test_get_feature_columns_excludes_label_and_excluded():
    preparer = _make_preparer(exclude_columns=["id", "timestamp"])
    fake_df = FakeColumns(["id", "timestamp", "feature_a", "feature_b", "target"])

    preparer.get_feature_columns(fake_df)

    assert preparer.feature_columns == ["feature_a", "feature_b"]


def test_get_feature_columns_preserves_source_order():
    preparer = _make_preparer(exclude_columns=[])
    fake_df = FakeColumns(["z_feature", "a_feature", "target"])

    preparer.get_feature_columns(fake_df)

    assert preparer.feature_columns == ["z_feature", "a_feature"]


def test_get_feature_columns_with_no_exclusions():
    preparer = _make_preparer(exclude_columns=[])
    fake_df = FakeColumns(["feature_a", "feature_b", "target"])

    preparer.get_feature_columns(fake_df)

    assert preparer.feature_columns == ["feature_a", "feature_b"]


def test_to_pandas_xy_splits_features_and_label(spark):
    preparer = _make_preparer(spark=spark, exclude_columns=["id"])
    df = spark.createDataFrame(
        [(1, 10.0, 20.0, 1.0), (2, 15.0, 25.0, 0.0)],
        ["id", "feature_a", "feature_b", "target"],
    )
    preparer.get_feature_columns(df)

    X, y = preparer.to_pandas_xy(df)

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert "id" not in X.columns
    assert list(y) == [1.0, 0.0]


def test_to_pandas_xy_converts_all_columns_to_float(spark):
    # The source code comments this cast as a fix for a Decimal -> JSON
    # serialization error — this test locks in that behavior.
    preparer = _make_preparer(spark=spark, exclude_columns=[])
    df = spark.createDataFrame([(1, 2.5)], ["feature_a", "target"])
    preparer.get_feature_columns(df)

    X, y = preparer.to_pandas_xy(df)

    assert X["feature_a"].dtype == float
    assert y.dtype == float
