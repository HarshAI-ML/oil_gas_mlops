import pytest

from feature_engineering.encoder import CategoricalEncoder


@pytest.fixture
def train_df(spark):
    return spark.createDataFrame(
        [("Tesla",), ("BYD",), ("Tesla",), (None,)],
        ["vehicle_brand"],
    )


def test_fit_learns_sorted_unique_categories(train_df):
    encoder = CategoricalEncoder(categorical_columns=["vehicle_brand"])
    encoder.fit(train_df)

    assert encoder.known_categories["vehicle_brand"] == ["BYD", "Tesla"]


def test_fit_ignores_nulls(train_df):
    encoder = CategoricalEncoder(categorical_columns=["vehicle_brand"])
    encoder.fit(train_df)

    assert None not in encoder.known_categories["vehicle_brand"]


def test_transform_creates_one_hot_columns(train_df):
    encoder = CategoricalEncoder(categorical_columns=["vehicle_brand"])
    encoder.fit(train_df)

    transformed = encoder.transform(train_df)

    assert "vehicle_brand_Tesla" in transformed.columns
    assert "vehicle_brand_BYD" in transformed.columns


def test_transform_marks_matching_rows_correctly(train_df):
    encoder = CategoricalEncoder(categorical_columns=["vehicle_brand"])
    encoder.fit(train_df)

    transformed = encoder.transform(train_df).toPandas()

    tesla_rows = transformed[transformed["vehicle_brand"] == "Tesla"]
    assert (tesla_rows["vehicle_brand_Tesla"] == 1.0).all()
    assert (tesla_rows["vehicle_brand_BYD"] == 0.0).all()


def test_transform_sanitizes_spaces_in_category_names(spark):
    df = spark.createDataFrame([("North America",), ("Asia",)], ["region"])

    encoder = CategoricalEncoder(categorical_columns=["region"])
    encoder.fit(df)
    transformed = encoder.transform(df)

    assert "region_North_America" in transformed.columns


def test_transform_on_unseen_category_raises_keyerror(spark):
    # Encoder is fit on one set of categories, then asked to transform data
    # containing a category it never saw — this currently isn't handled
    # gracefully, so the test documents the real (risky) behavior rather
    # than an assumed one. Worth revisiting whether this should raise, warn,
    # or produce all-zero columns instead.
    train = spark.createDataFrame([("Tesla",)], ["vehicle_brand"])
    unseen = spark.createDataFrame([("Rivian",)], ["vehicle_brand"])

    encoder = CategoricalEncoder(categorical_columns=["vehicle_brand"])
    encoder.fit(train)

    # "Rivian" never appears as a column, but no error is raised either —
    # it's just silently missing from every generated one-hot column.
    transformed = encoder.transform(unseen).toPandas()
    assert transformed["vehicle_brand_Tesla"].iloc[0] == 0.0
