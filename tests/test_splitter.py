from feature_engineering.splitter import TimeSeriesSplitter


def _make_date_df(spark, dates):
    return spark.createDataFrame([(d,) for d in dates], ["transaction_date"])


def test_split_respects_ratio(spark):
    dates = [f"2024-01-{day:02d}" for day in range(1, 11)]  # 10 distinct dates
    df = _make_date_df(spark, dates)

    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    early_df, late_df, cutoff_date = splitter.split(df, split_ratio=0.8)

    assert cutoff_date == "2024-01-08"
    assert early_df.count() == 8
    assert late_df.count() == 2


def test_split_is_chronological_not_positional(spark):
    # Rows are inserted out of order on purpose — the splitter must sort
    # by date itself, not trust input row order.
    dates = ["2024-01-05", "2024-01-01", "2024-01-03"]
    df = _make_date_df(spark, dates)

    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    early_df, _, _ = splitter.split(df, split_ratio=0.67)

    early_dates = sorted(r["transaction_date"] for r in early_df.collect())
    assert early_dates == ["2024-01-01", "2024-01-03"]


def test_split_produces_no_overlap(spark):
    dates = [f"2024-01-{day:02d}" for day in range(1, 21)]
    df = _make_date_df(spark, dates)

    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    early_df, late_df, _ = splitter.split(df, split_ratio=0.5)

    early_dates = {r["transaction_date"] for r in early_df.collect()}
    late_dates = {r["transaction_date"] for r in late_df.collect()}

    assert early_dates.isdisjoint(late_dates)


def test_split_covers_every_row(spark):
    dates = [f"2024-01-{day:02d}" for day in range(1, 16)]
    df = _make_date_df(spark, dates)

    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    early_df, late_df, _ = splitter.split(df, split_ratio=0.6)

    assert early_df.count() + late_df.count() == df.count()


def test_split_handles_duplicate_dates_as_one_distinct_value(spark):
    # Multiple rows can share the same date — distinct() should still
    # treat them as a single point when computing the cutoff.
    dates = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]
    df = _make_date_df(spark, dates)

    splitter = TimeSeriesSplitter(spark, date_column="transaction_date")
    early_df, late_df, cutoff_date = splitter.split(df, split_ratio=0.67)

    # 3 distinct dates, 67% -> index 1 -> "2024-01-02"
    assert cutoff_date == "2024-01-02"
    assert early_df.count() == 3  # both 01-01 rows + the one 01-02 row
    assert late_df.count() == 1
