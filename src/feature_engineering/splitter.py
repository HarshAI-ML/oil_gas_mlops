from pyspark.sql.functions import col


class TimeSeriesSplitter:

    def __init__(self, spark, date_column):
        self.spark = spark
        self.date_column = date_column

    def split(self, df, split_ratio=0.8):
        distinct_dates = [
            row[self.date_column]
            for row in df.select(self.date_column).distinct().orderBy(self.date_column).collect()
        ]

        cutoff_index = int(len(distinct_dates) * split_ratio) - 1
        cutoff_date = distinct_dates[cutoff_index]

        early_df = df.filter(col(self.date_column) <= cutoff_date)
        late_df = df.filter(col(self.date_column) > cutoff_date)

        return early_df, late_df, cutoff_date
