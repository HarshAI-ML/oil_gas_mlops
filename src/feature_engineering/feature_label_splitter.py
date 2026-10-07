class FeatureLabelSplitter:

    def __init__(self, spark, source_table, key_columns, label_column,
                 features_table, labels_table):
        self.spark = spark
        self.source_table = source_table
        self.key_columns = key_columns
        self.label_column = label_column
        self.features_table = features_table
        self.labels_table = labels_table
        self.df = None

    def read_source(self):
        self.df = self.spark.table(self.source_table)

    def split_and_write(self):
        feature_columns = [c for c in self.df.columns if c != self.label_column]
        features_df = self.df.select(*feature_columns)

        labels_df = self.df.select(*self.key_columns, self.label_column)

        features_df.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(self.features_table)
        labels_df.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(self.labels_table)

        print("Features (X) table written: " + self.features_table)
        print("Labels (Y) table written: " + self.labels_table)

    def run(self):
        self.read_source()
        self.split_and_write()
