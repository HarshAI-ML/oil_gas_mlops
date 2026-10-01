from pyspark.sql.functions import col


class DataQualityChecker:

    def __init__(self, spark, source_df, config_table, source_table_name):
        self.spark = spark
        self.df = source_df
        self.config_table = config_table
        self.source_table_name = source_table_name
        self.rules = None

    def load_rules(self):
        rules_df = (
            self.spark.table(self.config_table)
            .filter(col("table_name") == self.source_table_name)
            .filter(col("is_active") == True)
        )
        self.rules = rules_df.collect()

    def apply_not_null(self, column_name):
        before_count = self.df.count()
        self.df = self.df.filter(col(column_name).isNotNull())
        after_count = self.df.count()
        print("NOT_NULL on " + column_name + ": dropped " + str(before_count - after_count) + " rows")

    def apply_unique(self, column_name):
        before_count = self.df.count()
        self.df = self.df.dropDuplicates([column_name])
        after_count = self.df.count()
        print("UNIQUE on " + column_name + ": dropped " + str(before_count - after_count) + " duplicate rows")

    def apply_range(self, column_name, check_params):
        params = self._parse_params(check_params)
        before_count = self.df.count()

        if "min" in params:
            self.df = self.df.filter(col(column_name) >= float(params["min"]))
        if "max" in params:
            self.df = self.df.filter(col(column_name) <= float(params["max"]))

        after_count = self.df.count()
        print("RANGE on " + column_name + ": dropped " + str(before_count - after_count) + " rows")

    def apply_allowed_values(self, column_name, check_params):
        params = self._parse_params(check_params)

        allowed_list = [
            value.strip().capitalize()
            for value in params["values"].split("|")
        ]

        before_count = self.df.count()
        self.df = self.df.filter(col(column_name).isin(allowed_list))
        after_count = self.df.count()
        print("ALLOWED_VALUES on " + column_name + ": dropped " + str(before_count - after_count) + " rows")

    def _parse_params(self, check_params):
        result = {}
        if check_params is not None:
            pairs = check_params.split(",")
            for pair in pairs:
                key_value = pair.split("=")
                result[key_value[0].strip()] = key_value[1].strip()
        return result

    def run_checks(self):
        for rule in self.rules:
            check_type = rule["check_type"]
            column_name = rule["column_name"]
            check_params = rule["check_params"]

            if check_type == "NOT_NULL":
                self.apply_not_null(column_name)
            elif check_type == "UNIQUE":
                self.apply_unique(column_name)
            elif check_type == "RANGE":
                self.apply_range(column_name, check_params)
            elif check_type == "ALLOWED_VALUES":
                self.apply_allowed_values(column_name, check_params)
            else:
                print("Unknown check_type: " + check_type + " (skipped)")

    def run(self):
        self.load_rules()
        self.run_checks()
        return self.df