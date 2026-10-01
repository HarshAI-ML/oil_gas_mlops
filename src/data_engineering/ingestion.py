from pyspark.sql.functions import current_timestamp, lit


class ExcelToBronzeIngestion:
    '''ingestion class to extract excel file and transform into table form and store in bronze schema '''

    def __init__(self, spark, source_path, sheet_name, target_catalog, target_schema,
                 target_table, write_mode="overwrite", header=True, infer_schema=True):
        self.spark = spark
        self.source_path = source_path
        self.sheet_name = sheet_name
        self.target_catalog = target_catalog
        self.target_schema = target_schema
        self.target_table = target_table
        self.write_mode = write_mode
        self.header = header
        self.infer_schema = infer_schema
        self.df = None

    def read_source(self):
        
        self.df = (
            self.spark.read.format("excel")
            .option("header", self.header)
            .option("inferSchema", self.infer_schema)
            .option("headerRows", 1)
            .option("dataAddress", self.sheet_name)
            .load(self.source_path)
        )
        

    def add_metadata_columns(self):
        if self.df is None:
            raise ValueError("Call read_source() before add_metadata_columns().")

        self.df = (
            self.df
            .withColumn("ingestion_ts", current_timestamp())
            
        )
        

    def write_to_bronze(self):
        if self.df is None:
            raise ValueError("Call read_source() and add_metadata_columns() first.")

        full_table_name = self.target_catalog + "." + self.target_schema + "." + self.target_table

        (
            self.df.write.format("delta")
            .mode(self.write_mode)
            .saveAsTable(full_table_name)
        )
        print("Data written to " + full_table_name)

    def run(self):
        self.read_source()
        self.add_metadata_columns()
        self.write_to_bronze()

