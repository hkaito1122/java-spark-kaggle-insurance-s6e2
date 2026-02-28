package jp.java.spark.kaggle.s6e2.read;

import java.io.Serializable;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ReadInputFile implements Serializable {
    public Dataset<Row> readCsv(SparkSession spark, String path) {
        return spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);
    }

}
