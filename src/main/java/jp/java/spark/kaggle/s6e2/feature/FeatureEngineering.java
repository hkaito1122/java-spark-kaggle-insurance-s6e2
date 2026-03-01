package jp.java.spark.kaggle.s6e2.feature;

import java.io.Serializable;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class FeatureEngineering implements Serializable {
    public static Dataset<Row> feature(SparkSession spark, Dataset<Row> df) {
        return df;
    }
}
