package jp.java.spark.kaggle.s6e2.bi;

import java.io.Serializable;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ResearchData implements Serializable {

    public static void analyze(SparkSession spark, Dataset<Row> df) {
        df.printSchema();
        df.describe().show();

        System.out.println("Total count: " + df.count());

        // Check for null values in each column
        for (String col : df.columns()) {
            long nullCount = df.filter(df.col(col).isNull()).count();
            System.out.println(col + " null count: " + nullCount);
        }
    }

}
