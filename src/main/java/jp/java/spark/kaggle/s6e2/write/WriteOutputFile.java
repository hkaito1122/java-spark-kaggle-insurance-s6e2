package jp.java.spark.kaggle.s6e2.write;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class WriteOutputFile implements Serializable {
    static final String ID = "id";
    static final String TARGET = "Heart Disease";

    public static void writeCsv(Dataset<Row> df, String path) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter fomatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String dir = new String(now.format(fomatter));
        String output_path = path + dir + "/output";

        Dataset<Row> submission = df.select(ID, TARGET);

        submission.coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv(output_path);
    }
}
