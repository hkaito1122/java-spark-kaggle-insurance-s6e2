package jp.java.spark.kaggle.s6e2;

import java.io.Serializable;

import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import jp.java.spark.kaggle.s6e2.model.LearningToPredict;
import jp.java.spark.kaggle.s6e2.read.ReadInputFile;
import jp.java.spark.kaggle.s6e2.write.WriteOutputFile;

public class Main implements Serializable {
    final static String TRAIN_PATH = "/Users/hayashikaito/dev/java-spark-kaggle-insurance-s6e2/src/main/resources/input/train.csv";
    final static String TEST_PATH = "/Users/hayashikaito/dev/java-spark-kaggle-insurance-s6e2/src/main/resources/input/test.csv";
    final static String OUTPUT_DIR_PATH = "/Users/hayashikaito/dev/java-spark-kaggle-insurance-s6e2/src/main/resources/output/";
    final static String TARGET = "Heart Disease";
    final static String FEATURE = "features";

    public static void main(String[] args) {
        // SparkSessionの生成
        SparkSession spark = SparkSession.builder()
                .appName("KaggleS6E2SimplePipeline")
                .master("local[*]") // ローカル実行用
                .getOrCreate();

        // train,testデータの読み込み
        ReadInputFile readInputFile = new ReadInputFile();
        Dataset<Row> train_df = readInputFile.readCsv(spark, TRAIN_PATH);
        Dataset<Row> test_df = readInputFile.readCsv(spark, TEST_PATH);

        // 学習
        Transformer model = LearningToPredict.learning(train_df, TARGET, FEATURE, "prediction", "accuracy");

        // 予測
        Dataset<Row> predictions = model.transform(test_df)
                .withColumnRenamed("prediction_label", TARGET); // 出力用にカラム名を元に戻す

        // 出力ファイル生成
        WriteOutputFile.writeCsv(predictions, OUTPUT_DIR_PATH);
        spark.stop();

    }
}