package jp.java.spark.kaggle.s6e2;

import java.io.Serializable;

import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import jp.java.spark.kaggle.s6e2.model.LearningToPredict;
import jp.java.spark.kaggle.s6e2.read.ReadInputFile;
import jp.java.spark.kaggle.s6e2.write.WriteOutputFile;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.linalg.Vector;

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
        Transformer model = LearningToPredict.learning(train_df, TARGET, FEATURE, "prediction", "areaUnderROC");
        // 予測
        Dataset<Row> predictions = model.transform(test_df);
        spark.udf().register("extract_prob", (UDF1<Vector, Double>) vector -> vector.apply(1), DataTypes.DoubleType);

        // 作成した extract_prob 関数を使って id と確率カラムを選択
        Dataset<Row> submission = predictions.selectExpr(
                "id",
                "extract_prob(probability) AS `" + TARGET + "`");
        // 出力ファイル生成
        WriteOutputFile.writeCsv(submission, OUTPUT_DIR_PATH);
        spark.stop();

    }
}