package jp.java.spark.kaggle.s6e2;

import java.io.Serializable;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import jp.java.spark.kaggle.s6e2.bi.ResearchData;
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

        // ★リファクタ: UDFの登録は全体の設定なので最初にやっておくと見通しが良いです
        spark.udf().register("extract_prob", (UDF1<Vector, Double>) vector -> vector.apply(1), DataTypes.DoubleType);

        // train,testデータの読み込み
        ReadInputFile readInputFile = new ReadInputFile();
        Dataset<Row> train_df = readInputFile.readCsv(spark, TRAIN_PATH);
        Dataset<Row> test_df = readInputFile.readCsv(spark, TEST_PATH);

        // ==========================================
        // ★ 提案1 & 2: パーティション最適化とキャッシュ
        // ==========================================
        Dataset<Row> optimizedTrainDf = train_df.repartition(8).cache();

        // testデータは学習と違い「1回しか計算されない」ため、実は cache() は不要です。
        // ただし、M4コアの並列処理の恩恵を受けるために repartition(8) は非常に有効です。
        Dataset<Row> optimizedTestDf = test_df.repartition(8).cache();

        // ==========================================
        // ★ 提案3: チェックポイントディレクトリの設定
        // ==========================================
        // ★リファクタ: アプリケーション全体で1つのディレクトリのみ指定すればOKです
        String checkpointPath = "tmp/spark-checkpoints";

        try {
            FileSystem fs = FileSystem.get(spark.sparkContext().hadoopConfiguration());
            Path path = new Path(checkpointPath);

            // ディレクトリが既に存在していれば、中身ごと削除
            if (fs.exists(path)) {
                fs.delete(path, true);
                System.out.println("Previous checkpoint directory cleaned up.");
            }
        } catch (Exception e) {
            System.err.println("Failed to clean up checkpoint directory: " + e.getMessage());
        }

        // チェックポイントディレクトリを設定 (SparkContextに対して1回だけ)
        spark.sparkContext().setCheckpointDir(checkpointPath);

        // bi
        ResearchData.analyze(spark, optimizedTrainDf);

        // ==========================================

        // // ==========================================
        // // ★ モデルの学習
        // // ==========================================
        // Transformer model = LearningToPredict.learning(optimizedTrainDf, TARGET,
        // FEATURE, "prediction", "areaUnderROC");

        // // 学習が終わったらTrainデータのメモリを解放
        // optimizedTrainDf.unpersist();

        // // ==========================================
        // // ★ 予測とデータ出力
        // // ==========================================
        // Dataset<Row> predictions = model.transform(optimizedTestDf);

        // Dataset<Row> submission = predictions.selectExpr(
        // "id",
        // "extract_prob(probability) AS `" + TARGET + "`");

        // // 出力ファイル生成 (※ ここで初めて実際の予測計算が走ります！)
        // WriteOutputFile.writeCsv(submission, OUTPUT_DIR_PATH);

        // // ★リファクタ: Testデータのメモリ解放は、必ずアクション（出力）が「終わった後」に行う
        // optimizedTestDf.unpersist();

        spark.stop();
    }
}