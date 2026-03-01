package jp.java.spark.kaggle.s6e2.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

public class LearningToPredict implements Serializable {

    public static CrossValidatorModel learning(Dataset<Row> optimizedTrainDf, String targetCol,
            String featuresCol, String predictionCol, String metricName) {

        // 1. 前処理ステージの作成
        List<PipelineStage> stages = createStringIndexerStages(optimizedTrainDf);

        // 2. 特徴量カラムの選定
        List<String> featureCols = new ArrayList<>();
        for (StructField field : optimizedTrainDf.schema().fields()) {
            String name = field.name();
            if (!name.equals(targetCol) && !name.equals("id")) {
                if (field.dataType().equals(DataTypes.StringType)) {
                    featureCols.add(name + "_indexed");
                } else {
                    featureCols.add(name);
                }
            }
        }

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols.toArray(new String[0]))
                .setOutputCol(featuresCol);
        stages.add(assembler);

        // ★追加：ターゲットカラム（正解ラベル）が文字列型の場合、"_indexed" カラムを使用するよう切り替え
        String actualLabelCol = targetCol;
        if (optimizedTrainDf.schema().apply(targetCol).dataType().equals(DataTypes.StringType)) {
            actualLabelCol = targetCol + "_indexed";
        }

        // ==========================================
        // ★ 提案4: GBTモデル自体の高速化オプション
        // ==========================================
        GBTClassifier gbt = new GBTClassifier()
                .setLabelCol(actualLabelCol)
                .setFeaturesCol(featuresCol)
                .setPredictionCol(predictionCol)
                .setCacheNodeIds(true) // ツリーの計算済みノード情報をメモリに保持（高速化）
                .setCheckpointInterval(10); // 10エポック（木）ごとに計算履歴を断ち切り、遅延を防ぐ

        Pipeline preprocessingPipeline = new Pipeline().setStages(stages.toArray(new PipelineStage[0]));
        Pipeline wholePipeline = new Pipeline().setStages(new PipelineStage[] { preprocessingPipeline, gbt });

        // 4. 評価器の設定
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol(actualLabelCol) // ★変更: targetCol から actualLabelCol に変更
                .setRawPredictionCol("rawPrediction")
                .setMetricName(metricName);

        // 5. パラメータグリッドの作成 (GBT用の強力なチューニング設定)
        // ローカルマシンのスペックに合わせて値は調整してください
        var paramGrid = new ParamGridBuilder()
                .addGrid(gbt.maxIter(), new int[] { 50, 100 }) // 木の数（エポック数）
                .addGrid(gbt.maxDepth(), new int[] { 4, 6 }) // 木の深さ（過学習制御）
                .addGrid(gbt.stepSize(), new double[] { 0.1, 0.05 }) // 学習率
                .build();

        // 6. CrossValidatorの設定
        CrossValidator cv = new CrossValidator()
                .setEstimator(wholePipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(3);

        System.out.println("Starting Cross Validation (Optimizing for " + metricName + ")...");
        CrossValidatorModel cvModel = cv.fit(optimizedTrainDf);

        // 7. 性能評価の出力
        double score = evaluator.evaluate(cvModel.transform(optimizedTrainDf));
        System.out.println("CV Training Result (" + metricName + "): " + score);

        return cvModel;
    }

    private static List<PipelineStage> createStringIndexerStages(Dataset<Row> df) {
        List<PipelineStage> stages = new ArrayList<>();
        for (StructField field : df.schema().fields()) {
            if (field.dataType().equals(DataTypes.StringType)) {
                String colName = field.name();
                stages.add(new StringIndexer()
                        .setInputCol(colName)
                        .setOutputCol(colName + "_indexed")
                        .setHandleInvalid("keep"));
            }
        }
        return stages;
    }
}