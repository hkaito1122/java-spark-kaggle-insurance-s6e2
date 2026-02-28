package jp.java.spark.kaggle.s6e2.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
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

    /**
     * 学習、交差検証、性能評価を一括で行う
     * 
     * @return CrossValidatorModel (前処理パイプラインを内包した学習済みモデル)
     */
    public static CrossValidatorModel learning(Dataset<Row> trainDf, String targetCol,
            String featuresCol, String predictionCol, String metricName) {

        // 1. 前処理ステージの作成 (StringIndexer)
        List<PipelineStage> stages = createStringIndexerStages(trainDf);

        // 2. 特徴量カラムの選定とVectorAssemblerの作成
        List<String> featureCols = new ArrayList<>();
        for (StructField field : trainDf.schema().fields()) {
            String name = field.name();
            // IDやターゲット以外のカラムを収集
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
        if (trainDf.schema().apply(targetCol).dataType().equals(DataTypes.StringType)) {
            actualLabelCol = targetCol + "_indexed";
        }

        // 3. モデルの定義 (Random Forest)
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol(actualLabelCol) // ★変更: targetCol から actualLabelCol に変更
                .setFeaturesCol(featuresCol)
                .setPredictionCol(predictionCol);

        // 4. 前処理のみのパイプラインを作成
        Pipeline preprocessingPipeline = new Pipeline().setStages(stages.toArray(new PipelineStage[0]));
        Pipeline wholePipeline = new Pipeline().setStages(new PipelineStage[] { preprocessingPipeline, rf });

        // 5. 評価器の設定
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(actualLabelCol) // ★変更: targetCol から actualLabelCol に変更
                .setPredictionCol(predictionCol)
                .setMetricName(metricName);

        // 6. パラメータグリッドの作成
        var paramGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] { 10, 20 })
                .build();

        // 7. CrossValidatorの設定
        CrossValidator cv = new CrossValidator()
                .setEstimator(wholePipeline) // パイプライン全体をセット
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(2);

        // 8. 学習の実行 (fit)
        System.out.println("Starting Cross Validation...");
        CrossValidatorModel cvModel = cv.fit(trainDf);

        // 9. 性能評価の出力
        // 内部で transform が走り、前処理 -> 予測 -> 評価が行われる
        double score = evaluator.evaluate(cvModel.transform(trainDf));
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