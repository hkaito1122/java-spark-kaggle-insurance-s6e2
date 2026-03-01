package jp.java.spark.kaggle.s6e2.bi;

import java.io.File;
import java.io.PrintWriter;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ResearchData implements Serializable {

    // 出力先のファイルパス（環境に合わせて適宜修正してください）
    final static String OUTPUT_FILE_PATH = "/Users/hayashikaito/dev/java-spark-kaggle-insurance-s6e2/src/main/resources/tmp/data_profile_report.txt";

    public static void analyze(SparkSession spark, Dataset<Row> df) {
        StringBuilder report = new StringBuilder();

        // 1. スキーマ情報
        report.append("=========================================\n");
        report.append("1. Schema Information\n");
        report.append("=========================================\n");
        // printSchema() の代わりに treeString() を使って文字列として取得
        report.append(df.schema().treeString()).append("\n");

        // 2. 全体件数
        report.append("=========================================\n");
        report.append("2. Total Record Count\n");
        report.append("=========================================\n");
        report.append("Total count: ").append(df.count()).append("\n\n");

        // 3. Null値のカウント
        report.append("=========================================\n");
        report.append("3. Null Counts per Column\n");
        report.append("=========================================\n");
        for (String col : df.columns()) {
            long nullCount = df.filter(df.col(col).isNull()).count();
            report.append(col).append(" null count: ").append(nullCount).append("\n");
        }
        report.append("\n");

        // 4. 基本統計量 (describe)
        report.append("=========================================\n");
        report.append("4. Basic Statistics (Describe)\n");
        report.append("=========================================\n");
        Dataset<Row> descDf = df.describe();

        String[] columns = descDf.columns();
        List<Row> rows = descDf.collectAsList();

        // --- ① 各カラムの「最大文字数」を計算する ---
        int[] colWidths = new int[columns.length];
        // まずはヘッダー（カラム名）の長さを初期値としてセット
        for (int i = 0; i < columns.length; i++) {
            colWidths[i] = columns[i].length();
        }
        // 全行のデータをチェックし、一番長い文字数を上書き記録していく
        for (Row row : rows) {
            for (int i = 0; i < columns.length; i++) {
                String val = row.get(i) != null ? row.get(i).toString() : "null";
                if (val.length() > colWidths[i]) {
                    colWidths[i] = val.length();
                }
            }
        }

        // --- ② フォーマット用の文字列（例： "| %-15s | %-20s |"）を動的に作成 ---
        StringBuilder formatBuilder = new StringBuilder("| ");
        for (int width : colWidths) {
            // "%-" は左詰め、"s" は文字列。余裕を持たせるために幅に +2 しています
            formatBuilder.append("%-").append(width + 2).append("s | ");
        }
        String formatStr = formatBuilder.toString() + "\n";

        // --- ③ 整形してレポートに書き込む ---
        // ヘッダー行の書き込み
        report.append(String.format(formatStr, (Object[]) columns));

        // ヘッダーとデータの間に区切り線を引く（見た目をリッチに）
        int totalWidth = formatStr.length() - 1; // \nを引いた長さ
        report.append("-".repeat(Math.max(0, totalWidth))).append("\n");

        // データ行の書き込み
        for (Row row : rows) {
            String[] rowData = new String[columns.length];
            for (int i = 0; i < columns.length; i++) {
                rowData[i] = row.get(i) != null ? row.get(i).toString() : "null";
            }
            report.append(String.format(formatStr, (Object[]) rowData));
        }

        // ==========================================
        // ★ ファイルへの書き込み（上書きモード）
        // ==========================================
        try {
            File file = new File(OUTPUT_FILE_PATH);

            // resources配下に tmp ディレクトリが存在しない場合は自動作成する
            file.getParentFile().mkdirs();

            // PrintWriterはデフォルトでファイルを「上書き（Overwrite）」します
            try (PrintWriter writer = new PrintWriter(file, StandardCharsets.UTF_8.name())) {
                writer.print(report.toString());
                System.out.println("✅ データ分析レポートを保存しました: " + file.getAbsolutePath());
            }
        } catch (Exception e) {
            System.err.println("ファイルの書き込みに失敗しました: " + e.getMessage());
        }
    }
}