package org.example;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {

    static Random rnd = new Random();

    // Однократный эксперимент — сколько орлов
    static int toss100(double p) {
        int h = 0;
        for (int i = 0; i < 100; i++)
            if (rnd.nextDouble() < p) h++;
        return h;
    }

    // Максимальная серия орлов
    static int maxRun(List<Integer> seq) {
        int max = 0, cur = 0;
        for (int v : seq) {
            if (v == 1) {
                cur++;
                if (cur > max) max = cur;
            } else cur = 0;
        }
        return max;
    }

    // Есть ли серия из k подряд
    static boolean hasRun(List<Integer> seq, int k) {
        int cur = 0;
        for (int v : seq) {
            if (v == 1) {
                cur++;
                if (cur >= k) return true;
            } else cur = 0;
        }
        return false;
    }

    // Однократная последовательность
    static List<Integer> seq100(double p) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < 100; i++)
            r.add(rnd.nextDouble() < p ? 1 : 0);
        return r;
    }

    // Построение графика
    static void saveChart(String title, XYSeries series, String path) throws IOException {
        XYSeriesCollection ds = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createXYLineChart(
                title, "p", "value", ds
        );
        ChartUtils.saveChartAsPNG(new File(path), chart, 900, 600);
    }

    public static void main(String[] args) throws Exception {

        int N = 100000; // число моделирований

        // 1. Среднее число орлов при p = 0.5
        double p0 = 0.5;
        double sum = 0;
        for (int i = 0; i < N; i++) sum += toss100(p0);
        System.out.println("1) Среднее число орлов = " + sum / N);

        // 2. Вероятность > 60 орлов
        int cnt = 0;
        for (int i = 0; i < N; i++)
            if (toss100(p0) > 60) cnt++;
        System.out.println("2) P(H > 60) = " + (cnt * 1.0 / N));

        // 3. Вероятности по интервалам
        int[] borders = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        int[] freq = new int[borders.length - 1];

        for (int i = 0; i < N; i++) {
            int h = toss100(p0);
            for (int j = 0; j < borders.length - 1; j++)
                if (h >= borders[j] && h <= borders[j + 1]) freq[j]++;
        }
        System.out.println("3) Интервалы:");
        for (int j = 0; j < freq.length; j++) {
            double pr = freq[j] * 1.0 / N;
            System.out.println(borders[j] + "-" + borders[j + 1] + ": " + pr);
        }

        // 4. 95%-интервал (центральнаый квантиль)
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < N; i++) all.add(toss100(p0));

        Collections.sort(all);
        int low = all.get((int) (0.025 * N));
        int high = all.get((int) (0.975 * N));

        System.out.println("4) 95% интервал: [" + low + ", " + high + "]");

        // 5. Вероятность серии из 5 орлов
        int runs = 0;
        for (int i = 0; i < N; i++) {
            if (hasRun(seq100(p0), 5)) runs++;
        }
        System.out.println("5) P(серия >= 5) = " + (runs * 1.0 / N));

        //6. ГРАФИКИ ДЛЯ РАЗНЫХ p
        XYSeries s1 = new XYSeries("E[H]");
        XYSeries s2 = new XYSeries("95% interval width");
        XYSeries s3 = new XYSeries("P(run>=5)");
        XYSeries s4 = new XYSeries("max run");

        for (double p = 0.0; p <= 1.0; p += 0.02) {

            double sumH = 0;
            int cntRun = 0;
            double sumMaxRun = 0;

            List<Integer> vals = new ArrayList<>();

            for (int k = 0; k < N; k++) {
                List<Integer> seq = seq100(p);
                int h = 0;
                for (int x : seq) if (x == 1) h++;

                vals.add(h);
                sumH += h;
                if (hasRun(seq, 5)) cntRun++;
                sumMaxRun += maxRun(seq);
            }

            Collections.sort(vals);
            int L = vals.get((int) (0.025 * N));
            int R = vals.get((int) (0.975 * N));

            s1.add(p, sumH / N);
            s2.add(p, R - L);
            s3.add(p, cntRun * 1.0 / N);
            s4.add(p, sumMaxRun / N);
        }

        saveChart("6.i ожид. число орлов", s1, "chart1_expected.png");
        saveChart("6.ii ширина 95% интервала", s2, "chart2_width.png");
        saveChart("6.iii вероятность серии >=5", s3, "chart3_run5.png");
        saveChart("6.iv ожидаемая макс серия", s4, "chart4_maxrun.png");

        System.out.println("Все графики сохранены в файлы PNG.");
    }
}
