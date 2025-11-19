import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {

    interface S {
        int move(int r, List<Integer> my, List<Integer> op);
    }

    static class Alex implements S {
        public int move(int r, List<Integer> my, List<Integer> op) { return 1; }
    }

    static class Bob implements S {
        public int move(int r, List<Integer> my, List<Integer> op) { return 0; }
    }

    static class Clara implements S {
        public int move(int r, List<Integer> my, List<Integer> op) {
            if (r == 0) return 0;
            return op.get(op.size() - 1);
        }
    }

    static class Denis implements S {
        public int move(int r, List<Integer> my, List<Integer> op) {
            if (r == 0) return 0;
            return op.get(op.size() - 1) == 0 ? 1 : 0;
        }
    }

    static class Emma implements S {
        public int move(int r, List<Integer> my, List<Integer> op) {
            int roundOneBased = r + 1;
            if (roundOneBased % 20 == 0) return 1;
            return 0;
        }
    }

    static class Frida implements S {
        public int move(int r, List<Integer> my, List<Integer> op) {
            for (int v : op) if (v == 1) return 1;
            return 0;
        }
    }

    static class George implements S {
        // Two-strike: пока у оппонента <2 предательств — 0, после >=2 — всегда 1
        public int move(int r, List<Integer> my, List<Integer> op) {
            int cnt = 0;
            for (int v : op) if (v == 1) cnt++;
            return cnt >= 2 ? 1 : 0;
        }
    }

    static final int ROUNDS = 200;

    static int payoff(int a, int b) {
        if (a == 0 && b == 0) return 3;
        if (a == 0 && b == 1) return 0;
        if (a == 1 && b == 0) return 5;
        return 1;
    }

    static Result playMatch(S s1, S s2) {
        List<Integer> a = new ArrayList<>();
        List<Integer> b = new ArrayList<>();
        int sumA = 0;
        int sumB = 0;
        int longestDom = 0;
        int curDom = 0;
        Integer curWinner = null; // 0 none, 1 A, 2 B

        for (int r = 0; r < ROUNDS; r++) {
            int m1 = s1.move(r, a, b);
            int m2 = s2.move(r, b, a);
            a.add(m1);
            b.add(m2);
            int p1 = payoff(m1, m2);
            int p2 = payoff(m2, m1);
            sumA += p1;
            sumB += p2;

            if (p1 == 5 && p2 == 0) {
                if (curWinner != null && curWinner == 1) {
                    curDom++;
                } else {
                    curWinner = 1;
                    curDom = 1;
                }
            } else if (p2 == 5 && p1 == 0) {
                if (curWinner != null && curWinner == 2) {
                    curDom++;
                } else {
                    curWinner = 2;
                    curDom = 1;
                }
            } else {
                curWinner = null;
                curDom = 0;
            }
            if (curDom > longestDom) longestDom = curDom;
        }
        return new Result(sumA, sumB, longestDom);
    }

    static class Result {
        int a;
        int b;
        int dom;
        Result(int a, int b, int dom) { this.a = a; this.b = b; this.dom = dom; }
    }

    public static void main(String[] args) throws Exception {
        List<String> names = List.of("Alex","Bob","Clara","Denis","Emma","Frida","George");
        List<S> strat = List.of(
                new Alex(),
                new Bob(),
                new Clara(),
                new Denis(),
                new Emma(),
                new Frida(),
                new George()
        );

        int n = names.size();
        int[][] scores = new int[n][n];
        int[][] series = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Result r = playMatch(strat.get(i), strat.get(j));
                scores[i][j] = r.a;
                series[i][j] = r.dom;
            }
        }

        printTable("Матрица очков (A)", names, scores);
        printTable("Матрица длин доминирующих серий", names, series);

        saveCsv("scores.csv", names, scores);
        saveCsv("series.csv", names, series);
        System.out.println("CSV saved: scores.csv, series.csv");
    }

    static void printTable(String title, List<String> names, int[][] m) {
        System.out.println();
        System.out.println(title);
        System.out.print(String.format("%15s", ""));
        for (String s : names) System.out.print(String.format("%10s", s));
        System.out.println();
        for (int i = 0; i < names.size(); i++) {
            System.out.print(String.format("%15s", names.get(i)));
            for (int j = 0; j < names.size(); j++) {
                System.out.print(String.format("%10d", m[i][j]));
            }
            System.out.println();
        }
    }

    static void saveCsv(String file, List<String> names, int[][] m) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append(",");
        for (int i = 0; i < names.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(names.get(i));
        }
        sb.append("\n");
        for (int i = 0; i < names.size(); i++) {
            sb.append(names.get(i));
            for (int j = 0; j < names.size(); j++) {
                sb.append(",");
                sb.append(m[i][j]);
            }
            sb.append("\n");
        }
        Files.writeString(Path.of(file), sb.toString());
    }
}
