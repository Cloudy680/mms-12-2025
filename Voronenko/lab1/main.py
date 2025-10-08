import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

def simulate_coin_flips(n_flips=100, p=0.5):
    results = []
    for _ in range(n_flips):
        if random.random() < p:
            results.append('H')
        else:
            results.append('T')
    return results


def analyze_series(flips):
    max_series = 0
    current_series = 0
    has_5_series = False

    for flip in flips:
        if flip == 'H':
            current_series += 1
            max_series = max(max_series, current_series)
            if current_series >= 5:
                has_5_series = True
        else:
            current_series = 0

    return max_series, has_5_series


def run_experiments(n_experiments=10000, n_flips=100, p=0.5):
    heads_counts = []
    max_series_lengths = []
    has_5_series_flags = []

    for _ in range(n_experiments):
        flips = simulate_coin_flips(n_flips, p)
        heads_count = flips.count('H')
        max_series, has_5_series = analyze_series(flips)

        heads_counts.append(heads_count)
        max_series_lengths.append(max_series)
        has_5_series_flags.append(has_5_series)

    return heads_counts, max_series_lengths, has_5_series_flags


def calculate_interval_probabilities(heads_counts):
    intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]

    probabilities = {}
    total = len(heads_counts)

    for i, (start, end) in enumerate(intervals):
        count = sum(1 for h in heads_counts if start <= h < end)
        probabilities[f"[{start},{end - 1}]"] = count / total

    return probabilities


def calculate_prediction_interval(heads_counts, confidence=0.95):
    heads_counts_sorted = sorted(heads_counts)
    n = len(heads_counts_sorted)

    lower_idx = int((1 - confidence) / 2 * n)
    upper_idx = int((1 + confidence) / 2 * n)

    lower_bound = heads_counts_sorted[lower_idx]
    upper_bound = heads_counts_sorted[upper_idx]

    return lower_bound, upper_bound, upper_bound - lower_bound


def main():
    n_experiments = 10000
    n_flips = 100
    p = 0.5

    print(f"\nКоличество экспериментов: {n_experiments}")
    print(f"Бросков в эксперименте: {n_flips}\n")

    heads_counts, max_series_lengths, has_5_series_flags = run_experiments(n_experiments, n_flips, p)

    avg_heads = np.mean(heads_counts)
    print(f"1. Среднее число орлов: {avg_heads:.2f}")

    prob_more_than_60 = sum(1 for h in heads_counts if h > 60) / n_experiments
    print(f"2. Вероятность получить >60 орлов: {prob_more_than_60:.4f}")

    interval_probs = calculate_interval_probabilities(heads_counts)
    print("\n3. Вероятности по интервалам:")
    for interval, prob in interval_probs.items():
        print(f"   {interval}: {prob:.4f}")

    lower, upper, width = calculate_prediction_interval(heads_counts, 0.95)
    print(f"\n4. Интервал с вероятностью 0.95: [{lower}, {upper}]")
    print(f"   Ширина интервала: {width}")

    prob_5_series = sum(has_5_series_flags) / n_experiments
    print(f"5. Вероятность серии из 5 орлов: {prob_5_series:.4f}")

    avg_max_series = np.mean(max_series_lengths)
    print(f"6. Средняя длина максимальной серии: {avg_max_series:.2f}")

    print("\nНесимметричная монета: ")
    analyze_asymmetric_coin()


def analyze_asymmetric_coin():
    n_experiments = 1000
    n_flips = 100

    p_values = np.linspace(0.01, 0.99, 50)
    results = {
        'avg_heads': [],
        'interval_widths': [],
        'prob_5_series': [],
        'avg_max_series': []
    }

    print("Вычисление для различных p...")
    for i, p in enumerate(p_values):
        heads_counts, max_series_lengths, has_5_series_flags = run_experiments(
            n_experiments, n_flips, p)

        results['avg_heads'].append(np.mean(heads_counts))
        _, _, width = calculate_prediction_interval(heads_counts, 0.95)
        results['interval_widths'].append(width)
        results['prob_5_series'].append(sum(has_5_series_flags) / n_experiments)
        results['avg_max_series'].append(np.mean(max_series_lengths))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(p_values, results['avg_heads'], 'b-', linewidth=2)
    ax1.set_xlabel('Вероятность орла (p)')
    ax1.set_ylabel('Среднее число орлов')
    ax1.set_title('Зависимость среднего числа орлов от p')
    ax1.grid(True)

    ax2.plot(p_values, results['interval_widths'], 'r-', linewidth=2)
    ax2.set_xlabel('Вероятность орла (p)')
    ax2.set_ylabel('Ширина интервала')
    ax2.set_title('Ширина доверительного интервала')
    ax2.grid(True)

    ax3.plot(p_values, results['prob_5_series'], 'g-', linewidth=2)
    ax3.set_xlabel('Вероятность орла (p)')
    ax3.set_ylabel('Вероятность')
    ax3.set_title('Вероятность серии из 5 орлов подряд')
    ax3.grid(True)

    ax4.plot(p_values, results['avg_max_series'], 'm-', linewidth=2)
    ax4.set_xlabel('Вероятность орла (p)')
    ax4.set_ylabel('Длина серии')
    ax4.set_title('Средняя длина максимальной серии орлов')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()