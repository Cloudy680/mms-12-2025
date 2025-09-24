import numpy as np
import matplotlib.pyplot as plt

def run_experiment(n_flips=100, p=0.5):
    return np.random.binomial(1, p, n_flips)

def has_series_of_heads(flips, series_length=5):
    if len(flips) < series_length:
        return False
    for i in range(len(flips) - series_length + 1):
        if np.all(flips[i:i+series_length] == 1):
            return True
    return False

def max_series_length(flips):
    max_len = 0
    current_len = 0
    for flip in flips:
        if flip == 1:
            current_len += 1
        else:
            max_len = max(max_len, current_len)
            current_len = 0
    max_len = max(max_len, current_len)
    return max_len

def solve_symmetric_case(n_simulations=10000, n_flips=100):
    print("Симметричная монета (p=0.5)")
    heads_counts = []
    series_found_count = 0
    
    for _ in range(n_simulations):
        flips = run_experiment(n_flips, p=0.5)
        heads_counts.append(np.sum(flips))
        if has_series_of_heads(flips, 5):
            series_found_count += 1
            
    heads_counts = np.array(heads_counts)
    
    avg_heads = np.mean(heads_counts)
    print(f"1. Какое число орлов выпадает в среднем: {avg_heads:.2f}")
    
    prob_gt_60 = np.sum(heads_counts > 60) / n_simulations
    print(f"2. С какой вероятностью можно получить число орлов больше 60: {prob_gt_60:.4f}")
    
    print("3. Для интервалов ... оцените вероятность выпадения числа орлов:")
    for i in range(10):
        lower = i * 10
        upper = (i + 1) * 10
        if i < 9:
            count = np.sum((heads_counts >= lower) & (heads_counts < upper))
            print(f"   [{lower}, {upper}): {count/n_simulations:.4f}")
        else:
            count = np.sum((heads_counts >= 90) & (heads_counts <= 100))
            print(f"   [90, 100]: {count/n_simulations:.4f}")

    lower_bound = np.percentile(heads_counts, 2.5)
    upper_bound = np.percentile(heads_counts, 97.5)
    print(f"4. Внутри какого интервала с вероятность 0.95 стоит ожидать значение числа орлов: [{lower_bound:.0f}, {upper_bound:.0f}]")
    
    prob_series = series_found_count / n_simulations
    print(f"5. С какой вероятностью найдется хотябы одна серия из 5 орлов подряд: {prob_series:.4f}")

def solve_asymmetric_case(n_simulations=1000, n_flips=100):
    print("\nНесимметричная монета")
    p_values = np.linspace(0, 1, 51)
    
    avg_heads_list = []
    interval_widths_list = []
    series_prob_list = []
    max_series_len_list = []
    
    for p in p_values:
        heads_counts = []
        series_found_count = 0
        max_series_lengths = []
        
        for _ in range(n_simulations):
            flips = run_experiment(n_flips, p)
            heads_counts.append(np.sum(flips))
            if has_series_of_heads(flips, 5):
                series_found_count += 1
            max_series_lengths.append(max_series_length(flips))

        heads_counts = np.array(heads_counts)
        
        avg_heads_list.append(np.mean(heads_counts))
        
        lower_bound = np.percentile(heads_counts, 2.5)
        upper_bound = np.percentile(heads_counts, 97.5)
        interval_widths_list.append(upper_bound - lower_bound)
        
        series_prob_list.append(series_found_count / n_simulations)
        max_series_len_list.append(np.mean(max_series_lengths))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Зависимость метрик от вероятности выпадения орла (p)")
    
    axs[0, 0].plot(p_values, avg_heads_list)
    axs[0, 0].set_title("Ожидаемое число орлов")
    axs[0, 0].set_xlabel("p")
    axs[0, 0].set_ylabel("Среднее число орлов")
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(p_values, interval_widths_list)
    axs[0, 1].set_title("Ширина предсказательного интервала (95%)")
    axs[0, 1].set_xlabel("p")
    axs[0, 1].set_ylabel("Ширина интервала")
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(p_values, series_prob_list)
    axs[1, 0].set_title("Вероятность серии из 5 орлов")
    axs[1, 0].set_xlabel("p")
    axs[1, 0].set_ylabel("Вероятность")
    axs[1, 0].grid(True)

    axs[1, 1].plot(p_values, max_series_len_list)
    axs[1, 1].set_title("Средняя длина максимальной серии орлов")
    axs[1, 1].set_xlabel("p")
    axs[1, 1].set_ylabel("Длина")
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("lab1_plots.png")
    print("\nГрафики сохранены в файл 'lab1_plots.png'")

if __name__ == "__main__":
    solve_symmetric_case()
    solve_asymmetric_case()
