import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

class CoinExperiment:
    def __init__(self, n_tosses=100, n_experiments=100, random_seed=None):
        self.n_tosses = n_tosses
        self.n_experiments = n_experiments
        if random_seed is not None:
            np.random.seed(random_seed)
        self.results = None
        self.all_tosses = None
        
    def simulate_experiments(self, p=0.5):
        self.all_tosses = np.random.choice([0, 1], size=(self.n_experiments, self.n_tosses), p=[1-p, p])
        self.results = np.sum(self.all_tosses, axis=1)
        return self.results, self.all_tosses
    
    def find_series(self, tosses, series_length=5):
        max_series = 0
        current_series = 0
        
        for result in tosses:
            if result == 1: 
                current_series += 1
                if current_series > max_series:
                    max_series = current_series
            else: 
                current_series = 0
        
        has_series = max_series >= series_length
        return has_series, max_series
    
    def multiple_experiments(self, n_experiments, p=0.5):
        experiments = np.random.choice([0, 1], size=(n_experiments, self.n_tosses), p=[1-p, p])
        return experiments
    
    def analyze_experiments(self, p=0.5):
        if self.results is None:
            self.simulate_experiments(p)
            
        mean_heads = np.mean(self.results)
        
        prob_gt_60 = np.mean(self.results > 60)
        
        bins = np.arange(0, 101, 10)
        bin_probs = []
        for i in range(len(bins)-1):
            if i == len(bins)-2:
                mask = (self.results >= bins[i]) & (self.results <= bins[i+1])
            else:
                mask = (self.results >= bins[i]) & (self.results < bins[i+1])
            bin_probs.append(np.mean(mask))
        
        sorted_results = sorted(self.results)
        lower_index = int(0.025 * len(sorted_results))
        upper_index = int(0.975 * len(sorted_results))
        lower_index = max(0, lower_index)
        upper_index = min(len(sorted_results) - 1, upper_index)
        
        confidence_interval = (sorted_results[lower_index], sorted_results[upper_index])
        interval_width = sorted_results[upper_index] - sorted_results[lower_index]
        
        series_count = 0
        max_series_total = 0
        for exp in self.all_tosses:
            has_series, max_series = self.find_series(exp, 5)
            if has_series:
                series_count += 1
            max_series_total += max_series
        
        prob_series_5 = series_count / self.n_experiments
        mean_max_series = max_series_total / self.n_experiments
        
        return {
            'mean_heads': mean_heads,
            'prob_gt_60': prob_gt_60,
            'bin_probs': bin_probs,
            'confidence_interval': confidence_interval,
            'interval_width': interval_width,
            'prob_series_5': prob_series_5,
            'mean_max_series': mean_max_series,
            'results': self.results,
            'all_tosses': self.all_tosses
        }

def execute_analysis():
    experiment = CoinExperiment(n_tosses=100, n_experiments=10000, random_seed=42)
    
    print("Лабораторная №1: Моделирование эксперимента с бросанием монеты")
    print("=" * 70)
    print(f"Количество экспериментов: 100")
    print(f"Бросков в каждом эксперименте: 100")
    print("Генерация: 0 - решка, 1 - орел")
    print()
    
    print("Анализ симметричной монеты (p=0.5):")
    print("=" * 50)
    
    symmetric_results = experiment.analyze_experiments(p=0.5)
    
    print("ЭКСПЕРИМЕНТАЛЬНЫЕ РЕЗУЛЬТАТЫ (100 экспериментов):")
    print(f"1. Среднее количество орлов: {symmetric_results['mean_heads']:.2f}")
    print(f"2. Вероятность превышения 60 орлов: {symmetric_results['prob_gt_60']:.4f}")
    print(f"4. 95% доверительный интервал: [{symmetric_results['confidence_interval'][0]:.1f}, {symmetric_results['confidence_interval'][1]:.1f}]")
    print(f"   Размер интервала: {symmetric_results['interval_width']:.1f}")
    print(f"5. Вероятность серии из 5 орлов: {symmetric_results['prob_series_5']:.4f}")
    print(f"6. Средняя длина максимальной серии: {symmetric_results['mean_max_series']:.2f}")
    
    print("\n3. Вероятности по интервалам:")
    intervals = ["[0,10)", "[10,20)", "[20,30)", "[30,40)", "[40,50)", 
                "[50,60)", "[60,70)", "[70,80)", "[80,90)", "[90,100]"]
    
    for interval, probability in zip(intervals, symmetric_results['bin_probs']):
        print(f"   {interval}: {probability:.4f}")
    
    print(f"\nПример первого эксперимента (первые 30 бросков):")
    first_experiment_tosses = symmetric_results['all_tosses'][0][:30]
    tosses_str = ''.join(['О' if x == 1 else 'Р' for x in first_experiment_tosses])
    print(f"   {tosses_str}")
    print(f"   Всего орлов в первом эксперименте: {symmetric_results['results'][0]}")
    
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(2, 2, 1)
    plt.hist(symmetric_results['results'], bins=20, alpha=0.7, density=True, edgecolor='black')
    plt.axvline(symmetric_results['mean_heads'], color='red', linestyle='--', label=f'Среднее: {symmetric_results["mean_heads"]:.2f}')
    plt.xlabel('Количество орлов')
    plt.ylabel('Плотность вероятности')
    plt.title('Распределение количества орлов (100 экспериментов)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\n" + "=" * 70)
    print("Анализ зависимости от вероятности p (пункт 6):")
    print("=" * 70)
    
    experiments_count_p = 1000
    p_values = []
    for i in range(1, 100, 2):
        p_values.append(i / 100.0)
    
    avg_heads = []
    interval_widths = []
    prob_series_5_list = []
    avg_max_series_list = []
    
    for p in p_values:
        experiments_p = experiment.multiple_experiments(experiments_count_p, p)
        head_sums_p = [sum(exp) for exp in experiments_p]
        avg_heads.append(np.mean(head_sums_p))
        
        if len(head_sums_p) > 1:
            sorted_head_sums_p = sorted(head_sums_p)
            lower_index = int(0.025 * len(sorted_head_sums_p))
            upper_index = int(0.975 * len(sorted_head_sums_p))
            lower_index = max(0, lower_index)
            upper_index = min(len(sorted_head_sums_p) - 1, upper_index)
            interval_widths.append(sorted_head_sums_p[upper_index] - sorted_head_sums_p[lower_index])
        else:
            interval_widths.append(0)
        
        series_count = 0
        max_series_total = 0
        for exp in experiments_p:
            has_series, max_series = experiment.find_series(exp, 5)
            if has_series:
                series_count += 1
            max_series_total += max_series
        
        prob_series_5_list.append(series_count / experiments_count_p)
        avg_max_series_list.append(max_series_total / experiments_count_p)
    
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(p_values, avg_heads, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Среднее количество орлов')
    plt.title('6.1: Зависимость среднего количества орлов от p')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(p_values, interval_widths, 's-', linewidth=2, markersize=4, color='orange')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Ширина 95% интервала')
    plt.title('6.2: Зависимость ширины интервала от p')
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(p_values, prob_series_5_list, '^-', linewidth=2, markersize=4, color='green')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Вероятность серии из 5')
    plt.title('6.3: Вероятность наличия серии из 5 орлов')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(p_values, avg_max_series_list, 'd-', linewidth=2, markersize=6, color='purple')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Длина максимальной серии')
    plt.title('6.4: Средняя длина максимальной серии')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nАнализ проведен для {len(p_values)} значений p от {p_values[0]:.2f} до {p_values[-1]:.2f}")
    print(f"Количество экспериментов для каждого p: {experiments_count_p}")
    
    print("\nСохраненные значения для 100 экспериментов:")
    print(f"Количество орлов в каждом эксперименте: {symmetric_results['results'][:10]}...")
    print(f"Всего сохранено: {len(symmetric_results['results'])} значений")

if __name__ == "__main__":
    execute_analysis()
