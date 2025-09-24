import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

class CoinExperiment:
    def __init__(self, n_tosses=100, n_simulations=10000):
        self.n_tosses = n_tosses
        self.n_simulations = n_simulations
        
    def simulate_experiment(self, p=0.5):
        results = np.random.binomial(self.n_tosses, p, self.n_simulations)
        return results
    
    def analyze_experiment(self, p=0.5):
        results = self.simulate_experiment(p)
        
        mean_heads = np.mean(results)
        
        prob_gt_60 = np.mean(results > 60)
        
        bins = np.arange(0, 101, 10)
        bin_probs = []
        for i in range(len(bins)-1):
            if i == len(bins)-2:
                mask = (results >= bins[i]) & (results <= bins[i+1])
            else:
                mask = (results >= bins[i]) & (results < bins[i+1])
            bin_probs.append(np.mean(mask))
        
        lower_bound = np.percentile(results, 2.5)
        upper_bound = np.percentile(results, 97.5)
        interval_width = upper_bound - lower_bound
        
        prob_series_5 = self.calculate_series_probability(results, series_length=5)
        
        mean_max_series = self.calculate_mean_max_series(results)
        
        return {
            'mean_heads': mean_heads,
            'prob_gt_60': prob_gt_60,
            'bin_probs': bin_probs,
            'confidence_interval': (lower_bound, upper_bound),
            'interval_width': interval_width,
            'prob_series_5': prob_series_5,
            'mean_max_series': mean_max_series,
            'results': results
        }
    
    def calculate_series_probability(self, results, series_length=5):
        has_series_count = 0
        
        for experiment_result in results:
            tosses = np.random.binomial(1, 0.5, self.n_tosses)
            current_streak = 0
            found_series = False
            
            for toss in tosses:
                if toss == 1:
                    current_streak += 1
                    if current_streak >= series_length:
                        found_series = True
                        break
                else:
                    current_streak = 0
            
            if found_series:
                has_series_count += 1
        
        return has_series_count / len(results)
    
    def calculate_mean_max_series(self, results):
        max_series_lengths = []
        
        for _ in range(self.n_simulations):
            tosses = np.random.binomial(1, 0.5, self.n_tosses)
            current_streak = 0
            max_streak = 0
            
            for toss in tosses:
                if toss == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            max_series_lengths.append(max_streak)
        
        return np.mean(max_series_lengths)

def execute_analysis():
    experiment = CoinExperiment(n_tosses=100, n_simulations=10000)
    
    print("Анализ симметричной монеты (p=0.5):")
    print("=" * 50)
    
    symmetric_results = experiment.analyze_experiment(p=0.5)
    
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
    
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(2, 2, 1)
    plt.hist(symmetric_results['results'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(symmetric_results['mean_heads'], color='red', linestyle='--', label=f'Среднее: {symmetric_results["mean_heads"]:.2f}')
    plt.xlabel('Количество орлов')
    plt.ylabel('Частота')
    plt.title('Распределение количества орлов (p=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    probability_values = np.linspace(0.1, 0.9, 17)
    averages = []
    interval_widths = []
    series_probabilities = []
    max_series_values = []
    
    for prob in probability_values:
        analysis_results = experiment.analyze_experiment(p=prob)
        averages.append(analysis_results['mean_heads'])
        interval_widths.append(analysis_results['interval_width'])
        series_probabilities.append(analysis_results['prob_series_5'])
        max_series_values.append(analysis_results['mean_max_series'])
    
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(probability_values, averages, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Среднее количество орлов')
    plt.title('Зависимость среднего количества орлов от p')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(probability_values, interval_widths, 's-', linewidth=2, markersize=6, color='orange')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Ширина 95% интервала')
    plt.title('Зависимость ширины интервала от p')
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(probability_values, series_probabilities, '^-', linewidth=2, markersize=6, color='green', label='Вероятность серии из 5')
    plt.plot(probability_values, max_series_values, 'd-', linewidth=2, markersize=6, color='purple', label='Ср. длина макс. серии')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Вероятность / Длина')
    plt.title('Влияние p на серии орлов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nТеоретические расчеты для симметричной монеты:")
    print(f"Ожидаемое среднее значение: {100 * 0.5}")
    print(f"Стандартное отклонение: {np.sqrt(100 * 0.5 * 0.5):.2f}")
    print(f"Теоретический 95% интервал: [{50 - 1.96*5}, {50 + 1.96*5}]")

if __name__ == "__main__":
    execute_analysis()
