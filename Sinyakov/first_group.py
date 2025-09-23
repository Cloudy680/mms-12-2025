import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

class CoinExperiment:
    def __init__(self, n_tosses=100, n_simulations=10000):
        self.n_tosses = n_tosses
        self.n_simulations = n_simulations
        
    def simulate_experiment(self, p=0.5):
        """Моделирование эксперимента с вероятностью орла p"""
        results = np.random.binomial(self.n_tosses, p, self.n_simulations)
        return results
    
    def analyze_experiment(self, p=0.5):
        """Анализ результатов эксперимента"""
        results = self.simulate_experiment(p)
        
        # 1. Среднее число орлов
        mean_heads = np.mean(results)
        
        # 2. Вероятность > 60 орлов
        prob_gt_60 = np.mean(results > 60)
        
        # 3. Вероятности по интервалам
        bins = np.arange(0, 101, 10)
        bin_probs = []
        for i in range(len(bins)-1):
            if i == len(bins)-2:  # Последний интервал [90, 100]
                mask = (results >= bins[i]) & (results <= bins[i+1])
            else:
                mask = (results >= bins[i]) & (results < bins[i+1])
            bin_probs.append(np.mean(mask))
        
        # 4. 95% доверительный интервал
        lower_bound = np.percentile(results, 2.5)
        upper_bound = np.percentile(results, 97.5)
        interval_width = upper_bound - lower_bound
        
        # 5. Вероятность серии из 5 орлов подряд
        prob_series_5 = self.calculate_series_probability(results, series_length=5)
        
        # 6. Средняя длина максимальной серии
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
        """Вычисление вероятности наличия серии из n орлов подряд"""
        has_series_count = 0
        
        for experiment_result in results:
            # Для каждого эксперимента проверяем серии
            tosses = np.random.binomial(1, 0.5, self.n_tosses)  # Симулируем броски
            current_streak = 0
            found_series = False
            
            for toss in tosses:
                if toss == 1:  # Орел
                    current_streak += 1
                    if current_streak >= series_length:
                        found_series = True
                        break
                else:  # Решка
                    current_streak = 0
            
            if found_series:
                has_series_count += 1
        
        return has_series_count / len(results)
    
    def calculate_mean_max_series(self, results):
        """Вычисление средней длины максимальной серии орлов"""
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

def main():
    # Создаем экземпляр эксперимента
    experiment = CoinExperiment(n_tosses=100, n_simulations=10000)
    
    print("Эксперимент с симметричной монетой (p=0.5):")
    print("=" * 50)
    
    # Анализ для симметричной монеты
    results_symmetric = experiment.analyze_experiment(p=0.5)
    
    print(f"1. Среднее число орлов: {results_symmetric['mean_heads']:.2f}")
    print(f"2. Вероятность > 60 орлов: {results_symmetric['prob_gt_60']:.4f}")
    print(f"4. 95% доверительный интервал: [{results_symmetric['confidence_interval'][0]:.1f}, {results_symmetric['confidence_interval'][1]:.1f}]")
    print(f"   Ширина интервала: {results_symmetric['interval_width']:.1f}")
    print(f"5. Вероятность серии из 5 орлов: {results_symmetric['prob_series_5']:.4f}")
    print(f"6. Средняя длина максимальной серии: {results_symmetric['mean_max_series']:.2f}")
    
    print("\n3. Вероятности по интервалам:")
    bins = ["[0,10)", "[10,20)", "[20,30)", "[30,40)", "[40,50)", 
            "[50,60)", "[60,70)", "[70,80)", "[80,90)", "[90,100]"]
    
    for bin_name, prob in zip(bins, results_symmetric['bin_probs']):
        print(f"   {bin_name}: {prob:.4f}")
    
    # Визуализация распределения
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_symmetric['results'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(results_symmetric['mean_heads'], color='red', linestyle='--', label=f'Среднее: {results_symmetric["mean_heads"]:.2f}')
    plt.xlabel('Число орлов')
    plt.ylabel('Частота')
    plt.title('Распределение числа орлов (p=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Анализ для различных значений p
    p_values = np.linspace(0.1, 0.9, 17)
    means = []
    widths = []
    series_probs = []
    max_series_lengths = []
    
    for p in p_values:
        results = experiment.analyze_experiment(p=p)
        means.append(results['mean_heads'])
        widths.append(results['interval_width'])
        series_probs.append(results['prob_series_5'])
        max_series_lengths.append(results['mean_max_series'])
    
    plt.subplot(2, 2, 2)
    plt.plot(p_values, means, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Среднее число орлов')
    plt.title('Зависимость среднего числа орлов от p')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(p_values, widths, 's-', linewidth=2, markersize=6, color='orange')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Ширина 95% интервала')
    plt.title('Зависимость ширины интервала от p')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(p_values, series_probs, '^-', linewidth=2, markersize=6, color='green', label='Вероятность серии из 5')
    plt.plot(p_values, max_series_lengths, 'd-', linewidth=2, markersize=6, color='purple', label='Ср. длина макс. серии')
    plt.xlabel('Вероятность орла (p)')
    plt.ylabel('Вероятность / Длина')
    plt.title('Зависимость серий от p')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительная информация
    print("\nТеоретические значения для симметричной монеты:")
    print(f"Ожидаемое среднее: {100 * 0.5}")
    print(f"Стандартное отклонение: {np.sqrt(100 * 0.5 * 0.5):.2f}")
    print(f"Теоретический 95% интервал: [{50 - 1.96*5}, {50 + 1.96*5}]")

if __name__ == "__main__":
    main()