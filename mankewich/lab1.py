import random
import numpy as np
import matplotlib.pyplot as plt

class CoinExperiment:
    def __init__(self, tosses=100, randomSeed=None):
        self.tosses = tosses
        if randomSeed is not None:
            random.seed(randomSeed)
    
    def experiment(self, p=0.5):
        results = []
        for _ in range(self.tosses):
            randNum = random.random()
            if randNum < p:
                results.append(1)
            else:
                results.append(0)
        return results
    
    def multipleExperiments(self, experimentsCount=10000, p=0.5):
        results = []
        for _ in range(experimentsCount):
            tosses = self.experiment(p)
            headsCount = sum(tosses)
            results.append(headsCount)
        return results
    
    def findSeries(self, tosses, seriesLength=5):
        maxSeries = 0
        currentSeries = 0
        
        for result in tosses:
            if result == 1:  # орел
                currentSeries += 1
                if currentSeries > maxSeries:
                    maxSeries = currentSeries
            else:  # решка
                currentSeries = 0
        
        hasSeries = maxSeries >= seriesLength
        return hasSeries, maxSeries

def countCondition(numbers, conditionFunc):
    count = 0
    for num in numbers:
        if conditionFunc(num):
            count += 1
    return count

def main():
    coin = CoinExperiment(tosses=100, randomSeed=123)
    
    experimentsCount = 10000
    results = coin.multipleExperiments(experimentsCount)
    
    meanHeads = np.mean(results)
    print(f"1) Среднее число орлов: {meanHeads:.2f} (близко к 50.0)")
    
    countMoreThan60 = countCondition(results, lambda x: x > 60)
    probMoreThan60 = countMoreThan60 / experimentsCount
    print(f"2) Вероятность получить > 60 орлов: {probMoreThan60:.4f}")
    
    intervals = [(i, i+10) for i in range(0, 100, 10)]
    intervalsProbs = []
    
    print("\n3) Вероятности по интервалам:")
    for start, end in intervals:
        if end == 100:
            countInInterval = countCondition(results, lambda x: start <= x <= end)
        else:
            countInInterval = countCondition(results, lambda x: start <= x < end)
        
        prob = countInInterval / experimentsCount
        intervalsProbs.append(prob)
        print(f"   [{start:2d}, {end:2d}): {prob:.4f}")
    
    lower = np.percentile(results, 2.5)
    upper = np.percentile(results, 97.5)
    intervalWidth = upper - lower
    
    print(f"\n4) 95% вероятностный интервал: [{lower:.1f}, {upper:.1f}]")
    print(f"   Ширина интервала: {intervalWidth:.1f}")
    
    countInInterval = countCondition(results, lambda x: lower <= x <= upper)
    actualProb = countInInterval / experimentsCount
    print(f"   Фактическая вероятность в интервале: {actualProb:.4f}")
    
    seriesResults = []
    maxSeriesResults = []
    
    for i in range(experimentsCount):
        tosses = coin.experiment()
        hasSeries, maxSeries = coin.findSeries(tosses, 5)
        seriesResults.append(hasSeries)
        maxSeriesResults.append(maxSeries)
    
    countHasSeries = sum(1 for hasSeries in seriesResults if hasSeries)
    probSeries5 = countHasSeries / experimentsCount
    
    avgMaxSeries = np.mean(maxSeriesResults)
    
    print(f"\n5) Вероятность серии из 5 орлов подряд: {probSeries5:.4f}")
    print(f"   Средняя длина максимальной серии: {avgMaxSeries:.2f}")
    
    maxSeriesEver = max(maxSeriesResults)
    print(f"   Максимальная серия за все эксперименты: {maxSeriesEver}")

def analyzeDependenceOnP():
    coin = CoinExperiment(tosses=100, randomSeed=42)
    
    pValues = []
    for i in range(1, 100, 2):  
        pValues.append(i / 100.0)
    
    experimentsCount = 2000 
    
    expectedHeads = []
    intervalWidths = []
    probSeries5 = []
    avgMaxSeries = []
    
    print("Вычисление зависимостей...")
    for i, p in enumerate(pValues):
        if i % 10 == 0:
            print(f"  Обработано {i}/{len(pValues)} значений p")
        
        results = coin.multipleExperiments(experimentsCount, p)
        expectedHeads.append(np.mean(results))
        
        lower = np.percentile(results, 2.5)
        upper = np.percentile(results, 97.5)
        intervalWidths.append(upper - lower)
        
        seriesCount = 0
        maxSeriesTotal = 0
        
        for _ in range(experimentsCount):
            tosses = coin.experiment(p)
            hasSeries, maxSeries = coin.findSeries(tosses, 5)
            if hasSeries:
                seriesCount += 1
            maxSeriesTotal += maxSeries
        
        probSeries5.append(seriesCount / experimentsCount)
        avgMaxSeries.append(maxSeriesTotal / experimentsCount)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(pValues, expectedHeads, 'b-', linewidth=2, label='Экспериментальное')
    theoretical = [100 * p for p in pValues]
    ax1.plot(pValues, theoretical, 'r--', alpha=0.7, label='Теоретическое')
    ax1.set_xlabel('Вероятность орла (p)')
    ax1.set_ylabel('Ожидаемое число орлов')
    ax1.set_title('Зависимость ожидаемого числа орлов от p')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pValues, intervalWidths, 'g-', linewidth=2)
    ax2.set_xlabel('Вероятность орла (p)')
    ax2.set_ylabel('Ширина 95% интервала')
    ax2.set_title('Зависимость ширины интервала от p')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(pValues, probSeries5, 'purple', linewidth=2)
    ax3.set_xlabel('Вероятность орла (p)')
    ax3.set_ylabel('Вероятность серии из 5 орлов')
    ax3.set_title('Зависимость вероятности серии от p')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(pValues, avgMaxSeries, 'orange', linewidth=2)
    ax4.set_xlabel('Вероятность орла (p)')
    ax4.set_ylabel('Средняя длина макс. серии')
    ax4.set_title('Зависимость длины серии от p')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pValues, expectedHeads, intervalWidths, probSeries5, avgMaxSeries

def demonstrateSingleExperiment():
    coin = CoinExperiment(tosses=20, randomSeed=123)
    
    print("ДЕМОНСТРАЦИЯ ОДНОГО ЭКСПЕРИМЕНТА:")
    print("=" * 40)
    
    tosses = coin.experiment(p=0.5)
    print("Результаты бросков (1-орел, 0-решка):")
    print(tosses)
    
    headsCount = sum(tosses)
    print(f"Количество орлов: {headsCount}")
    
    hasSeries, maxSeries = coin.findSeries(tosses, 3)
    print(f"Есть серия из 3+ орлов: {hasSeries}")
    print(f"Максимальная серия орлов: {maxSeries}")
    
    print("\nВизуализация серий:")
    currentSeries = 0
    seriesInfo = []
    for i, result in enumerate(tosses):
        if result == 1:
            currentSeries += 1
        else:
            if currentSeries > 0:
                seriesInfo.append((i-currentSeries, i-1, currentSeries))
                currentSeries = 0
        symbol = 'О' if result == 1 else 'Р'
        print(symbol, end=' ')
    
    if currentSeries > 0:
        seriesInfo.append((len(tosses)-currentSeries, len(tosses)-1, currentSeries))
    
    print("\n\nНайденные серии орлов:")
    for start, end, length in seriesInfo:
        print(f"  Серия из {length} орлов: позиции {start}-{end}")

if __name__ == "__main__":
    print("АНАЛИЗ ЭКСПЕРИМЕНТА С ПОДБРАСЫВАНИЕМ МОНЕТЫ")

    demonstrateSingleExperiment()
    
    print("\nПОЛНЫЙ АНАЛИЗ ДЛЯ 100 БРОСКОВ")
    
    main()
    
    print("\nАНАЛИЗ ЗАВИСИМОСТИ ОТ ВЕРОЯТНОСТИ p")

    analyzeDependenceOnP()