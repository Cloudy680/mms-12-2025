import matplotlib.pyplot as plt
import numpy as np


ROUNDS_COUNT = 200
WINNING_TABLE = [[3, 0], [5, 1]]

#history1 - история игрока
#history2 - история оппонента

def Alex(history1: list, history2: list) -> int:
    return 1

def Bob(history1: list, history2: list) -> int:
    return 0

def Clara(history1: list, history2: list) -> int:
    if len(history1) == 0:
        return 0
    else:
        return history2[-1]
    
def Denis(history1: list, history2: list) -> int:
    if len(history1) == 0:
        return 0
    else:
        return 1 - history2[-1]

def Emma(history1: list, history2: list) -> int:
    if len(history1) == 0: 
        return 0
    elif (len(history1) + 1) % 20 == 0:
        return 1
    else: return 0

def Frida(history1: list, history2: list) -> int:
    if len(history1) == 0:
        return 0
    elif 1 in history2: 
        return 1
    else: return 0

def George(history1: list, history2: list) -> int:
    if len(history2) == 0: 
        return 0
    elif len(history2) == 1:
        return ((1 + history2[-1])) % 2
    return (((len(history1) % 5) + history2[-1]) * (len(history1) % 5 + history2[-2])) % 2

def game(strategy1, strategy2):
    history1 = []
    history2 = []
    gamer1_score = 0
    gamer2_score = 0

    max_series1 = 0
    max_series2 = 0
    series1 = 0
    series2 = 0
    for i in range(ROUNDS_COUNT):
        history1.append(strategy1(history1, history2))
        history2.append(strategy2(history2, history1[0: -1]))

        i, j = history1[-1], history2[-1]
        gamer1_score += WINNING_TABLE[i][j]
        gamer2_score += WINNING_TABLE[j][i]

        if WINNING_TABLE[i][j] == WINNING_TABLE[1][0]:
            series1 += 1
            series2 = 0
            max_series1 = max(max_series1, series1)
        elif WINNING_TABLE[j][i] == WINNING_TABLE[1][0]:
            series1 = 0
            series2 += 1
            max_series2 = max(max_series2, series2)
        else: 
            series1 = 0
            series2 = 0

    return [(gamer1_score, gamer2_score), (max_series1, max_series2)]


def create_results_matrix(strategies):
    n = len(strategies)
    score_matrix = np.zeros((n, n, 2))
    series_matrix = np.zeros((n, n, 2))
    
    strategy_names = [s.__name__ for s in strategies]
    
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            scores, series = game(strat1, strat2)
            score_matrix[i, j, 0] = scores[0] 
            score_matrix[i, j, 1] = scores[1] 
            series_matrix[i, j, 0] = series[0]
            series_matrix[i, j, 1] = series[1]
    
    return strategy_names, score_matrix, series_matrix

def create_combined_plot(strategy_names, score_matrix, series_matrix):    
    n = len(strategy_names)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    

    
    total_scores = score_matrix[:,:,0] + score_matrix[:,:,1]
    im1 = ax1.imshow(total_scores, cmap='Reds', aspect='equal')
    for i in range(n):
        for j in range(n):
            score_i = int(score_matrix[i, j, 0])
            score_j = int(score_matrix[i, j, 1])
            
            text = ax1.text(j, i, f'({score_i},{score_j})',
                          ha='center', va='center', fontsize=8,
                          fontweight='bold')
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(n))
    ax1.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax1.set_yticklabels(strategy_names)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Сумма очков в паре')
    

    
    max_series = np.maximum(series_matrix[:,:,0], series_matrix[:,:,1])
    im2 = ax2.imshow(max_series, cmap='Greens', aspect='equal')
    
    for i in range(n):
        for j in range(n):
            series_i = int(series_matrix[i, j, 0])
            series_j = int(series_matrix[i, j, 1])
            
            text = ax2.text(j, i, f'({series_i},{series_j})',
                          ha='center', va='center', fontsize=8,
                          fontweight='bold')
    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.set_yticklabels([])
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Максимальная серия в паре')
    

    plt.suptitle(f'СРАВНЕНИЕ СТРАТЕГИЙ В ДИЛЕММЕ ЗАКЛЮЧЕННОГО ({ROUNDS_COUNT} раундов)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def create_summary_table(strategy_names, score_matrix):
    print("СУММАРНЫЕ РЕЗУЛЬТАТЫ ПО СТРАТЕГИЯМ")
    print("-" * 45)
    
    total_scores = []
    for i, name in enumerate(strategy_names):
        total_score = np.sum(score_matrix[i, :, 0])
        avg_score = total_score / len(strategy_names)
        total_scores.append((name, total_score, avg_score))
    
    total_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Стратегия':<12} | {'Всего очков':<12} | {'Среднее за игру':<15}")
    print("-" * 45)
    for name, total, avg in total_scores:
        print(f"{name:<12} | {total:<12.0f} | {avg:<15.2f}")

if __name__ == "__main__":
    strategies = [Alex, Bob, Clara, Denis, Emma, Frida, George]
    
    strategy_names, score_matrix, series_matrix = create_results_matrix(strategies)
    
    create_combined_plot(strategy_names, score_matrix, series_matrix)
    
    create_summary_table(strategy_names, score_matrix)
