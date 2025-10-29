import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
import random
from collections import Counter
import math

class Strategy(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        pass

class Alex(Strategy):
    def get_name(self) -> str:
        return "Alex"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        return 1

class Bob(Strategy):
    def get_name(self) -> str:
        return "Bob"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        return 0

class Clara(Strategy):
    def get_name(self) -> str:
        return "Clara"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        return opponent_moves[-1]

class Denis(Strategy):
    def get_name(self) -> str:
        return "Denis"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        return 1 - opponent_moves[-1]

class Emma(Strategy):
    def get_name(self) -> str:
        return "Emma"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(my_moves) % 20 == 19:
            return 1
        return 0

class Frida(Strategy):
    def get_name(self) -> str:
        return "Frida"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        if 1 in opponent_moves:
            return 1
        return 0

class George(Strategy):
    def get_name(self) -> str:
        return "George"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        
        if len(opponent_moves) >= 10:
            last_ten = opponent_moves[-10:]
            if all(move == 1 for move in last_ten):
                return 0
        
        return opponent_moves[-1]

class Hank(Strategy):
    def get_name(self) -> str:
        return "Hank"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        return random.randint(0, 1)

class Ivan(Strategy):
    def get_name(self) -> str:
        return "Ivan"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        return 0 if random.random() < 0.9 else 1

class Jack(Strategy):
    def get_name(self) -> str:
        return "Jack"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        
        if opponent_moves[-1] == 0:
            return 0
        else:
            return 0 if random.random() < 0.25 else 1

class Kevin(Strategy):
    def get_name(self) -> str:
        return "Kevin"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        
        if random.random() < 0.25:
            return 1 - opponent_moves[-1]
        else:
            return opponent_moves[-1]

class Lucas(Strategy):
    def __init__(self):
        self.period = random.randint(1, 50)
    
    def get_name(self) -> str:
        return f"Lucas(p{self.period})"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(my_moves) % self.period == self.period - 1:
            return 1
        return 0

class Max(Strategy):
    def __init__(self):
        self.current_move = 0
        self.moves_left = random.randint(0, 20)
    
    def get_name(self) -> str:
        return "Max"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if self.moves_left == 0:
            self.current_move = 1 - self.current_move
            self.moves_left = random.randint(0, 20)
        
        self.moves_left -= 1
        return self.current_move

class Natan(Strategy):
    def get_name(self) -> str:
        return "Natan"
    
    def make_move(self, my_moves: List[int], opponent_moves: List[int]) -> int:
        if len(opponent_moves) == 0:
            return 0
        
        recent_moves = opponent_moves[-10:] if len(opponent_moves) >= 10 else opponent_moves
        cooperation_rate = sum(1 for move in recent_moves if move == 0) / len(recent_moves)
        
        cooperation_prob = 0.2 + 0.7 * cooperation_rate
        
        return 0 if random.random() < cooperation_prob else 1

class PrisonersDilemma:
    def __init__(self):
        self.payoff = np.array([
            [3, 0],
            [5, 1]
        ])
    
    def play_game(self, player_a: Strategy, player_b: Strategy, rounds: int = 200) -> Dict:
        moves_a, moves_b = [], []
        score_a, score_b = 0, 0
        
        for round_num in range(rounds):
            move_a = player_a.make_move(moves_a, moves_b)
            move_b = player_b.make_move(moves_b, moves_a)
            
            moves_a.append(move_a)
            moves_b.append(move_b)
            
            score_a += self.payoff[move_a, move_b]
            score_b += self.payoff[move_b, move_a]
        
        return {
            'score_a': score_a,
            'score_b': score_b,
            'moves_a': moves_a,
            'moves_b': moves_b
        }

def calculate_sample_statistics(samples: List[float]) -> Dict[str, float]:
    n = len(samples)
    if n == 0:
        return {}
    
    mean_value = np.mean(samples)
    variance = np.var(samples, ddof=1)
    median_value = np.median(samples)
    
    if n > 1:
        k = int(math.log2(1 + n))
        k = max(2, k)
        
        min_val, max_val = min(samples), max(samples)
        
        if min_val == max_val:
            mode_value = min_val
        else:
            interval_width = (max_val - min_val) / k
            
            frequencies = [0] * k
            for value in samples:
                interval_idx = min(int((value - min_val) / interval_width), k - 1)
                frequencies[interval_idx] += 1
            
            max_freq_idx = np.argmax(frequencies)
            mode_interval_start = min_val + max_freq_idx * interval_width
            mode_interval_end = mode_interval_start + interval_width
            mode_value = (mode_interval_start + mode_interval_end) / 2
    else:
        mode_value = samples[0]
    
    return {
        'mean': mean_value,
        'variance': variance,
        'median': median_value,
        'mode': mode_value,
        'std': math.sqrt(variance),
        'min': min(samples),
        'max': max(samples),
        'count': n
    }

def simulate_multiple_games(game: PrisonersDilemma, player_a: Strategy, player_b: Strategy, 
                           num_simulations: int = 1000, rounds: int = 200) -> Dict:
    scores_a = []
    
    for _ in range(num_simulations):
        result = game.play_game(player_a, player_b, rounds)
        scores_a.append(result['score_a'])
    
    statistics = calculate_sample_statistics(scores_a)
    
    return {
        'scores': scores_a,
        'statistics': statistics
    }

def print_comprehensive_results(all_results: List[List[Dict]], strategy_names: List[str]):
    n = len(strategy_names)
    
    print("\n" + "="*120)
    print("ТАБЛИЦА СРЕДНИХ ЗНАЧЕНИЙ ОЧКОВ")
    print("="*120)
    print(f"{'':<15}", end="")
    for name in strategy_names:
        print(f"{name:<10}", end="")
    print()
    
    for i in range(n):
        print(f"{strategy_names[i]:<15}", end="")
        for j in range(n):
            mean_score = all_results[i][j]['statistics']['mean']
            print(f"{mean_score:<10.1f}", end="")
        print()
    
    print("\n" + "="*120)
    print("ТАБЛИЦА ДИСПЕРСИЙ ОЧКОВ")
    print("="*120)
    print(f"{'':<15}", end="")
    for name in strategy_names:
        print(f"{name:<10}", end="")
    print()
    
    for i in range(n):
        print(f"{strategy_names[i]:<15}", end="")
        for j in range(n):
            variance = all_results[i][j]['statistics']['variance']
            print(f"{variance:<10.1f}", end="")
        print()
    
    print("\n" + "="*120)
    print("ТАБЛИЦА МЕДИАН ОЧКОВ")
    print("="*120)
    print(f"{'':<15}", end="")
    for name in strategy_names:
        print(f"{name:<10}", end="")
    print()
    
    for i in range(n):
        print(f"{strategy_names[i]:<15}", end="")
        for j in range(n):
            median = all_results[i][j]['statistics']['median']
            print(f"{median:<10.1f}", end="")
        print()

def print_detailed_strategy_analysis(all_results: List[List[Dict]], strategy_names: List[str]):
    n = len(strategy_names)
    
    print("\n" + "="*120)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ СТРАТЕГИЙ")
    print("="*120)
    
    total_scores = []
    for i in range(n):
        total_score = sum(all_results[i][j]['statistics']['mean'] for j in range(n))
        variance_avg = np.mean([all_results[i][j]['statistics']['variance'] for j in range(n)])
        total_scores.append({
            'name': strategy_names[i],
            'total': total_score,
            'avg_variance': variance_avg,
            'index': i
        })
    
    total_scores.sort(key=lambda x: x['total'], reverse=True)
    
    print("\nРейтинг стратегий по общей эффективности:")
    print("-" * 80)
    for rank, strategy in enumerate(total_scores, 1):
        print(f"{rank:2d}. {strategy['name']:<15} | "
              f"Сумма очков: {strategy['total']:>7.1f} | "
              f"Ср. дисперсия: {strategy['avg_variance']:>6.1f}")
    
    print("\n" + "="*120)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА ДЛЯ ТОП-3 СТРАТЕГИЙ")
    print("="*120)
    
    for rank in range(min(3, len(total_scores))):
        idx = total_scores[rank]['index']
        strategy_name = strategy_names[idx]
        
        print(f"\n{rank + 1}. {strategy_name}:")
        print("-" * 50)
        
        all_scores = []
        for j in range(n):
            all_scores.extend(all_results[idx][j]['scores'])
        
        overall_stats = calculate_sample_statistics(all_scores)
        
        print(f"   Среднее:      {overall_stats['mean']:.1f}")
        print(f"   Медиана:      {overall_stats['median']:.1f}")
        print(f"   Мода:         {overall_stats['mode']:.1f}")
        print(f"   Дисперсия:    {overall_stats['variance']:.1f}")
        print(f"   Стандартное отклонение: {overall_stats['std']:.1f}")
        print(f"   Диапазон:     [{overall_stats['min']:.0f}, {overall_stats['max']:.0f}]")
        print(f"   Количество симуляций: {overall_stats['count']}")

def main():
    deterministic_strategies = [Alex(), Bob(), Clara(), Denis(), Emma(), Frida(), George()]
    stochastic_strategies = [Hank(), Ivan(), Jack(), Kevin(), Lucas(), Max(), Natan()]
    
    all_strategies = deterministic_strategies + stochastic_strategies
    strategy_names = [strategy.get_name() for strategy in all_strategies]
    n = len(all_strategies)
    
    game = PrisonersDilemma()
    
    all_results = [[{} for _ in range(n)] for _ in range(n)]
    
    print("ЛАБОРАТОРНАЯ РАБОТА №3: СРАВНЕНИЕ ДЕТЕРМИНИРОВАННЫХ И СТОХАСТИЧЕСКИХ СТРАТЕГИЙ")
    print("=" * 100)
    print("Проведение симуляций...")
    
    for i in range(n):
        for j in range(n):
            player_a = all_strategies[i]
            player_b = all_strategies[j]
            
            is_stochastic = (i >= len(deterministic_strategies) or 
                           j >= len(deterministic_strategies) or
                           isinstance(player_a, (Lucas, Max)) or
                           isinstance(player_b, (Lucas, Max)))
            
            if is_stochastic:
                num_simulations = 1000
                result = simulate_multiple_games(game, player_a, player_b, num_simulations)
            else:
                single_result = game.play_game(player_a, player_b)
                result = {
                    'scores': [single_result['score_a']],
                    'statistics': calculate_sample_statistics([single_result['score_a']])
                }
            
            all_results[i][j] = result
    
    print_comprehensive_results(all_results, strategy_names)
    print_detailed_strategy_analysis(all_results, strategy_names)
    
    print("\n" + "="*120)
    print("АНАЛИЗ СТОХАСТИЧЕСКИХ СТРАТЕГИЙ")
    print("="*120)
    
    stochastic_indices = list(range(len(deterministic_strategies), n))
    for idx in stochastic_indices:
        strategy_name = strategy_names[idx]
        variances = [all_results[idx][j]['statistics']['variance'] for j in range(n)]
        avg_variance = np.mean(variances)
        
        print(f"{strategy_name:<15}: Средняя дисперсия = {avg_variance:.1f}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
