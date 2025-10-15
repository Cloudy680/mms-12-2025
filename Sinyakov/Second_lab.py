import numpy as np
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

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
        
        if len(opponent_moves) >= 3:
            last_three = opponent_moves[-3:]
            if all(move == 1 for move in last_three):
                return 0
        
        return opponent_moves[-1]

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
    
    def calculate_dominant_streak(self, moves_a: List[int], moves_b: List[int]) -> int:
        max_streak = 0
        current_streak = 0
        
        for move_a, move_b in zip(moves_a, moves_b):
            if (move_a == 1 and move_b == 0) or (move_a == 0 and move_b == 1):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak

def print_results_table(scores: np.ndarray, streaks: np.ndarray, strategy_names: List[str]):
    n = len(strategy_names)
    
    print("\n" + "="*80)
    print("ТАБЛИЦА ОБЩИХ ОЧКОВ:")
    print("="*80)
    print(f"{'':<12}", end="")
    for name in strategy_names:
        print(f"{name:<12}", end="")
    print()
    
    for i in range(n):
        print(f"{strategy_names[i]:<12}", end="")
        for j in range(n):
            print(f"{scores[i, j]:<12}", end="")
        print()
    
    print("\n" + "="*80)
    print("ТАБЛИЦА ДЛИН НАИБОЛЬШИХ ДОМИНИРУЮЩИХ СЕРИЙ:")
    print("="*80)
    print(f"{'':<12}", end="")
    for name in strategy_names:
        print(f"{name:<12}", end="")
    print()
    
    for i in range(n):
        print(f"{strategy_names[i]:<12}", end="")
        for j in range(n):
            print(f"{streaks[i, j]:<12}", end="")
        print()

def analyze_results(scores: np.ndarray, strategy_names: List[str]):
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("="*80)
    
    total_points = np.sum(scores, axis=1)
    
    print("\nОбщее количество очков для каждой стратегии:")
    for i, name in enumerate(strategy_names):
        print(f"{name}: {total_points[i]} очков")
    
    best_score = np.max(total_points)
    best_strategies = [strategy_names[i] for i in range(len(strategy_names)) 
                      if total_points[i] == best_score]
    
    print(f"\nЛучшие стратегии по общему количеству очков:")
    for strategy in best_strategies:
        print(f"- {strategy}: {best_score} очков")
    
    average_points = np.mean(scores, axis=1)
    print(f"\nСреднее количество очков за игру:")
    for i, name in enumerate(strategy_names):
        print(f"{name}: {average_points[i]:.1f} очков")

def main():
    strategies = [
        Alex(), Bob(), Clara(), 
        Denis(), Emma(), Frida(), George()
    ]
    
    strategy_names = [strategy.get_name() for strategy in strategies]
    n = len(strategies)
    
    game = PrisonersDilemma()
    
    total_scores = np.zeros((n, n), dtype=int)
    dominant_streaks = np.zeros((n, n), dtype=int)
    
    print("Проведение экспериментов...")
    for i in range(n):
        for j in range(n):
            result = game.play_game(strategies[i], strategies[j])
            total_scores[i, j] = result['score_a']
            dominant_streaks[i, j] = game.calculate_dominant_streak(
                result['moves_a'], result['moves_b']
            )
    
    print_results_table(total_scores, dominant_streaks, strategy_names)
    analyze_results(total_scores, strategy_names)
    
    print("\n" + "="*80)
    print("ПРИМЕР ИГРЫ: Clara vs Alex")
    print("="*80)
    
    example_result = game.play_game(Clara(), Alex(), rounds=10)
    print("Ходы Clara:", example_result['moves_a'])
    print("Ходы Alex: ", example_result['moves_b'])
    print("Очки Clara:", example_result['score_a'])
    print("Очки Alex: ", example_result['score_b'])

if __name__ == "__main__":
    main()
