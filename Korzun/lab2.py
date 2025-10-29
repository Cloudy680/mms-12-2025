W = [[3, 0], [5, 1]]

def alex(my_history, opponent_history, round_index):
    return 1

def bob(my_history, opponent_history, round_index):
    return 0

def clara(my_history, opponent_history, round_index):
    if round_index == 1:
        return 0
    return opponent_history[-1]

def denis(my_history, opponent_history, round_index):
    if round_index == 1:
        return 0
    return 1 - opponent_history[-1]

def emma(my_history, opponent_history, round_index):
    if round_index % 20 == 0:
        return 1
    return 0

def frida(my_history, opponent_history, round_index):
    if 1 in opponent_history:
        return 1
    return 0

def george(my_history, opponent_history, round_index):
    if round_index == 1:
        return 0
    my_last = my_history[-1]
    opp_last = opponent_history[-1]
    last_payoff = W[my_last][opp_last]
    if last_payoff in (1, 3, 5):
        return  my_last
    return  1 - my_last

def play_match(strategy_a, strategy_b, rounds=200):
    history_a = []
    history_b = []
    score_a = 0
    score_b = 0
    current_dom_a = 0
    current_dom_b = 0
    max_dom_a = 0
    max_dom_b = 0
    for r in range(1, rounds + 1):
        move_a = strategy_a(history_a, history_b, r)
        move_b = strategy_b(history_b, history_a, r)
        history_a.append(move_a)
        history_b.append(move_b)
        s_a = W[move_a][move_b]
        s_b = W[move_b][move_a]
        score_a += s_a
        score_b += s_b
        if move_a == 1 and move_b == 0:
            current_dom_a += 1
            current_dom_b = 0
        elif move_a == 0 and move_b == 1:
            current_dom_b += 1
            current_dom_a = 0
        else:
            current_dom_a = 0
            current_dom_b = 0
        if current_dom_a > max_dom_a:
            max_dom_a = current_dom_a
        if current_dom_b > max_dom_b:
            max_dom_b = current_dom_b
    return score_a, score_b, max_dom_a, max_dom_b

def run_tournament():
    strategies = {
        "Alex": alex,
        "Bob": bob,
        "Clara": clara,
        "Denis": denis,
        "Emma": emma,
        "Frida": frida,
        "George": george,
    }
    names = list(strategies.keys())
    total_scores = {name: 0 for name in names}
    max_dom_runs = {name: 0 for name in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            sa, sb, da, db = play_match(strategies[a], strategies[b])
            total_scores[a] += sa
            total_scores[b] += sb
            if da > max_dom_runs[a]:
                max_dom_runs[a] = da
            if db > max_dom_runs[b]:
                max_dom_runs[b] = db
    return names, total_scores, max_dom_runs

def print_tables(names, total_scores, max_dom_runs):
    print("Сравнение стратегий. Игра: Дилемма заключенного, 200 партий")
    print()
    print("Таблица 1. Общее число очков (сумма по всем матчам)")
    header = f"{'Стратегия':<12}{'Очки':>10}"
    print(header)
    print("-" * len(header))
    for name, score in sorted(total_scores.items(), key=lambda x: (-x[1], x[0])):
        print(f"{name:<12}{score:>10}")
    print()
    print("Таблица 2. Длина наибольшей доминирующей серии (5 против 0)")
    header2 = f"{'Стратегия':<12}{'Длина':>10}"
    print(header2)
    print("-" * len(header2))
    for name, dom in sorted(max_dom_runs.items(), key=lambda x: (-x[1], x[0])):
        print(f"{name:<12}{dom:>10}")

if __name__ == "__main__":
    names, total_scores, max_dom_runs = run_tournament()
    print_tables(names, total_scores, max_dom_runs)


