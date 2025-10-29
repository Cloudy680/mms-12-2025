import pandas as pd

def Alex_strategy(last_opponent, part_count, first_game):
    return 1

def Bob_strategy(last_opponent, part_count, first_game):
    return 0

def Clara_strategy(last_opponent, part_count, first_game):
    if first_game:
        return 0

    return last_opponent

def Denis_strategy(last_opponent, part_count, first_game):
    if first_game:
        return 0

    if last_opponent == 1:
        return 0
    return 1

def Emma_strategy(last_opponent, part_count, first_game):
    if part_count % 20 == 0:
        return 1
    return 0


def Frida_strategy(last_opponent, part_count, first_game):
    if not hasattr(Frida_strategy, 'decis'):
        Frida_strategy.decis = 0

    if first_game:
        Frida_strategy.decis = 0

    if last_opponent == 1:
        Frida_strategy.decis = 1

    return Frida_strategy.decis

def George_strategy(last_opponent, part_count, first_game):
    if not hasattr(George_strategy, 'betrayal_count'):
        George_strategy.betrayal_count = 0

    if first_game:
        George_strategy.betrayal_count = 0
        return 0

    if last_opponent == 1:
        George_strategy.betrayal_count += 1

    if George_strategy.betrayal_count >= GAME_COUNT // 4:
        return 1

    return last_opponent

W = ((3, 0),
     (5, 1))


participants = {
    "Alex": Alex_strategy,
    "Bob": Bob_strategy,
    "Clara": Clara_strategy,
    "Denis": Denis_strategy,
    "Emma": Emma_strategy,
    "Frida": Frida_strategy,
    "George": George_strategy
}
GAME_COUNT = 200

def main():
    strategy_versus = []

    part_keys = list(participants.keys())

    for first_parti in range(len(part_keys)):
        results = []
        for second_parti in range(len(part_keys)):
            versus = f"{part_keys[first_parti]} versus {part_keys[second_parti]}"
            first_opponent_score = 0
            last_opponent_for_first, last_opponent_for_second = 0, 0
            first_game = True

            dominate_serie, current_serie = 0, 0
            for game_count in range(1, GAME_COUNT + 1):

                f_op_decision = participants[part_keys[first_parti]](last_opponent_for_first, game_count, first_game)

                sec_op_desicion = participants[part_keys[second_parti]](last_opponent_for_second, game_count, first_game)

                last_opponent_for_first = sec_op_desicion
                last_opponent_for_second = f_op_decision

                first_opponent_score += W[f_op_decision][sec_op_desicion]
                if W[f_op_decision][sec_op_desicion] == 5:
                    current_serie += 1
                else:
                    current_serie = 0

                if dominate_serie < current_serie:
                    dominate_serie = current_serie

                first_game = False

            results.append(f'{first_opponent_score} : {dominate_serie}')

        strategy_versus.append(results)

    result = pd.DataFrame(strategy_versus, columns=part_keys, index=part_keys)
    print(result)

if __name__ == "__main__":
    main()