from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import random

PLAYS_AMOUNT = 200
SIMULATIONS = 100


class Strategy(ABC):
    @abstractmethod
    def make_move(self, opp_history, test_number):
        pass

    @abstractmethod
    def reset(self):
        self.points = 0


# Alex: always 1
class Alex(Strategy):
    points = 0
    name = "Alex"

    def make_move(self, opp_history, test_number):
        return 1

    def reset(self):
        self.points = 0


# Bob: always 0
class Bob(Strategy):
    points = 0
    name = "Bob"

    def make_move(self, opp_history, test_number):
        return 0

    def reset(self):
        self.points = 0


# Clara: 1 than prev opp move
class Clara(Strategy):
    points = 0
    name = "Clara"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        else:
            return opp_history[-1]

    def reset(self):
        self.points = 0


# Denis: 1 than opposite opp move
class Denis(Strategy):
    points = 0
    name = "Denis"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        else:
            if opp_history[-1] == 1:
                return 0
            else:
                return 1

    def reset(self):
        self.points = 0


# Emma: always 1 but every 20 play is 0
class Emma(Strategy):
    points = 0
    name = "Emma"

    def make_move(self, opp_history, test_number):
        if test_number % 20 == 0:
            return 1
        else:
            return 0

    def reset(self):
        self.points = 0


# Frida: while opp does 1 is 1, if opp does 0 every next move is 0
class Frida(Strategy):
    if_opp_did_1 = False
    points = 0
    name = "Frida"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        if opp_history[-1] == 1:
            self.if_opp_did_1 = True
        if self.if_opp_did_1:
            return 1
        else:
            return 0

    def reset(self):
        self.points = 0
        self.if_opp_did_1 = False


# George: my deterministic strategy
class George(Strategy):
    points = 0
    opp_1_count = 0
    name = "George"

    def make_move(self, opp_history, test_number):
        if test_number > 1 and opp_history[-1] == 1:
            self.opp_1_count += 1
        if test_number == 1:
            return 0
        else:
            if self.opp_1_count == 10:
                self.opp_1_count = 0
                return 0
            else:
                return opp_history[-1]

    def reset(self):
        self.points = 0
        self.opp_1_count = 0


class Hank(Strategy):
    points = 0
    name = "Hank"

    def make_move(self, opp_history, test_number):
        return random.choices([0, 1], weights=[0.5, 0.5])[0]

    def reset(self):
        self.points = 0


class Ivan(Strategy):
    points = 0
    name = "Ivan"

    def make_move(self, opp_history, test_number):
        return random.choices([0, 1], weights=[0.9, 0.1])[0]

    def reset(self):
        self.points = 0


class Jack(Strategy):
    points = 0
    name = "Jack"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        if opp_history[-1] == 0:
            return 0
        else:
            return random.choices([0, 1], weights=[0.25, 0.75])[0]

    def reset(self):
        self.points = 0


class Kevin(Strategy):
    points = 0
    name = "Kevin"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        if opp_history[-1] == 0:
            return random.choices([0, 1], weights=[0.75, 0.25])[0]
        else:
            return random.choices([0, 1], weights=[0.25, 0.75])[0]

    def reset(self):
        self.points = 0


class Lucas(Strategy):
    def __init__(self):
        self.points = 0
        self.name = "Lucas"
        self.n = random.randint(1, 50)
        self.moves_counter = 0

    def make_move(self, opp_history, test_number):
        if self.moves_counter == self.n:
            self.moves_counter = 0
            return 1
        else:
            self.moves_counter += 1
            return 0

    def reset(self):
        self.points = 0
        self.n = random.randint(1, 50)
        self.moves_counter = 0


class Max(Strategy):
    def __init__(self):
        self.points = 0
        self.name = "Max"
        self.n = random.randint(1, 20)
        self.moves_counter = 0
        self.now = 0

    def make_move(self, opp_history, test_number):
        if self.moves_counter == self.n:
            self.moves_counter = 0
            self.n = random.randint(1, 20)
            self.now = 1 - self.now
            return self.now
        else:
            self.moves_counter += 1
            return self.now

    def reset(self):
        self.points = 0
        self.moves_counter = 0
        self.n = random.randint(1, 20)
        self.now = 0


class Natan(Strategy):
    points = 0
    name = "Natan"

    def make_move(self, opp_history, test_number):
        if test_number == 1:
            return 0
        else:
            prob = sum(opp_history) / 100
            if prob > 1:
                counter = 0
                temp = prob
                while temp > 1:
                    temp -= 1
                    counter += 1
                prob = prob - counter
            return random.choices([0, 1], weights=[1 - prob, prob])[0]

    def reset(self):
        self.points = 0


results = [[3, 0], [5, 1]]


def print_dataframe(title, df):
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    with pd.option_context('display.width', None,
                           'display.max_columns', None,
                           'display.float_format', '{:.1f}'.format,
                           'display.colheader_justify', 'center'):
        print(df)

    print(f"{'=' * 80}")


def print_statistics(title, data_dict, value_name):
    print(f"\n{title}")

    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)

    for i, (player, value) in enumerate(sorted_items, 1):
        print(f"{i:2}. {player:<10} {value:>10} {value_name}")


def calculate_mode(data):
    if not data:
        return 0

    n = len(data)
    k = int(np.log2(1 + n))

    if k < 1:
        k = 1

    hist, bin_edges = np.histogram(data, bins=k)
    max_bin_index = np.argmax(hist)

    mode_estimate = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    return mode_estimate


def calculate_variance(data):

    if len(data) == 0:
        return 0

    n = len(data)
    mean = sum(data) / n

    sum_of_squares = 0
    for value in data:
        deviation = value - mean
        sum_of_squares += deviation * deviation

    variance = sum_of_squares / (n - 1)

    return variance


def main():
    # players:
    alex = Alex()
    bob = Bob()
    clara = Clara()
    denis = Denis()
    emma = Emma()
    frida = Frida()
    george = George()
    hank = Hank()
    ivan = Ivan()
    jack = Jack()
    kevin = Kevin()
    lucas = Lucas()
    maxim = Max()
    natan = Natan()

    players = [alex, bob, clara, denis, emma, frida, george, hank, ivan, jack, kevin, lucas, maxim, natan]
    player_names = [p.name for p in players]

    stochastic_players = [hank, ivan, jack, kevin, lucas, maxim, natan]

    # result matrices initialisation
    points_matrix = [[0 for _ in players] for _ in players]
    wins_matrix = [[0 for _ in players] for _ in players]
    series_matrix = [[0 for _ in players] for _ in players]

    # Matrices for statistics on stochastic strategies
    avg_points_matrix = [[0 for _ in players] for _ in players]
    median_points_matrix = [[0 for _ in players] for _ in players]
    mode_points_matrix = [[0 for _ in players] for _ in players]
    variance_points_matrix = [[0 for _ in players] for _ in players]

    # players wins
    wins = {"Alex": 0, "Bob": 0, "Clara": 0, "Denis": 0, "Emma": 0, "Frida": 0, "George": 0, "Hank": 0, "Ivan": 0,
            "Jack": 0, "Kevin": 0, "Lucas": 0, "Max": 0, "Natan": 0}
    # player total points
    total_points = {"Alex": 0, "Bob": 0, "Clara": 0, "Denis": 0, "Emma": 0, "Frida": 0, "George": 0, "Hank": 0,
                    "Ivan": 0, "Jack": 0, "Kevin": 0, "Lucas": 0, "Max": 0, "Natan": 0}
    # player series counter
    max_dominant_series = {"Alex": 0, "Bob": 0, "Clara": 0, "Denis": 0, "Emma": 0, "Frida": 0, "George": 0, "Hank": 0,
                           "Ivan": 0, "Jack": 0, "Kevin": 0, "Lucas": 0, "Max": 0, "Natan": 0}

    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players):
            all_points1 = []
            all_points2 = []
            all_series1 = []
            all_series2 = []
            all_wins = []

            for sim in range(SIMULATIONS):
                player1.reset()
                player2.reset()

                history1 = []
                history2 = []

                current_series1 = 0
                current_series2 = 0
                max_series1 = 0
                max_series2 = 0

                for game_num in range(1, PLAYS_AMOUNT + 1):
                    move1 = player1.make_move(history2, game_num)
                    move2 = player2.make_move(history1, game_num)

                    history1.append(move1)
                    history2.append(move2)

                    player1.points += results[move1][move2]
                    player2.points += results[move2][move1]

                    if results[move1][move2] == 5 and results[move2][move1] == 0:
                        current_series1 += 1
                        current_series2 = 0
                        max_series1 = max(max_series1, current_series1)
                    elif results[move2][move1] == 5 and results[move1][move2] == 0:
                        current_series2 += 1
                        current_series1 = 0
                        max_series2 = max(max_series2, current_series2)
                    else:
                        current_series1 = 0
                        current_series2 = 0


                all_points1.append(player1.points)
                all_points2.append(player2.points)
                all_series1.append(max_series1)
                all_series2.append(max_series2)

                if player1.points > player2.points:
                    all_wins.append(1)
                elif player1.points == player2.points:
                    all_wins.append(0.5)
                else:
                    all_wins.append(0)


            all_points1_sum = np.sum(all_points1)
            all_points2_sum = np.sum(all_points2)
            avg_points1 = np.mean(all_points1)
            avg_points2 = np.mean(all_points2)
            median_points1 = np.median(all_points1)
            median_points2 = np.median(all_points2)
            mode_points1 = calculate_mode(all_points1)
            mode_points2 = calculate_mode(all_points2)
            variance_points1 = calculate_variance(all_points1)
            variance_points2 = calculate_variance(all_points2)

            avg_series1 = np.mean(all_series1)
            avg_series2 = np.mean(all_series2)
            win_rate = np.mean(all_wins)

            avg_points_matrix[i][j] = avg_points1
            avg_points_matrix[j][i] = avg_points2
            median_points_matrix[i][j] = median_points1
            median_points_matrix[j][i] = median_points2
            mode_points_matrix[i][j] = mode_points1
            mode_points_matrix[j][i] = mode_points2
            variance_points_matrix[i][j] = variance_points1
            variance_points_matrix[j][i] = variance_points2

            points_matrix[i][j] = all_points1_sum
            series_matrix[i][j] = avg_series1

            if i == j:
                actual_points = all_points1_sum / 2
                points_matrix[i][j] = actual_points
                wins_matrix[i][j] = 0.5
                avg_points_matrix[i][j] = avg_points1 / 2
                median_points_matrix[i][j] = median_points1 / 2
                mode_points_matrix[i][j] = mode_points1 / 2
                total_points[player1.name] += actual_points
                max_dominant_series[player1.name] = max(max_dominant_series[player1.name], avg_series1)
            else:
                points_matrix[j][i] = all_points2_sum
                series_matrix[j][i] = avg_series2

                if win_rate > 0.5:
                    wins_matrix[i][j] = 1
                    wins_matrix[j][i] = 0
                    wins[player1.name] += 1
                elif win_rate == 0.5:
                    wins_matrix[i][j] = 0.5
                    wins_matrix[j][i] = 0.5
                    wins[player1.name] += 0.5
                    wins[player2.name] += 0.5
                else:
                    wins_matrix[i][j] = 0
                    wins_matrix[j][i] = 1
                    wins[player2.name] += 1

                total_points[player1.name] += all_points1_sum
                total_points[player2.name] += all_points2_sum

                max_dominant_series[player1.name] = max(max_dominant_series[player1.name], avg_series1)
                max_dominant_series[player2.name] = max(max_dominant_series[player2.name], avg_series2)


    points_df = pd.DataFrame(points_matrix, index=player_names, columns=player_names)
    wins_df = pd.DataFrame(wins_matrix, index=player_names, columns=player_names)
    series_df = pd.DataFrame(series_matrix, index=player_names, columns=player_names)

    avg_points_df = pd.DataFrame(avg_points_matrix, index=player_names, columns=player_names)
    median_points_df = pd.DataFrame(median_points_matrix, index=player_names, columns=player_names)
    mode_points_df = pd.DataFrame(mode_points_matrix, index=player_names, columns=player_names)
    variance_points_df = pd.DataFrame(variance_points_matrix, index=player_names, columns=player_names)

    print_dataframe("POINTS", points_df)
    print_dataframe("WINS", wins_df)
    print_dataframe("MAX SERIA", series_df)

    print_dataframe("AVERAGE POINTS (stochastic)", avg_points_df)
    print_dataframe("MEDIAN POINTS (stochastic)", median_points_df)
    print_dataframe("MODE POINTS (stochastic)", mode_points_df)
    print_dataframe("VARIANCE POINTS (stochastic)", variance_points_df)


    with open("result.txt", "w") as fout:

        fout.write("POINTS:\n")
        fout.write(points_df.to_string() + "\n\n")

        fout.write("WINS:\n")
        fout.write(wins_df.to_string() + "\n\n")

        fout.write("MAX SERIA:\n")
        fout.write(series_df.to_string() + "\n\n")

        fout.write("AVERAGE POINTS (stochastic):\n")
        fout.write(avg_points_df.to_string() + "\n\n")

        fout.write("MEDIAN POINTS (stochastic):\n")
        fout.write(median_points_df.to_string() + "\n\n")

        fout.write("MODE POINTS (stochastic):\n")
        fout.write(mode_points_df.to_string() + "\n\n")

        fout.write("VARIANCE POINTS (stochastic):\n")
        fout.write(variance_points_df.to_string() + "\n\n")


if __name__ == "__main__":
    main()
