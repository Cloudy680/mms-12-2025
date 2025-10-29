from abc import ABC, abstractmethod
import itertools

PLAYS_AMOUNT = 200


class Strategy(ABC):
    @abstractmethod
    def make_move(self, opp_move, test_number):
        pass

    @abstractmethod
    def reset(self):
        self.points = 0

#Alex: always 1
class Alex(Strategy):

    points = 0
    name = "Alex"

    def make_move(self, opp_move, test_number):
        return 1

    def reset(self):
        self.points = 0

#Bob: always 0
class Bob(Strategy):

    points = 0
    name = "Bob"

    def make_move(self, opp_move, test_number):
        return 0

    def reset(self):
        self.points = 0

#Clara: 1 than prev opp move
class Clara(Strategy):

    points = 0
    name = "Clara"

    def make_move(self, opp_move, test_number):
        if test_number == 1:
            return 0
        else:
            return opp_move

    def reset(self):
        self.points = 0
#Denis: 1 than opposite opp move
class Denis(Strategy):

    points = 0
    name = "Denis"

    def make_move(self, opp_move, test_number):
        if test_number == 1:
            return 0
        else:
            if opp_move == 1:
                return 0
            else:
                return 1

    def reset(self):
        self.points = 0

#Emma: always 1 but every 20 play is 0
class Emma(Strategy):

    points = 0
    name = "Emma"

    def make_move(self, opp_move, test_number):
        if test_number % 20 == 0:
            return 1
        else:
            return 0

    def reset(self):
        self.points = 0

#Frida: while opp does 1 is 1, if opp does 0 every next move is 0
class Frida(Strategy):

    if_opp_did_1 = False
    points = 0
    name = "Frida"

    def make_move(self, opp_move, test_number):
        if test_number == 1:
            return 0
        if opp_move == 1:
            self.if_opp_did_1 = True
        if self.if_opp_did_1:
            return 1
        else:
            return 0

    def reset(self):
        self.points = 0
        self.if_opp_did_1 = False


class George(Strategy):

    points = 0
    opp_1_count = 0
    name = "George"


    def make_move(self, opp_move, test_number):
        if opp_move == 1:
            self.opp_1_count += 1
        if test_number == 1:
            return 0
        else:
            if self.opp_1_count == 10:
                self.opp_1_count = 0
                return 0
            else:
                return opp_move

    def reset(self):
        self.points = 0


results = [[3, 0], [5, 1]]



def main():

    #players:
    alex = Alex()
    bob = Bob()
    clara = Clara()
    denis = Denis()
    emma = Emma()
    frida = Frida()
    george = George()


    #players
    players = [alex, bob, clara, denis, emma, frida, george]

    #players wins
    wins = {"Alex" : 0, "Bob" : 0, "Clara" : 0, "Denis" : 0, "Emma" : 0, "Frida" : 0, "George": 0}
    #player total points
    total_points = {"Alex": 0, "Bob": 0, "Clara": 0, "Denis": 0, "Emma": 0, "Frida": 0, "George": 0}
    #player series counter
    max_dominant_series = {"Alex": 0, "Bob": 0, "Clara": 0, "Denis": 0, "Emma": 0, "Frida": 0, "George": 0}

    # players_pairs = []
    # for i in players_copy:
    #     for j in players_copy:
    #         players_pairs.append((i, j))
    #     players_copy.remove(i)
    # for pair in players_pairs:
    #     print(f"{pair[0].name}, {pair[1].name}")

    players_pairs = list(itertools.combinations_with_replacement(players, 2))

    # for pair in players_pairs:
    #     print(f"{pair[0].name}, {pair[1].name}")


    for pair in players_pairs:
        pair[0].reset()
        pair[1].reset()
        a_move = None
        b_move = None
        current_series_a = 0
        current_series_b = 0
        max_series_a = 0
        max_series_b = 0
        for i in range(1, PLAYS_AMOUNT + 1):
            a_move = pair[0].make_move(b_move, i)
            b_move = pair[1].make_move(a_move, i)
            pair[0].points += results[a_move][b_move]
            pair[1].points += results[b_move][a_move]

            if results[a_move][b_move] == 5 and results[b_move][a_move] == 0:
                current_series_a += 1
                current_series_b = 0
                max_series_a = max(max_series_a, current_series_a)
            elif results[b_move][a_move] == 5 and results[a_move][b_move] == 0:
                current_series_b += 1
                current_series_a = 0
                max_series_b = max(max_series_b, current_series_b)
            else:
                current_series_a = 0
                current_series_b = 0

        if pair[0].name == pair[1].name:
            total_points[pair[0].name] += pair[0].points // 2
        else:
            total_points[pair[0].name] += pair[0].points
            total_points[pair[1].name] += pair[1].points

        max_dominant_series[pair[0].name] = max(max_dominant_series[pair[0].name], max_series_a)
        max_dominant_series[pair[1].name] = max(max_dominant_series[pair[1].name], max_series_b)

        print("*********************************")
        if pair[0].name == pair[1].name:
            print(pair[0].name)
            print(pair[0].points // 2)
            print()
            print(pair[1].name)
            print(pair[1].points // 2)
        else:
            print(pair[0].name)
            print(pair[0].points)
            print()
            print(pair[1].name)
            print(pair[1].points)

        if pair[0].points > pair[1].points:
            wins[pair[0].name] += 1
        elif pair[0].points == pair[1].points:
            wins[pair[0].name] += 0.5
            wins[pair[1].name] += 0.5
        else:
            wins[pair[1].name] += 1


    print("Wins: ", wins)

    print("Total points:", total_points)
    print("Max dominant series:", max_dominant_series)

    print("\nYou can find all statistics in file result.txt")

    with open("result.txt", "w") as fout:
        # fout.write("Wins statistics:\n")
        # for player in wins:
        #     fout.write(f"{player}   -   {str(wins[player])}\n")
        # fout.write("\n\n")

        # fout.write("Winners of the tournament:\n")
        # max_wins = max(wins.values())
        # for player in wins:
        #     if wins[player] == max_wins:
        #         fout.write(f"{str(player)}   -   {str(wins[player])}\n")

        fout.write("Total points statistics:\n")
        for player in total_points:
            fout.write(f"{player}   -   {str(total_points[player])}\n")

        fout.write("\nWinners by total points:\n")
        max_points = max(total_points.values())
        for player in total_points:
            if total_points[player] == max_points:
                fout.write(f"{str(player)}   -   {str(total_points[player])}\n")

        fout.write("\n\nMax dominant series statistics:\n")
        for player in max_dominant_series:
            fout.write(f"{player}   -   {str(max_dominant_series[player])}\n")

        fout.write("\nWinners by max dominant series:\n")
        max_series = max(max_dominant_series.values())
        for player in max_dominant_series:
            if max_dominant_series[player] == max_series:
                fout.write(f"{str(player)}   -   {str(max_dominant_series[player])}\n")



if __name__ == "__main__":
    main()