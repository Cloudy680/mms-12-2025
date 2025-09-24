import random
import matplotlib.pyplot as plt

random.seed(42)

TEST_AMOUNT = 1000
FLIP_AMOUNT = 100
PROBABILITY_OF_HEADS_IN_INTERVAL = 0.95
SERIES_LENGTH = 5
PROBABILITY_OF_HEADS_DELTA = 0.05


def do_all_tests(p):
    options = [0, 1]
    probabilities = [1-p, p]
    total = 0
    heads_more_than_60 = 0
    heads_sum = []
    seria_tests = 0.0
    max_seria = 0
    for i in range(TEST_AMOUNT):
        heads = 0
        seria_count = 0
        seria_here = False
        for j in range(FLIP_AMOUNT):
            coin = random.choices(options, weights=probabilities, k=1)[0]
            heads += coin
            if coin == 1:
                seria_count += 1
            elif coin == 0:
                seria_count = 0
            if seria_count == 5:
                seria_here = True
        if seria_here == True:
            seria_tests += 1.0
            if max_seria < seria_count:
                max_seria = seria_count
        total += heads
        heads_sum.append(heads)
        if heads > 60:
            heads_more_than_60 += 1
    return heads_sum, total, heads_more_than_60, seria_tests, max_seria

def count_chances_in_intervals(heads_sum):
    intervals = {(0, 10) : 0.0, (10, 20) : 0.0, (20, 30) : 0.0, (30, 40): 0.0, (40, 50):0.0, (50, 60):0.0, (60, 70):0.0, (70, 80):0.0, (80, 90):0.0, (90, 100):0.0}
    for interval in intervals:
        for h in heads_sum:
            if h >= interval[0] and h < interval[1]:
                intervals[interval] += 1.0
        intervals[interval] = intervals[interval]/TEST_AMOUNT
    return intervals

def find_interval_with_prob_heads (heads_sum, p):
    def_interval = [int(p * 100), int(p * 100)]
    while True:
        if def_interval[0] < 0 or def_interval[1] > 100:
            break
        am = 0.0
        for i in range(len(heads_sum)):
            if heads_sum[i] >= def_interval[0] and heads_sum[i] <= def_interval[1]:
                am += 1.0
        if am / TEST_AMOUNT >= 0.95:
            break
        else:
            def_interval[0] -= 1
            def_interval[1] += 1
    return def_interval


def if_heads_prob_eq_p():
    x = [0, 0.2, 0.5, 1]
    y = [0, 20, 50, 100]
    plt.plot(x, y,
            linewidth=4,
            color='#ffd700',
            alpha=1)
    plt.title("Expected number of heads(probability of heads = p)")
    plt.xlabel("Probability of getting heads")
    plt.ylabel(f"Heads amount in {TEST_AMOUNT} tests")
    plt.grid(True, alpha=0.4)
    plt.show()

def heads_prediction_interval_width(p):
    options = [0, 1]
    probabilities = [1-p, p]
    total = 0
    heads_sum = []
    for test in range(TEST_AMOUNT):
        heads = 0
        for flip in range(FLIP_AMOUNT):
            coin = random.choices(options, weights=probabilities, k=1)[0]
            heads += coin
        total += heads
        heads_sum.append(heads)
    interval = find_interval_with_prob_heads(heads_sum, p)
    return interval[1] - interval[0]


def length_of_interval():
    intervals = []
    probabilities = []
    heads_probability = 0.0
    while heads_probability < 1:
        probabilities.append(heads_probability)
        intervals.append(heads_prediction_interval_width(heads_probability))
        heads_probability += PROBABILITY_OF_HEADS_DELTA
    plt.plot(probabilities, intervals,
             linewidth=4,
             color='#ffd700',
             alpha=1)
    plt.title("Width of the prediction interval(probability of heads = p)")
    plt.xlabel("Probability of heads")
    plt.ylabel(f"Length of interval")
    plt.grid(True, alpha=0.4)
    plt.show()

def series_probabilities():
    heads_probability = 0.0
    seria_tests_arr = []
    probabilities = []
    while heads_probability < 1:
        probabilities.append(heads_probability)
        heads_sum, total, heads_more_than_60, seria_tests, max_seria = do_all_tests(heads_probability)
        heads_probability += PROBABILITY_OF_HEADS_DELTA
        seria_tests_arr.append(seria_tests / TEST_AMOUNT)
    plt.plot(probabilities, seria_tests_arr,
             linewidth=4,
             color='#ffd700',
             alpha=1)
    plt.title("Probabilities of having a series of 5 heads(probability of heads = p)")
    plt.xlabel("Probability of heads")
    plt.ylabel(f"Series probabilities(length {SERIES_LENGTH})")
    plt.grid(True, alpha=0.4)
    plt.show()

def max_series():
    heads_probability = 0.0
    max_seria_tests_arr = []
    probabilities = []
    while heads_probability < 1:
        probabilities.append(heads_probability)
        heads_sum, total, heads_more_than_60, seria_tests, max_seria = do_all_tests(heads_probability)
        heads_probability += PROBABILITY_OF_HEADS_DELTA
        max_seria_tests_arr.append(max_seria)
    plt.plot(probabilities, max_seria_tests_arr,
             linewidth=4,
             color='#ffd700',
             alpha=1)
    plt.title("Length of the maximum series(probability of heads = p)")
    plt.xlabel("Probability of heads")
    plt.ylabel(f"Max series")
    plt.grid(True, alpha = 0.4)
    plt.show()


def main():

    heads_sum, total, heads_more_than_60, seria_tests, max_seria = do_all_tests(0.5)

    print("TASK 1:")
    print(f"Average heads amount ({TEST_AMOUNT} tests): {total / TEST_AMOUNT}")
    print("**********************************************************************************")


    print("TASK 2:")
    print(f"Probability of heads in amount of 60 : {(heads_more_than_60 / TEST_AMOUNT):.2}")
    print("**********************************************************************************")


    print("TASK 3:")
    intervals = count_chances_in_intervals(heads_sum)
    print(f"Probability of heads in intervals:")
    for interval in intervals:
        if interval[0] == 90:
            print(f"[{interval[0]}, {interval[1]}]", end="")
        else:
            print(f"[{interval[0]}, {interval[1]})", end = "")
        print(f" : {intervals[interval]}")
    print("**********************************************************************************")


    print("TASK 4:")
    interval_with_probability = find_interval_with_prob_heads(heads_sum, 0.5)
    print(f"Interval with probability of heads of {PROBABILITY_OF_HEADS_IN_INTERVAL} : [{interval_with_probability[0]}, {interval_with_probability[1]}]")
    print("**********************************************************************************")


    print(f"TASK 5:")
    print(f"Probability of seria {SERIES_LENGTH} : {seria_tests / TEST_AMOUNT}")
    print("**********************************************************************************")

    print("TASK 6:")
    choice = None
    while choice != 0:
        print("1) Expected number of heads(probability of heads = p)")
        print("2) Width of the prediction interval(probability of heads = p)")
        print("3) Probabilities of having a series of 5 heads(probability of heads = p)")
        print("4) Length of the maximum series(probability of heads = p)")
        print("0) Quit")
        choice = int(input("Enter the number of chart you want to see: "))
        if choice == 1:
            if_heads_prob_eq_p()
        elif choice == 2:
            length_of_interval()
        elif choice == 3:
            series_probabilities()
        elif choice == 4:
            max_series()

if __name__ == "__main__":
    main()