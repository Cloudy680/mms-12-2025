import random
import matplotlib.pyplot as plt

def coin_exp(tosses = 100, p = 0.5):
    res = []
    for i in range(tosses):
        if random.random() < p:
            res.append("О")
        else:
            res.append("Р")
    return res

def print_menu():
    print("Какой эксперимент вы хотите выполнить?")
    print("1. Какое число орлов выпадает в среднем")
    print("2. С какой вероятностью можно получить число орлов больше 60")
    print("3. Оценить вероятность выпадения числа орлов, принадлежащих интервалам [0, 10), [10, 20), . . . , [80, 90), [90, 100)")
    print("4. Внутри какого интервала с вероятность примерно 0.95 стоит ожидать значение числа орлов")
    print("5. С какой вероятностью найдется хотя бы одна серия из 5 орлов подряд")
    print("6. Посмотреть графики зависимости от вероятности p")
    print("\n7. Поменять число проводимых экспериментов")
    print("Введдите ноль/0, чтобы завершить работу программы!\n")
    print("Введите номер эксперимента, чтобы его выполнить:")
    answer = input()
    return answer

def first_exp(ress, exp_count):
    print(f"\nСреднее количество орлов при количестве экспериментов {exp_count}: {sum(ress) / len(ress)}")

def second_exp(ress, exp_count):
    count_60 = 0
    for item in ress:
        if item > 60:
            count_60 += 1
    print(f"\nС такой вероятностью можно получить число орлов больше 60 при количестве экспериментов {exp_count}: {count_60 / len(ress)}")

def third_exp(ress, exp_count):
    intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    ans = {}
    for interval in intervals:
        count_in_int = 0
        for item in ress:
            if interval[0] <= item < interval[1]:
                count_in_int += 1
        ans[interval] = count_in_int / len(ress)
    print(f"\nС такой вероятностью выпадет число орлов соответствующее этим интервалам при количестве экспериментов {exp_count}")
    for key in ans:
        print(f"{key}: {ans[key]:.6f}")

def fourth_exp(ress, exp_count, p = 0.5):
    left_ran, right_ran = p * 100 + 1, p * 100
    probabil = 0.5
    while probabil < 0.935:
        left_ran -= 1
        right_ran += 1
        counter = 0
        for item in ress:
            if left_ran <= item <= right_ran:
                counter += 1
        probabil = counter / len(ress)
    print(f"Внутри этого интервала с вероятностью {probabil} стоит ожидать значение числа орлов при количестве экспериментов {exp_count} и p = {p}:")
    print(f"[{left_ran}, {right_ran})")
    return (left_ran, right_ran)

def sixth_exp():
    def max_len_seri(p):
        max_len = 0
        for exp in range(100000):
            res = coin_exp(p = p)
            current_len = 0
            for item in res:
                if item == "О":
                    current_len += 1
                else:
                    if current_len > max_len:
                        max_len = current_len
                    current_len = 0
            if current_len > max_len:
                max_len = current_len
        return max_len
    print("1. Ожидаемое число орлов от p")
    print("2. Ширина предсказательного интервала от p")
    print("3. Вероятность наличия серии из 5 орлов")
    print("4. Длина максимально вероятной серии от p")
    print("\nНажмите Space, чтобы вернуться\n")
    answer = input("Введите номер для графика: ")
    while answer != " ":
        if answer == "1":
            x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            y = []
            for item in x:
                result, trash = new_exps(p=item)
                y.append(sum(result) / len(result))
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, 'bo-', linewidth=2, markersize=6)
            plt.title('Зависимость ожидаемого числа орлов от вероятности p', fontsize=14)
            plt.xlabel('Вероятность выпадения орла (p)', fontsize=12)
            plt.ylabel('Среднее число орлов в 100 бросках', fontsize=12)
            plt.grid(True, alpha=0.4, linestyle='-')
            plt.show()
        elif answer == "2":
            x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            y = []
            for item in x:
                result, trash = new_exps(p=item)
                an = fourth_exp(result, exp_count=len(result), p=item)
                y.append(an[1] - an[0])
            plt.figure(figsize=(12, 8))
            plt.bar(x, y, width=0.08, color='skyblue', edgecolor='black', alpha=0.8)
            plt.title('Ширина 95% доверительного интервала для числа орлов\nв зависимости от вероятности p',
                      fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Вероятность выпадения орла (p)', fontsize=14, fontweight='bold')
            plt.ylabel('Ширина интервала', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.xticks(x, [f'{val:.1f}' for val in x], fontsize=11)
            plt.yticks(fontsize=11)

            plt.xlim(-0.1, 1.1)
            plt.ylim(0, max(y) * 1.15)

            plt.tight_layout()
            plt.show()
        elif answer == "3":
            x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            y = []
            for item in x:
                result, suc = new_exps(p = item)
                y.append(suc)
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, 'bo-', linewidth=2, markersize=6)
            plt.title("Вероятность выпадения хотя бы 1 серии из 5 орлов\nв зависимости от вероятности p",
                      fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Вероятность выпадения орла (p)', fontsize=12)
            plt.ylabel("Вероятность выпадения хотя бы одной серии", fontsize=12)
            plt.grid(True, alpha=0.4, linestyle='-')
            plt.show()
        elif answer == "4":
            x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            y = []
            for item in x:
                y.append(max_len_seri(p = item))
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, 'bo-', linewidth=2, markersize=6)
            plt.title("Максимальная вероятная серия орлов\nв зависимости от вероятности p",
                      fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Вероятность выпадения орла (p)', fontsize=12)
            plt.ylabel("Максимально вероятная серия", fontsize=12)
            plt.grid(True, alpha=0.4, linestyle='-')
            plt.show()
        print("1. Ожидаемое число орлов от p")
        print("2. Ширина предсказательного интервала от p")
        print("3. Вероятность наличия серии из 5 орлов")
        print("4. Длина максимально вероятной серии от p")
        print("\nНажмите Space, чтобы вернуться\n")
        answer = input("Введите номер для графика: ")



def new_exps(exp_count = 10000, p = 0.5):
    result = []
    sucesses = 0
    for i in range(exp_count):
        has_series, current_serie = False, 0
        mini_res, eagles_count = coin_exp(p = p), 0
        for item in mini_res:
            if item == "О":
                current_serie += 1
                eagles_count += 1
            else:
                current_serie = 0
            if not has_series and current_serie >= 5:
                has_series = True
        result.append(eagles_count)
        if has_series:
            sucesses += 1
    return result, sucesses / exp_count


def main():
    print("Введите начальное количество экспериментов, которое вы хотите делать с монеткой (100, 1000, ...):")
    exp_count = int(input())
    result, suc = new_exps(exp_count)

    answer = print_menu()
    while answer != "0" or answer != "ноль":
        if answer == "1":
            first_exp(ress = result, exp_count = exp_count)
        elif answer == "2":
            second_exp(ress = result, exp_count = exp_count)
        elif answer == "3":
            third_exp(ress = result, exp_count = exp_count)
        elif answer == "4":
            fourth_exp(ress = result, exp_count = exp_count)
        elif answer == "5":
            print(f"С вероятностью {suc} найдется хотя бы одна серия из 5 орлов при количестве экспериментов {exp_count}")
        elif answer == "6":
            sixth_exp()
        elif answer == "7":
            exp_count = int(input("Введите новое число экспериментов: "))
        if answer != "6" and answer != "7":
            answer = input("\nЖелаете продолжить? (y/n): ")
            if answer != "y":
                break

        answer = print_menu()
        result, suc = new_exps(exp_count)


if __name__ == "__main__":
    main()

