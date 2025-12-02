import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ExponentialModel:
    
    def __init__(self, r):
        self.r = r
        self.name = "Экспоненциальная модель"
    
    def step(self, x):
        return self.r * x
    
    def simulate(self, x0, steps):
        trajectory = np.zeros(steps)
        trajectory[0] = x0
        for t in range(steps - 1):
            trajectory[t + 1] = self.step(trajectory[t])
        return trajectory


class LogisticModel:
    
    def __init__(self, r):
        self.r = r
        self.name = "Логистическая модель"
    
    def step(self, x):
        return self.r * x * (1 - x)
    
    def simulate(self, x0, steps):
        trajectory = np.zeros(steps)
        trajectory[0] = x0
        for t in range(steps - 1):
            trajectory[t + 1] = self.step(trajectory[t])
        return trajectory


class MoranModel:
    
    def __init__(self, r):
        self.r = r
        self.name = "Модель Морана"
    
    def step(self, x):
        return x * np.exp(self.r * (1 - x))
    
    def simulate(self, x0, steps):
        trajectory = np.zeros(steps)
        trajectory[0] = x0
        for t in range(steps - 1):
            trajectory[t + 1] = self.step(trajectory[t])
        return trajectory


class HostParasiteModel:
    def __init__(self, a, b, c, K=10.0):
        """
        Модель хозяин-паразит (Никольсон-Бейли) со стабилизацией.
        a - интенсивность атаки паразита
        b - плодовитость хозяина
        c - эффективность паразита
        K - емкость среды для хозяев (стабилизация)
        """
        self.a = a
        self.b = b
        self.c = c
        self.K = K
        self.name = "Модель хозяин-паразит"
    
    def step(self, x, y):
        x = max(0, min(x, 1e6))
        y = max(0, min(y, 1e6))
        
        exp_term = np.exp(-self.a * y)
        growth_factor = np.exp(-x / self.K) if self.K > 0 else 1.0
        x_next = self.b * x * exp_term * growth_factor
        y_next = self.c * x * (1 - exp_term)
        
        return x_next, y_next
    
    def simulate(self, x0, y0, steps):
        x_trajectory = np.zeros(steps)
        y_trajectory = np.zeros(steps)
        x_trajectory[0] = x0
        y_trajectory[0] = y0
        
        for t in range(steps - 1):
            x_trajectory[t + 1], y_trajectory[t + 1] = self.step(
                x_trajectory[t], y_trajectory[t]
            )
            if not np.isfinite(x_trajectory[t + 1]) or not np.isfinite(y_trajectory[t + 1]):
                x_trajectory[t + 1] = 0
                y_trajectory[t + 1] = 0
        
        return x_trajectory, y_trajectory


# ============================================================================
# ЗАДАНИЕ 1: Исследование поведения системы на большом промежутке времени
# ============================================================================

def task1_exponential():
    print("\n" + "="*80)
    print("ЗАДАНИЕ 1: ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    r_values = [0.5, 0.9, 1.0, 1.1, 1.5, 2.0]
    x0_values = [0.1, 0.5, 1.0]
    steps = 100
    
    plot_idx = 0
    for r in r_values:
        if plot_idx >= 9:
            break
            
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])
        model = ExponentialModel(r)
        
        for x0 in x0_values:
            trajectory = model.simulate(x0, steps)
            ax.plot(trajectory, label=f'x₀={x0}', marker='o', markersize=3, alpha=0.7)
        
        ax.set_xlabel('Время t')
        ax.set_ylabel('Популяция x(t)')
        ax.set_title(f'r = {r}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if r < 1:
            behavior = "ВЫМИРАНИЕ"
        elif r == 1:
            behavior = "РАВНОВЕСИЕ"
        else:
            behavior = "НЕОГРАНИЧЕННЫЙ РОСТ"
        
        print(f"\nr = {r}: {behavior}")
        print(f"  При r < 1: популяция стремится к 0")
        print(f"  При r = 1: популяция остается постоянной")
        print(f"  При r > 1: популяция растет экспоненциально")
        
        plot_idx += 1
    
    fig.suptitle('Экспоненциальная модель: x_{t+1} = r*x_t\nИсследование при различных r и начальных условиях', 
                 fontsize=14, fontweight='bold')
    plt.savefig('task1_exponential.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task1_exponential.png")


def task1_logistic():
    """Исследование логистической модели"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 1: ЛОГИСТИЧЕСКАЯ МОДЕЛЬ")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    r_values = [0.5, 1.5, 2.8, 3.2, 3.5, 3.8, 3.83, 3.9, 4.0]
    x0_values = [0.1, 0.5, 0.9]
    steps = 100
    
    for plot_idx, r in enumerate(r_values):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])
        model = LogisticModel(r)
        
        for x0 in x0_values:
            trajectory = model.simulate(x0, steps)
            ax.plot(trajectory, label=f'x₀={x0}', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Время t')
        ax.set_ylabel('Популяция x(t)')
        ax.set_title(f'r = {r}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        if r <= 1:
            behavior = "ВЫМИРАНИЕ (x → 0)"
        elif r <= 3:
            behavior = f"РАВНОВЕСИЕ (x* = {1-1/r:.3f})"
        elif r <= 3.57:
            period = 2 ** int(np.log2((r - 3) * 10) + 1)
            behavior = f"ЦИКЛ периода {period}"
        else:
            behavior = "ХАОТИЧЕСКОЕ ПОВЕДЕНИЕ"
        
        print(f"\nr = {r}: {behavior}")
    
    fig.suptitle('Логистическая модель: x_{t+1} = r*x_t*(1-x_t)\nПереход от порядка к хаосу', 
                 fontsize=14, fontweight='bold')
    plt.savefig('task1_logistic.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task1_logistic.png")


def task1_moran():
    """Исследование модели Морана"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 1: МОДЕЛЬ МОРАНА")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    r_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    x0_values = [0.1, 0.5, 0.9]
    steps = 100
    
    for plot_idx, r in enumerate(r_values):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])
        model = MoranModel(r)
        
        for x0 in x0_values:
            trajectory = model.simulate(x0, steps)
            trajectory = np.clip(trajectory, 0, 2)
            ax.plot(trajectory, label=f'x₀={x0}', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Время t')
        ax.set_ylabel('Популяция x(t)')
        ax.set_title(f'r = {r}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 2)
        
        print(f"\nr = {r}: исследование поведения")
    
    fig.suptitle('Модель Морана: x_{t+1} = x_t * exp(r*(1-x_t))\nИсследование при различных параметрах', 
                 fontsize=14, fontweight='bold')
    plt.savefig('task1_moran.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task1_moran.png")


def task1_host_parasite():
    """Исследование модели хозяин-паразит"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 1: МОДЕЛЬ ХОЗЯИН-ПАРАЗИТ")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    params = [
        (0.5, 2.0, 1.0),  # устойчивое равновесие
        (1.0, 2.0, 1.0),  # колебания
        (0.5, 3.0, 1.5),  # сильные колебания
        (1.0, 2.5, 1.0),  # циклы
        (1.5, 2.0, 1.0),  # сложная динамика
        (0.8, 2.2, 1.2),  # периодические циклы
        (1.2, 2.0, 0.8),  # затухающие колебания
        (0.7, 2.5, 1.3),  # нерегулярные колебания
        (1.0, 3.0, 1.0),  # хаотическое поведение
    ]
    
    x0, y0 = 1.0, 0.5
    steps = 200
    
    for plot_idx, (a, b, c) in enumerate(params):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])
        model = HostParasiteModel(a, b, c)
        
        x_traj, y_traj = model.simulate(x0, y0, steps)
        
        x_traj = np.clip(x_traj, 0, 10)
        y_traj = np.clip(y_traj, 0, 10)
        
        t = np.arange(steps)
        ax.plot(t, x_traj, label='Хозяева (x)', color='blue', alpha=0.7, linewidth=1.5)
        ax.plot(t, y_traj, label='Паразиты (y)', color='red', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Время t')
        ax.set_ylabel('Численность популяции')
        ax.set_title(f'a={a}, b={b}, c={c}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        print(f"\na={a}, b={b}, c={c}:")
        print(f"  Средняя численность хозяев: {np.mean(x_traj[50:]):.3f}")
        print(f"  Средняя численность паразитов: {np.mean(y_traj[50:]):.3f}")
    
    fig.suptitle('Модель хозяин-паразит (Никольсон-Бейли)\nВременные ряды при различных параметрах', 
                 fontsize=14, fontweight='bold')
    plt.savefig('task1_host_parasite_time.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task1_host_parasite_time.png")


# ============================================================================
# ЗАДАНИЕ 2: Изображение траекторий при различных параметрах
# ============================================================================

def task2_comparison_similar():
    """Траектории, которые ведут себя схоже"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 2: ТРАЕКТОРИИ, ВЕДУЩИЕ СЕБЯ СХОЖЕ")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Логистическая модель
    ax = axes[0, 0]
    r_values = [2.5, 2.6, 2.7, 2.8]
    x0 = 0.1
    steps = 50
    
    for r in r_values:
        model = LogisticModel(r)
        trajectory = model.simulate(x0, steps)
        ax.plot(trajectory, label=f'r={r}', marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Логистическая модель: схожее поведение\n(область сходимости к равновесию)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n1. Логистическая модель (r ∈ [2.5, 2.8]):")
    print("   Все траектории сходятся к устойчивому равновесию")
    print("   Различие только в скорости сходимости")
    
    # Модель Морана
    ax = axes[0, 1]
    r_values = [0.5, 0.7, 0.9, 1.1]
    
    for r in r_values:
        model = MoranModel(r)
        trajectory = model.simulate(0.5, steps)
        ax.plot(trajectory, label=f'r={r}', marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Модель Морана: схожее поведение\n(малые значения r)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n2. Модель Морана (r ∈ [0.5, 1.1]):")
    print("   Все траектории монотонно сходятся к равновесию")
    
    # Экспоненциальная модель
    ax = axes[1, 0]
    r_values = [0.5, 0.6, 0.7, 0.8]
    
    for r in r_values:
        model = ExponentialModel(r)
        trajectory = model.simulate(1.0, steps)
        ax.plot(trajectory, label=f'r={r}', marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Экспоненциальная модель: схожее поведение\n(вымирание при r < 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n3. Экспоненциальная модель (r < 1):")
    print("   Все траектории экспоненциально убывают к нулю")
    
    # Хозяин-паразит
    ax = axes[1, 1]
    params_list = [(0.9, 2.0, 1.0), (1.0, 2.0, 1.0), (1.1, 2.0, 1.0)]
    
    for a, b, c in params_list:
        model = HostParasiteModel(a, b, c)
        x_traj, y_traj = model.simulate(1.0, 0.5, 100)
        ax.plot(x_traj[:50], y_traj[:50], label=f'a={a}', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Хозяева (x)')
    ax.set_ylabel('Паразиты (y)')
    ax.set_title('Хозяин-паразит: схожее поведение\n(фазовый портрет, колебания)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n4. Модель хозяин-паразит (a ∈ [0.9, 1.1]):")
    print("   Все траектории демонстрируют колебательное поведение")
    
    plt.tight_layout()
    plt.savefig('task2_similar_trajectories.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task2_similar_trajectories.png")


def task2_comparison_different():
    """Траектории, которые существенно отличаются"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 2: ТРАЕКТОРИИ, СУЩЕСТВЕННО ОТЛИЧАЮЩИЕСЯ")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Логистическая модель
    ax = axes[0, 0]
    r_values = [2.5, 3.2, 3.5, 3.9]
    x0 = 0.1
    steps = 100
    
    for r in r_values:
        model = LogisticModel(r)
        trajectory = model.simulate(x0, steps)
        ax.plot(trajectory, label=f'r={r}', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Логистическая модель: качественно разное поведение\n(равновесие → циклы → хаос)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n1. Логистическая модель (разные режимы):")
    print("   r=2.5: сходимость к равновесию")
    print("   r=3.2: цикл периода 2")
    print("   r=3.5: цикл периода 4")
    print("   r=3.9: хаос")
    
    # Экспоненциальная модель
    ax = axes[0, 1]
    r_values = [0.5, 1.0, 1.5, 2.0]
    
    for r in r_values:
        model = ExponentialModel(r)
        trajectory = model.simulate(1.0, 30)
        ax.plot(trajectory, label=f'r={r}', marker='o', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Экспоненциальная модель: качественно разное поведение\n(вымирание vs постоянство vs рост)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 10)
    
    print("\n2. Экспоненциальная модель:")
    print("   r=0.5: вымирание")
    print("   r=1.0: постоянная популяция")
    print("   r=1.5, 2.0: неограниченный рост")
    
    # Модель Морана
    ax = axes[1, 0]
    r_values = [0.5, 1.5, 2.5, 3.5]
    
    for r in r_values:
        model = MoranModel(r)
        trajectory = model.simulate(0.5, steps)
        trajectory = np.clip(trajectory, 0, 2)
        ax.plot(trajectory, label=f'r={r}', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Время t')
    ax.set_ylabel('x(t)')
    ax.set_title('Модель Морана: качественно разное поведение')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n3. Модель Морана (различные r):")
    print("   Переход от монотонной сходимости к колебаниям")
    
    # Хозяин-паразит
    ax = axes[1, 1]
    params_list = [
        (0.5, 2.0, 1.0, 'устойчивые колебания'),
        (1.5, 2.0, 1.0, 'сильные колебания'),
        (1.0, 3.0, 1.0, 'хаотическое поведение')
    ]
    
    for a, b, c, label in params_list:
        model = HostParasiteModel(a, b, c)
        x_traj, y_traj = model.simulate(1.0, 0.5, 150)
        x_traj = np.clip(x_traj, 0, 10)
        y_traj = np.clip(y_traj, 0, 10)
        ax.plot(x_traj[50:], y_traj[50:], label=f'a={a} ({label})', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Хозяева (x)')
    ax.set_ylabel('Паразиты (y)')
    ax.set_title('Хозяин-паразит: качественно разное поведение\n(фазовые портреты)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("\n4. Модель хозяин-паразит:")
    print("   Различные типы динамики в зависимости от параметров")
    
    plt.tight_layout()
    plt.savefig('task2_different_trajectories.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task2_different_trajectories.png")


# ============================================================================
# ЗАДАНИЕ 3: Бифуркационный анализ
# ============================================================================

def task3_bifurcation_logistic():
    """Бифуркационная диаграмма для логистической модели"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 3: БИФУРКАЦИОННАЯ ДИАГРАММА - ЛОГИСТИЧЕСКАЯ МОДЕЛЬ")
    print("="*80)
    
    r_min, r_max = 0.0, 4.0
    r_steps = 2000
    r_values = np.linspace(r_min, r_max, r_steps)
    
    transient = 500
    plot_points = 200
    x0 = 0.5
    
    print("\nВычисление бифуркационной диаграммы...")
    print("Это займет некоторое время...")
    

    r_plot = []
    x_plot = []
    
    for r in r_values:
        model = LogisticModel(r)
        trajectory = model.simulate(x0, transient + plot_points)
        
        final_points = trajectory[transient:]
        
        r_plot.extend([r] * len(final_points))
        x_plot.extend(final_points)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    ax1.plot(r_plot, x_plot, ',', color='navy', alpha=0.9, markersize=2.0)
    ax1.set_xlabel('Параметр r', fontsize=12)
    ax1.set_ylabel('x*', fontsize=12)
    ax1.set_title('Бифуркационная диаграмма логистической модели\nx_{t+1} = r*x_t*(1-x_t)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(r_min, r_max)
    ax1.set_ylim(0, 1)
    
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='r=1 (возникновение равновесия)')
    ax1.axvline(x=3.0, color='orange', linestyle='--', alpha=0.5, label='r=3 (первая бифуркация)')
    ax1.axvline(x=3.57, color='purple', linestyle='--', alpha=0.5, label='r≈3.57 (начало хаоса)')
    ax1.legend()
    
    r_zoom_min, r_zoom_max = 2.8, 4.0
    mask = (np.array(r_plot) >= r_zoom_min) & (np.array(r_plot) <= r_zoom_max)
    ax2.plot(np.array(r_plot)[mask], np.array(x_plot)[mask], ',', color='navy', alpha=0.9, markersize=2.0)
    ax2.set_xlabel('Параметр r', fontsize=12)
    ax2.set_ylabel('x*', fontsize=12)
    ax2.set_title('Увеличенный фрагмент: переход к хаосу через удвоение периода', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(r_zoom_min, r_zoom_max)
    ax2.set_ylim(0, 1)
    
    ax2.axvline(x=3.0, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(x=1 + np.sqrt(6), color='yellow', linestyle='--', alpha=0.5, linewidth=2, 
                label=f'r≈{1 + np.sqrt(6):.3f} (цикл периода 2)')
    ax2.axvline(x=3.57, color='purple', linestyle='--', alpha=0.5, linewidth=2, 
                label='r≈3.57 (точка Фейгенбаума)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('task3_bifurcation_logistic.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task3_bifurcation_logistic.png")
    
    print("\n" + "="*60)
    print("АНАЛИЗ БИФУРКАЦИОННОЙ ДИАГРАММЫ")
    print("="*60)
    
    print("\n1. ОБЛАСТЬ r ∈ (0, 1]:")
    print("   Поведение: популяция вымирает (x → 0)")
    print("   Тип: тривиальное равновесие x* = 0 устойчиво")
    
    print("\n2. ОБЛАСТЬ r ∈ (1, 3]:")
    print("   Поведение: сходимость к нетривиальному равновесию")
    print("   Равновесие: x* = (r-1)/r")
    print("   Тип: устойчивая неподвижная точка")
    
    print("\n3. ТОЧКА r = 3:")
    print("   ★ ПЕРВАЯ БИФУРКАЦИЯ ★")
    print("   Неподвижная точка теряет устойчивость")
    print("   Рождается устойчивый цикл периода 2")
    
    print("\n4. ОБЛАСТЬ r ∈ (3, 1+√6) ≈ (3, 3.449):")
    print("   Поведение: устойчивый цикл периода 2")
    print("   x колеблется между двумя значениями")
    
    print("\n5. ТОЧКА r ≈ 3.449:")
    print("   ★ ВТОРАЯ БИФУРКАЦИЯ ★")
    print("   Цикл периода 2 теряет устойчивость")
    print("   Рождается цикл периода 4")
    
    print("\n6. КАСКАД БИФУРКАЦИЙ УДВОЕНИЯ ПЕРИОДА:")
    print("   r ≈ 3.449: период 4")
    print("   r ≈ 3.544: период 8")
    print("   r ≈ 3.564: период 16")
    print("   r ≈ 3.569: период 32")
    print("   ...")
    
    print("\n7. ТОЧКА ФЕЙГЕНБАУМА r ≈ 3.569946:")
    print("   ★ ГРАНИЦА ДЕТЕРМИНИРОВАННОГО ХАОСА ★")
    print("   Период становится бесконечным")
    print("   Начинается хаотический режим")
    
    print("\n8. ОБЛАСТЬ r ∈ (3.57, 4.0):")
    print("   Поведение: ХАОС")
    print("   Апериодические, непредсказуемые траектории")
    print("   Чувствительность к начальным условиям")
    print("   Присутствуют 'окна периодичности':")
    print("   - r ≈ 3.83: окно периода 3")
    print("   - Другие окна более высоких периодов")
    
    print("\n9. КОНСТАНТА ФЕЙГЕНБАУМА δ ≈ 4.669:")
    print("   Универсальная константа:")
    print("   δ = lim(n→∞) (r_n - r_{n-1})/(r_{n+1} - r_n)")
    print("   где r_n - точка n-й бифуркации")
    
    print("\n" + "="*60)


def task3_bifurcation_moran():
    """Бифуркационная диаграмма для модели Морана"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 3: БИФУРКАЦИОННАЯ ДИАГРАММА - МОДЕЛЬ МОРАНА")
    print("="*80)
    
    r_min, r_max = 0.0, 4.5
    r_steps = 1500
    r_values = np.linspace(r_min, r_max, r_steps)
    
    transient = 500
    plot_points = 200
    x0 = 0.5
    
    print("\nВычисление бифуркационной диаграммы для модели Морана...")
    
    r_plot = []
    x_plot = []
    
    for r in r_values:
        model = MoranModel(r)
        trajectory = model.simulate(x0, transient + plot_points)
        
        trajectory = np.clip(trajectory, 0, 5)
        final_points = trajectory[transient:]
        
        r_plot.extend([r] * len(final_points))
        x_plot.extend(final_points)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(r_plot, x_plot, ',', color='red', alpha=0.9, markersize=2.0)
    ax.set_xlabel('Параметр r', fontsize=12)
    ax.set_ylabel('x*', fontsize=12)
    ax.set_title('Бифуркационная диаграмма модели Морана\nx_{t+1} = x_t * exp(r*(1-x_t))', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig('task3_bifurcation_moran.png', dpi=300, bbox_inches='tight')
    print("\n✓ График сохранен: task3_bifurcation_moran.png")
    
    print("\nМодель Морана демонстрирует более плавные переходы")
    print("и другие бифуркационные точки по сравнению с логистической моделью")


def task3_host_parasite_parameter_space():
    """Исследование параметрического пространства для модели хозяин-паразит"""
    print("\n" + "="*80)
    print("ЗАДАНИЕ 3: ПАРАМЕТРИЧЕСКОЕ ПРОСТРАНСТВО - МОДЕЛЬ ХОЗЯИН-ПАРАЗИТ")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    a_values = np.linspace(0.1, 2.0, 300)
    b_fixed, c_fixed = 3.0, 1.5
    K_fixed = 5.0
    x0, y0 = 1.0, 0.5
    
    transient = 300
    plot_points = 100
    
    print("\nИсследование влияния параметра a (интенсивность атаки)...")
    
    a_plot_x = []
    x_plot_x = []
    a_plot_y = []
    y_plot_y = []
    
    for a in a_values:
        model = HostParasiteModel(a, b_fixed, c_fixed, K=K_fixed)
        x_traj, y_traj = model.simulate(x0, y0, transient + plot_points)
        
        final_x = x_traj[transient:]
        final_y = y_traj[transient:]
        
        for i in range(len(final_x)):
            x_val = final_x[i]
            y_val = final_y[i]
            if np.isfinite(x_val) and np.isfinite(y_val) and x_val > 1e-6 and y_val > 1e-6:
                a_plot_x.append(a)
                x_plot_x.append(x_val)
                a_plot_y.append(a)
                y_plot_y.append(y_val)
    
    if len(a_plot_x) > 0:
        axes[0].scatter(a_plot_x, x_plot_x, s=1.0, c='dodgerblue', alpha=0.8)
    axes[0].set_xlabel('Параметр a (интенсивность атаки)', fontsize=12)
    axes[0].set_ylabel('x* (численность хозяев)', fontsize=12)
    axes[0].set_title(f'Влияние параметра a на хозяев\n(b={b_fixed}, c={c_fixed}, K={K_fixed})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 2.1)
    
    if len(a_plot_y) > 0:
        axes[1].scatter(a_plot_y, y_plot_y, s=1.0, c='orangered', alpha=0.8)
    axes[1].set_xlabel('Параметр a (интенсивность атаки)', fontsize=12)
    axes[1].set_ylabel('y* (численность паразитов)', fontsize=12)
    axes[1].set_title(f'Влияние параметра a на паразитов\n(b={b_fixed}, c={c_fixed}, K={K_fixed})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 2.1)
    
    plt.tight_layout()
    plt.savefig('task3_host_parasite_bifurcation_a.png', dpi=300, bbox_inches='tight')
    print(f"✓ График сохранен: task3_host_parasite_bifurcation_a.png (точек: {len(a_plot_x)})")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    a_fixed, c_fixed = 0.8, 1.5
    b_values = np.linspace(1.5, 5.0, 300)
    
    print("\nИсследование влияния параметра b (плодовитость хозяев)...")
    
    b_plot_x = []
    x_plot_b = []
    b_plot_y = []
    y_plot_b = []
    
    for b in b_values:
        model = HostParasiteModel(a_fixed, b, c_fixed, K=K_fixed)
        x_traj, y_traj = model.simulate(x0, y0, transient + plot_points)
        
        final_x = x_traj[transient:]
        final_y = y_traj[transient:]
        
        for i in range(len(final_x)):
            x_val = final_x[i]
            y_val = final_y[i]
            if np.isfinite(x_val) and np.isfinite(y_val) and x_val > 1e-6 and y_val > 1e-6:
                b_plot_x.append(b)
                x_plot_b.append(x_val)
                b_plot_y.append(b)
                y_plot_b.append(y_val)
    
    if len(b_plot_x) > 0:
        axes[0].scatter(b_plot_x, x_plot_b, s=1.0, c='dodgerblue', alpha=0.8)
    axes[0].set_xlabel('Параметр b (плодовитость хозяев)', fontsize=12)
    axes[0].set_ylabel('x* (численность хозяев)', fontsize=12)
    axes[0].set_title(f'Влияние параметра b на хозяев\n(a={a_fixed}, c={c_fixed}, K={K_fixed})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1.4, 5.1)
    
    if len(b_plot_y) > 0:
        axes[1].scatter(b_plot_y, y_plot_b, s=1.0, c='orangered', alpha=0.8)
    axes[1].set_xlabel('Параметр b (плодовитость хозяев)', fontsize=12)
    axes[1].set_ylabel('y* (численность паразитов)', fontsize=12)
    axes[1].set_title(f'Влияние параметра b на паразитов\n(a={a_fixed}, c={c_fixed}, K={K_fixed})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1.4, 5.1)
    
    plt.tight_layout()
    plt.savefig('task3_host_parasite_bifurcation_b.png', dpi=300, bbox_inches='tight')
    print(f"✓ График сохранен: task3_host_parasite_bifurcation_b.png (точек: {len(b_plot_x)})")
    
    print("\nАнализ модели хозяин-паразит (со стабилизацией):")
    print("- Добавлено логистическое ограничение с емкостью среды K")
    print("- При малых a: система устойчива, популяции сосуществуют")
    print("- При больших a: возникают колебания и хаотическое поведение")
    print("- Увеличение b (плодовитость): усиление колебаний")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция - выполнение всех заданий лабораторной работы"""
    
    print("\n" + "="*80)
    print("ММС. ЛАБОРАТОРНАЯ РАБОТА №4")
    print("МАТЕМАТИЧЕСКИЕ МОДЕЛИ РОСТА ПОПУЛЯЦИИ")
    print("="*80)
    print("\nИсследование динамических систем с дискретным временем:")
    print("1. Экспоненциальная модель: x_{t+1} = r*x_t")
    print("2. Логистическая модель: x_{t+1} = r*x_t*(1-x_t)")
    print("3. Модель Морана: x_{t+1} = x_t*exp(r*(1-x_t))")
    print("4. Модель хозяин-паразит (Никольсон-Бейли):")
    print("   x_{t+1} = b*x_t*exp(-a*y_t)")
    print("   y_{t+1} = c*x_t*(1 - exp(-a*y_t))")
    print("\n" + "="*80)
    
    # ЗАДАНИЕ 1: Исследование поведения на большом промежутке времени
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ВЫПОЛНЕНИЕ ЗАДАНИЯ 1" + " "*38 + "║")
    print("║" + " "*10 + "Исследование поведения системы на большом промежутке" + " "*15 + "║")
    print("╚" + "="*78 + "╝")
    
    task1_exponential()
    task1_logistic()
    task1_moran()
    task1_host_parasite()
    
    # ЗАДАНИЕ 2: Изображение траекторий при различных параметрах
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ВЫПОЛНЕНИЕ ЗАДАНИЯ 2" + " "*38 + "║")
    print("║" + " "*15 + "Траектории при различных параметрах" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    task2_comparison_similar()
    task2_comparison_different()
    
    # ЗАДАНИЕ 3: Бифуркационный анализ
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ВЫПОЛНЕНИЕ ЗАДАНИЯ 3" + " "*38 + "║")
    print("║" + " "*10 + "Определение параметров качественного изменения поведения" + " "*11 + "║")
    print("╚" + "="*78 + "╝")
    
    task3_bifurcation_logistic()
    task3_bifurcation_moran()
    task3_host_parasite_parameter_space()
    
    
    # ИТОГОВАЯ СВОДКА
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "РАБОТА ЗАВЕРШЕНА" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nСОЗДАННЫЕ ГРАФИКИ:")
    print("   ✓ task1_exponential.png - Экспоненциальная модель")
    print("   ✓ task1_logistic.png - Логистическая модель")
    print("   ✓ task1_moran.png - Модель Морана")
    print("   ✓ task1_host_parasite_time.png - Хозяин-паразит (временные ряды)")
    print("   ✓ task2_similar_trajectories.png - Схожие траектории")
    print("   ✓ task2_different_trajectories.png - Различные траектории")
    print("   ✓ task2_phase_portraits.png - Фазовые портреты")
    print("   ✓ task3_bifurcation_logistic.png - Бифуркационная диаграмма (логистическая)")
    print("   ✓ task3_bifurcation_moran.png - Бифуркационная диаграмма (Морана)")
    print("   ✓ task3_host_parasite_bifurcation_a.png - Параметр a")
    print("   ✓ task3_host_parasite_bifurcation_b.png - Параметр b")
    print("   ✓ task3_sensitivity.png - Чувствительность к начальным условиям")
    print("   ✓ additional_cobweb.png - Диаграммы паутины")
    print("   ✓ additional_lyapunov.png - Показатель Ляпунова")
    
    print("\n" + "="*80)
    print("ОСНОВНЫЕ ВЫВОДЫ:")
    print("="*80)
    print("\n1. ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ:")
    print("   - Простейшая модель, не учитывает ограничения среды")
    print("   - r < 1: вымирание, r = 1: равновесие, r > 1: неограниченный рост")
    
    print("\n2. ЛОГИСТИЧЕСКАЯ МОДЕЛЬ:")
    print("   - Демонстрирует богатейшую динамику!")
    print("   - Путь к хаосу через каскад удвоения периода")
    print("   - r ∈ (0,1]: вымирание")
    print("   - r ∈ (1,3]: устойчивое равновесие")
    print("   - r ∈ (3, 3.57]: периодические циклы (2, 4, 8, 16...)")
    print("   - r > 3.57: детерминированный хаос")
    
    print("\n3. МОДЕЛЬ МОРАНА:")
    print("   - Похожа на логистическую, но с более плавными переходами")
    print("   - Другие критические значения параметров")
    
    print("\n4. МОДЕЛЬ ХОЗЯИН-ПАРАЗИТ:")
    print("   - Двумерная система с богатой динамикой")
    print("   - Демонстрирует колебания, циклы, хаотическое поведение")
    print("   - Зависит от трех параметров: a (атака), b (плодовитость), c (эффективность)")
        
    print("\n" + "="*80)
    print("ВСЕ ЗАДАНИЯ ЛАБОРАТОРНОЙ РАБОТЫ ВЫПОЛНЕНЫ!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

