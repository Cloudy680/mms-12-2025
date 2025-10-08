from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict


# Payoff matrix W as described in labs/2.md
# Rows: my move (0 = cooperate, 1 = defect)
# Cols: opponent move (0 = cooperate, 1 = defect)
W: List[List[int]] = [
    [3, 0],  # I play 0
    [5, 1],  # I play 1
]


class Strategy:
    """Base class for a deterministic strategy."""

    name: str

    def reset(self) -> None:
        """Reset any internal state between matches."""
        return

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        """Return the next move: 0 (cooperate) or 1 (defect).

        round_index is 1-based.
        """
        raise NotImplementedError


class Alex(Strategy):
    name = "Alex"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 1  # always defect


class Bob(Strategy):
    name = "Bob"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 0  # always cooperate


class Clara(Strategy):
    name = "Clara"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            return 0  # first move cooperate
        return opponent_history[-1]  # tit-for-tat: mirror opponent's last


class Denis(Strategy):
    name = "Denis"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            return 0  # first move cooperate
        return 1 - opponent_history[-1]  # opposite of opponent's last


class Emma(Strategy):
    name = "Emma"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        # cooperate except every 20th round
        return 1 if (round_index % 20 == 0) else 0


class Frida(Strategy):
    name = "Frida"

    def __init__(self) -> None:
        self._opponent_defected_ever = False

    def reset(self) -> None:
        self._opponent_defected_ever = False

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if opponent_history and opponent_history[-1] == 1:
            self._opponent_defected_ever = True
        # Grim trigger: cooperate until opponent defects once, then defect forever
        return 1 if self._opponent_defected_ever else 0


class George(Strategy):
    name = "George"

    def __init__(self) -> None:
        self._last_move: int | None = None

    def reset(self) -> None:
        self._last_move = None

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        # Win-Stay, Lose-Shift (Pavlov):
        # - First round cooperate
        # - If last round payoff >= 3, repeat the same move; else switch
        if not my_history:
            self._last_move = 0
            return 0

        last_my = my_history[-1]
        last_op = opponent_history[-1]
        last_payoff = W[last_my][last_op]
        if last_payoff >= 3:
            # keep last decision
            assert self._last_move is not None
            return self._last_move
        # switch decision
        new_move = 1 - last_my
        self._last_move = new_move
        return new_move


@dataclass
class MatchResult:
    history_a: List[int]
    history_b: List[int]
    score_a: int
    score_b: int
    max_dominating_run_a: int
    max_dominating_run_b: int


def play_match(strategy_a: Strategy, strategy_b: Strategy, rounds: int = 200) -> MatchResult:
    strategy_a.reset()
    strategy_b.reset()

    history_a: List[int] = []
    history_b: List[int] = []
    score_a = 0
    score_b = 0

    current_dom_run_a = 0
    current_dom_run_b = 0
    max_dom_run_a = 0
    max_dom_run_b = 0

    for r in range(1, rounds + 1):
        move_a = strategy_a.choose(history_a, history_b, r)
        move_b = strategy_b.choose(history_b, history_a, r)  # pass symmetric histories

        history_a.append(move_a)
        history_b.append(move_b)

        payoff_a = W[move_a][move_b]
        payoff_b = W[move_b][move_a]
        score_a += payoff_a
        score_b += payoff_b

        # Update dominating run counters
        if payoff_a == 5 and payoff_b == 0:
            current_dom_run_a += 1
            current_dom_run_b = 0
        elif payoff_b == 5 and payoff_a == 0:
            current_dom_run_b += 1
            current_dom_run_a = 0
        else:
            current_dom_run_a = 0
            current_dom_run_b = 0

        if current_dom_run_a > max_dom_run_a:
            max_dom_run_a = current_dom_run_a
        if current_dom_run_b > max_dom_run_b:
            max_dom_run_b = current_dom_run_b

    return MatchResult(
        history_a=history_a,
        history_b=history_b,
        score_a=score_a,
        score_b=score_b,
        max_dominating_run_a=max_dom_run_a,
        max_dominating_run_b=max_dom_run_b,
    )


def run_tournament() -> Tuple[Dict[str, int], Dict[str, int]]:
    strategies: List[Strategy] = [
        Alex(),
        Bob(),
        Clara(),
        Denis(),
        Emma(),
        Frida(),
        George(),
    ]

    total_scores: Dict[str, int] = {s.name: 0 for s in strategies}
    max_dominating_runs: Dict[str, int] = {s.name: 0 for s in strategies}

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a = strategies[i]
            b = strategies[j]

            # Use fresh instances per match to avoid shared state across matches
            a_instance = type(a)()
            b_instance = type(b)()

            result = play_match(a_instance, b_instance)

            total_scores[a.name] += result.score_a
            total_scores[b.name] += result.score_b

            if result.max_dominating_run_a > max_dominating_runs[a.name]:
                max_dominating_runs[a.name] = result.max_dominating_run_a
            if result.max_dominating_run_b > max_dominating_runs[b.name]:
                max_dominating_runs[b.name] = result.max_dominating_run_b

    return total_scores, max_dominating_runs


def save_results_markdown(path: str, total_scores: Dict[str, int], max_dom_runs: Dict[str, int]) -> None:
    lines: List[str] = []
    lines.append("# Результаты турнира: Лабораторная №2")
    lines.append("")
    lines.append("## Суммарные очки (200 раундов, круговой турнир)")
    lines.append("")
    lines.append("| Стратегия | Очки |")
    lines.append("| --- | ---: |")
    for name, score in sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {name} | {score} |")
    lines.append("")
    lines.append("## Длина наибольшей доминирующей серии")
    lines.append("")
    lines.append("| Стратегия | Макс. серия |")
    lines.append("| --- | ---: |")
    for name, run_len in sorted(max_dom_runs.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {name} | {run_len} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    total_scores, max_dominating_runs = run_tournament()
    save_results_markdown(
        path="lab2_results.md",
        total_scores=total_scores,
        max_dom_runs=max_dominating_runs,
    )
    # Also print a short summary
    print("Total scores:")
    for name, score in sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:>7}: {score}")
    print("Max dominating runs:")
    for name, run in sorted(max_dominating_runs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:>7}: {run}")


if __name__ == "__main__":
    main()


