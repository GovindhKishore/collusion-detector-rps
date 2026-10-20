import numpy as np
import numpy.typing as npt
import random


AGENTS = {'i', 'j', 'k'}
ACTION_IDX = {"R": 0, "P": 1, "S": 2}
BEATEN_BY = {"R": "S", "S": "P", "P": "R"}


def gamma(tables: dict[tuple[str, str, str] | str, npt.NDArray[np.float64]]) -> float:
    total_rounds = float(sum(t.sum() for t in tables.values()))

    if total_rounds == 0:
        return 0.0

    individual_influence = 0.0
    for t in tables.values():
        n = t.sum().item() # N(s)
        if n == 0:
            continue
        p_joint = t / n  # p(ai, aj | s)
        p_i = p_joint.sum(axis=1, keepdims=True)  # p(ai | s) assuming row labels corresponds to model i
        p_j = p_joint.sum(axis=0, keepdims=True)  # p(aj | s) assuming col labels corresponds to model j
        expected = p_i @ p_j  # independent prediction using marginal probs, (0, 1) element gives p(Ri | s) * p(Pj | s)
        m = p_joint > 0  # only considering probs > 0 to make 0 * log(0) = 0
        inner = (p_joint[m] * np.log(p_joint[m] / expected[m])).sum().item()
        individual_influence += (n / total_rounds) * inner  # outer sum p(s) component

    return individual_influence


def net_influence(gammas: dict[tuple[str, str], float], a: str, b: str) -> float:

    # the only agent other than a and b
    c = [x for x in AGENTS if x != a and x != b][0]

    influence_ab = gammas[(a, b)]  # individual influence of a and b
    influence_cb = gammas[(c, b)]  # the "max over others" term, only one other agent exists

    return influence_ab - influence_cb


def build_tables(
    actions: list[tuple[str, str, str]],
    use_state: bool,
) -> dict[tuple[str, str], dict[tuple[str, str, str] | str, npt.NDArray[np.float64]]]:

    pairs = {("i", "j"): (0, 1), ("j", "k"): (1, 2), ("i", "k"): (0, 2)}

    all_tables = {}
    for pair, (x, y) in pairs.items():
        tables = {}
        for t in range(len(actions)):
            if use_state and t > 0:
                state = actions[t - 1]
            else:
                state = "None"

            if state not in tables:
                tables[state] = np.zeros((3, 3))

            row = ACTION_IDX[actions[t][x]]  # first agent - rows
            col = ACTION_IDX[actions[t][y]] # second agent - columns
            tables[state][row, col] += 1
        all_tables[pair] = tables
    return all_tables


def detect(gammas: dict[tuple[str, str], float], alpha: float) -> tuple[list[tuple[str, str]], dict[str, float]]:
    g_ij = gammas[("i", "j")]
    g_ik = gammas[("i", "k")]
    g_jk = gammas[("j", "k")]

    net_ij = g_ij - g_jk
    net_ji = g_ij - g_ik
    net_ik = g_ik - g_jk
    net_ki = g_ik - g_ij
    net_jk = g_jk - g_ik
    net_kj = g_jk - g_ij

    net_influence_dict = {"net_ij": net_ij, "net_ji": net_ji, "net_ik": net_ik, "net_ki": net_ki, "net_jk": net_jk, "net_kj": net_kj}

    flagged = []
    if net_ij >= alpha and net_ji >= alpha:
        flagged.append(("i", "j"))
    if net_ik >= alpha and net_ki >= alpha:
        flagged.append(("i", "k"))
    if net_jk >= alpha and net_kj >= alpha:
        flagged.append(("j", "k"))

    return flagged, net_influence_dict


def generate_actions(n_rounds: int, seed: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    actions = []
    for _ in range(n_rounds):
        r = []
        for _ in range(3):
            r.append(rng.choice("RPS"))
        actions.append(tuple(r))
    return actions


def inject_collusion(actions: list[tuple[str, str, str]], cp: float, seed: int) -> list[tuple[str, str, str]]:
    n_collude = round(cp * len(actions))
    new_actions = []
    for count, (a, b, c) in enumerate(actions):
        if count < n_collude:
            new_actions.append((a, b, BEATEN_BY[a]))
        else:
            new_actions.append((a, b, c))

    rng = random.Random(seed)
    rng.shuffle(new_actions)
    return new_actions


def main() -> None:
    base_actions = generate_actions(n_rounds=1000, seed=0)

    action05 = inject_collusion(base_actions, cp=0.5, seed=1)
    action07 = inject_collusion(base_actions, cp=0.7, seed=2)
    action09 = inject_collusion(base_actions, cp=0.9, seed=3)
    action1 = inject_collusion(base_actions, cp=1.0, seed=4)

    datasets = {
        0.5: action05,
        0.7: action07,
        0.9: action09,
        1.0: action1,
    }

    alpha = 0.05
    for cp, actions in datasets.items():
        tables = build_tables(actions, use_state=False)

        gammas = {}
        for pair, pair_tables in tables.items():
            gammas[pair] = gamma(pair_tables)

        flagged, _ = detect(gammas, alpha)

        print(f"CP={cp}")
        print(f"  gammas: {gammas}")
        print(f"  flagged pairs: {flagged}")


if __name__ == "__main__":
    main()