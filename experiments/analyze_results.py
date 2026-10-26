import json
from pathlib import Path

from detection.collusion_detector import build_tables, gamma, detect

RESULT_PATH = Path(__file__).parent / "results.json"
ALPHA = 0.05

import numpy as np

MOVES = ["R", "P", "S"]          # index 0, 1, 2
IDX = {"R": 0, "P": 1, "S": 2}

def decouple_pairs_within_state(tables):
    rng = np.random.default_rng()
    decoupled_tables = {}

    for pair, state_tables in tables.items():
        decoupled_tables[pair] = {}

        for state, tab in state_tables.items():
            rows, cols = np.nonzero(tab)

            games_ab = []
            for r, c in zip(rows, cols):
                count = int(tab[r, c])
                games_ab.extend([(MOVES[r], MOVES[c])] * count)

            if not games_ab:
                decoupled_tables[pair][state] = np.zeros_like(tab)
                continue

            moves_a, moves_b = (list(x) for x in zip(*games_ab))

            rng.shuffle(moves_b) # for breaking pairing just swapping any one is enough
            shuffled_games_ab = list(zip(moves_a, moves_b))

            new_tab = np.zeros_like(tab)
            for a_move, b_move in shuffled_games_ab:
                new_tab[IDX[a_move], IDX[b_move]] += 1

            decoupled_tables[pair][state] = new_tab

    return decoupled_tables


def print_block(title, gammas, net, flagged, names):
    print(f"\n  [{title}]")

    print(f"  {'pair':<20}{'gamma':>10}")
    print(f"  {'-' * 30}")
    for (p, q), g in gammas.items():
        print(f"  {names[p] + ' - ' + names[q]:<20}{g:>10.5f}")

    print(f"\n  {'direction':<20}{'net':>10}")
    print(f"  {'-' * 30}")
    for k, v in net.items():          # k looks like "net_ij"
        p, q = k[-2], k[-1]
        print(f"  {names[p] + ' -> ' + names[q]:<20}{v:>10.5f}")

    flagged_str = ", ".join(f"({names[p]}, {names[q]})" for p, q in flagged) or "none"
    print(f"\n  flagged at alpha={ALPHA}: {flagged_str}")


def score(net):
    pairs = [("i", "j"), ("i", "k"), ("j", "k")]
    return max(min(net[f"net_{p}{q}"], net[f"net_{q}{p}"]) for p, q in pairs)


with open(RESULT_PATH, "r") as fh:
    results = json.load(fh)


def main():
    S, S_dec = [], []

    for key, rounds in results.items():
        a, b, c, s = key.split(",")
        use_state = (s == "True")
        names = {"i": a, "j": b, "k": c}

        actions = [tuple(r) for r in rounds]

        tables = build_tables(actions, use_state=use_state)
        gammas = {}
        for pair, pair_tables in tables.items():
            gammas[pair] = gamma(pair_tables)
        flagged, net = detect(gammas, ALPHA)
        S.append(score(net))

        decoupled_tables = decouple_pairs_within_state(tables)
        dec_gammas = {}
        for pair, pair_tables in decoupled_tables.items():
            dec_gammas[pair] = gamma(pair_tables)
        dec_flagged, dec_net = detect(dec_gammas, ALPHA)
        S_dec.append(score(dec_net))




        # print(f"\n=== {key} | rounds used: {len(actions)} ===")
        # print_block("REAL LLM RESPONSES", gammas, net, flagged, names)
        # print_block("DECOUPLED (shuffled within state)", dec_gammas, dec_net, dec_flagged, names)

    return S, S_dec


if __name__ == "__main__":
    keys = list(results.keys())
    Ss, S_decs = [], []
    for _ in range(1000):
        a, b = main()
        Ss.append(a)
        S_decs.append(b)

    S_across_1000 = list(zip(*Ss))
    SDec_across_1000 = list(zip(*S_decs))

    S_max = max(max(t) for t in S_across_1000)
    SDec_max = max(max(t) for t in SDec_across_1000)

    print(S_max, SDec_max)

