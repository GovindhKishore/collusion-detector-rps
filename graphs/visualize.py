import matplotlib.pyplot as plt

from detection.collusion_detector import (
    build_tables,
    gamma,
    detect,
    generate_actions,
    inject_collusion,
)

ALPHA = 0.05
CPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LENGTHS = list(range(50, 1001, 10))

LINES = {
    "net_ij": "Γ(A;B)",
    "net_ji": "Γ(B;A)",
    "net_ik": "Γ(A;C)",
    "net_ki": "Γ(C;A)",
    "net_jk": "Γ(B;C)",
    "net_kj": "Γ(C;B)",
}


def net_influences_for(actions: list[tuple[str, str, str]]) -> dict[str, float]:
    tables = build_tables(actions, use_state=True)
    gammas = {}
    for pair, pair_tables in tables.items():
        gammas[pair] = gamma(pair_tables)
    flagged, net_influence_dict = detect(gammas, ALPHA)
    return net_influence_dict


def plot_net_influence_grid() -> dict[float, dict[str, list[float]]]:
    base_actions = generate_actions(n_rounds=1000, seed=0)

    fig, axes = plt.subplots(5, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    all_history = {}
    for ax, cp in zip(axes, CPS):
        actions = inject_collusion(base_actions, cp=cp, seed=int(cp * 10))

        history = {key: [] for key in LINES}
        for n in LENGTHS:
            net = net_influences_for(actions[:n])
            for key in LINES:
                history[key].append(net[key])
        all_history[cp] = history

        for key, label in LINES.items():
            ax.plot(LENGTHS, history[key], marker="o", markersize=3, label=label)

        ax.axhline(ALPHA, linestyle="--", color="gray", label="α = 0.05")
        ax.set_title(f"CP = {cp}")
        ax.set_ylabel("Net Influence")
        ax.grid(alpha=0.3)

    axes[8].set_xlabel("Number of Games")
    axes[9].set_xlabel("Number of Games")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.007))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    fig.savefig("graphs/net_influence_cp.png", dpi=200)
    return all_history


def plot_net_influence_vs_cp(all_history: dict[float, dict[str, list[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for key, label in LINES.items():
        values = [all_history[cp][key][-1] for cp in CPS]
        ax.plot(CPS, values, marker="o", markersize=4, label=label)

    ax.axhline(ALPHA, linestyle="--", color="gray", label="α = 0.05")
    ax.set_xticks(CPS)
    ax.set_xlabel("Collusion Probability")
    ax.set_ylabel("Net Influence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("graphs/net_influence_vs_cp.png", dpi=200)


def write_cda_table(use_state: bool, n_iterations: int = 1000) -> None:
    cp_values = [0.3, 0.4]
    game_counts = list(range(100, 1001, 100))
    correct = {cp: {n: 0 for n in game_counts} for cp in cp_values}

    for cp in cp_values:
        for seed in range(n_iterations):
            base_actions = generate_actions(n_rounds=1000, seed=seed)
            actions = inject_collusion(base_actions, cp=cp, seed=100000 + seed)

            for n in game_counts:
                tables = build_tables(actions[:n], use_state=use_state)
                gammas = {}
                for pair, pair_tables in tables.items():
                    gammas[pair] = gamma(pair_tables)
                flagged, _ = detect(gammas, ALPHA)
                if flagged == [("i", "k")]:
                    correct[cp][n] += 1

    lines = ["Number of Games | CDA % (CP=0.3) | CDA % (CP=0.4)"]
    for n in game_counts:
        cda_03 = 100 * correct[0.3][n] / n_iterations
        cda_04 = 100 * correct[0.4][n] / n_iterations
        lines.append(f"{n:15d} | {cda_03:14.1f} | {cda_04:14.1f}")

    table = "\n".join(lines)
    print(table)
    with open("graphs/cda_table.txt", "w", encoding="utf-8") as f:
        f.write(table + "\n")



def main() -> None:
    all_history = plot_net_influence_grid()
    plot_net_influence_vs_cp(all_history)
    write_cda_table(use_state=True)

if __name__ == "__main__":
    main()