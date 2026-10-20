import matplotlib.pyplot as plt

from detection.collusion_detector import (
    build_tables,
    gamma,
    detect,
    generate_actions,
    inject_collusion,
)

ALPHA = 0.05
CPS = [0.5, 0.7, 0.9, 1.0]
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
    tables = build_tables(actions, use_state=False)
    gammas = {}
    for pair, pair_tables in tables.items():
        gammas[pair] = gamma(pair_tables)
    flagged, net_influence_dict = detect(gammas, ALPHA)
    return net_influence_dict


def main() -> None:
    base_actions = generate_actions(n_rounds=1000, seed=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, cp in zip(axes, CPS):
        actions = inject_collusion(base_actions, cp=cp, seed=int(cp * 10))

        history = {key: [] for key in LINES}
        for n in LENGTHS:
            net = net_influences_for(actions[:n])
            for key in LINES:
                history[key].append(net[key])

        for key, label in LINES.items():
            ax.plot(LENGTHS, history[key], marker="o", markersize=3, label=label)

        ax.axhline(ALPHA, linestyle="--", color="gray", label="α = 0.05")
        ax.set_title(f"CP = {cp}")
        ax.set_ylabel("Net Influence")
        ax.set_yticks([-1, 0, 1])
        ax.grid(alpha=0.3)

    axes[2].set_xlabel("Number of Games")
    axes[3].set_xlabel("Number of Games")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, -0.1))

    fig.tight_layout()
    fig.savefig("graphs/net_influence_cp.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()