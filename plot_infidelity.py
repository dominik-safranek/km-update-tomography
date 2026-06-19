import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator


def load_infidelity_results(filename: str):
    """
    Load results saved by save_infidelity_results(...).

    Expected arrays in the .npz file:
      - infidelity
      - iters
      - seeds
    """
    data = np.load(filename, allow_pickle=True)

    infidelity_list = data["infidelity"]
    iters_list = data["iters"]
    seeds = data["seeds"]

    results = []
    for seed, infid, iters in zip(seeds, infidelity_list, iters_list):
        results.append({
            "seed": int(seed),
            "infidelity": np.array(infid, dtype=float),
            "iters": np.array(iters, dtype=int),
        })
    return results


def plot_all_infidelity(results, output_png=None):
    plt.figure(figsize=(7, 4))

    for r in results:
        x = r["iters"]
        y = np.maximum(r["infidelity"], 1e-16)
        plt.semilogy(x, y, alpha=1)   #alpha=0.35

    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.6)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.25)

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    #plt.title("Infidelity trajectories")
    plt.tight_layout()

    plt.xlim(0, 10000)
    plt.ylim(1e-4, 1)    #1e-4

    if output_png:
        plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.show()


def plot_mean_infidelity(results, output_png=None):
    iters = results[0]["iters"]
    Y = np.array([r["infidelity"] for r in results], dtype=float)

    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)

    plt.figure(figsize=(7, 4))
    plt.semilogy(iters, np.maximum(mean, 1e-16), label="mean")
    plt.fill_between(
        iters,
        np.maximum(mean - std, 1e-16),
        np.maximum(mean + std, 1e-16),
        alpha=0.2,
        label="std",
    )

    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.6)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.25)

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.title("Mean infidelity")

    plt.xlim(0, 10000)
    plt.ylim(1e-4, 1)

    plt.legend()
    plt.tight_layout()



    if output_png:
        plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Change this to your saved file
    filename = "infidelity_runs_q6_20cases_stabilized.npz"

    results = load_infidelity_results(filename)

    print(f"Loaded {len(results)} runs from {filename}")
    print(f"First seed: {results[0]['seed']}")
    print(f"Final infidelity of first run: {results[0]['infidelity'][-1]:.6e}")

    # Generates the graphs with cleaner log-scale labels/grid
    plot_all_infidelity(results, output_png="all_infidelity.png")
    plot_mean_infidelity(results, output_png="mean_infidelity.png")
