import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import itertools
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

EPS = 1e-2  # state regularization
EPS_floor = 0  # EPS / 10000   # EPS / 10000     # Petz update stabilization. Set 0 for not stabilized evolution
EPSP = 1e-16    # approximate zero for expressions of type p_i log p_i


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def hermitian_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def safe_eigh(a: np.ndarray):
    vals, vecs = np.linalg.eigh(hermitian_part(a))
    vals = np.maximum(vals, 1e-16)
    return vals, vecs


def matrix_sqrt_from_eigh(vals: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T


def matrix_sqrt(a: np.ndarray) -> np.ndarray:
    vals, vecs = safe_eigh(a)
    return matrix_sqrt_from_eigh(vals, vecs)


def normalize_density_matrix(rho: np.ndarray) -> np.ndarray:
    rho = hermitian_part(rho)
    tr = np.real(np.trace(rho))

    return rho / tr


def random_density_matrix(
    d: int,
    rng: np.random.Generator,
    eps: float = 1e-5,
) -> np.ndarray:
    u = rng.random()

    if u < 0.2:
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

    else:
        if u < 0.5:
            k = min(d, int(rng.choice([2, 3, 4, max(2, d // 8), max(2, d // 4)])))
        elif u < 0.85:
            k = int(rng.integers(max(2, d // 3), max(3, d // 2) + 1))
            k = min(k, d)
        else:
            k = d

        a = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
        rho = a @ a.conj().T
        rho = normalize_density_matrix(rho)

    rho = (1.0 - eps) * rho + eps * np.eye(d, dtype=complex) / d
    return normalize_density_matrix(rho)


# -----------------------------------------------------------------------------
# Fidelity / infidelity
# -----------------------------------------------------------------------------

def fidelity_from_sqrt_rho(
    sqrt_rho: np.ndarray,
    sigma: np.ndarray,
) -> float:
    middle = hermitian_part(sqrt_rho @ sigma @ sqrt_rho)
    vals, _ = safe_eigh(middle)
    f = np.sum(np.sqrt(vals)) ** 2
    return float(np.clip(np.real(f), 0.0, 1.0))


def infidelity_from_sqrt_rho(
    sqrt_rho: np.ndarray,
    sigma: np.ndarray,
) -> float:
    return float(1.0 - fidelity_from_sqrt_rho(sqrt_rho, sigma))


# -----------------------------------------------------------------------------
# Pauli bases
# -----------------------------------------------------------------------------

def qubit_basis_unitary(label: str) -> np.ndarray:
    label = label.upper()

    if label == "Z":
        return np.eye(2, dtype=complex)

    if label == "X":
        return np.array(
            [[1.0, 1.0],
             [1.0, -1.0]],
            dtype=complex,
        ) / np.sqrt(2.0)

    if label == "Y":
        return np.array(
            [[1.0, 1.0],
             [1.0j, -1.0j]],
            dtype=complex,
        ) / np.sqrt(2.0)

    raise ValueError(f"Unknown basis label: {label}")


def all_pauli_settings(n_qubits: int) -> list[tuple[str, ...]]:
    return list(itertools.product(["X", "Y", "Z"], repeat=n_qubits))


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def check_inputs(
    settings: list[tuple[str, ...]],
    Nj: np.ndarray,
    target_probs: np.ndarray | None = None,
) -> None:
    if len(settings) != len(Nj):
        raise ValueError(
            f"Need one Nj per setting: len(settings)={len(settings)}, "
            f"len(Nj)={len(Nj)}."
        )

    if len(settings) == 0:
        raise ValueError("settings is empty.")

    n_qubits = len(settings[0])
    d = 2**n_qubits

    for j, s in enumerate(settings):
        if len(s) != n_qubits:
            raise ValueError(f"Setting {j} has wrong length: {s}")
        for lbl in s:
            if lbl not in ("X", "Y", "Z"):
                raise ValueError(f"Setting {j} has invalid label {lbl}.")

    if target_probs is not None:
        target_probs = np.asarray(target_probs)
        if target_probs.shape != (len(settings), d):
            raise ValueError(
                f"target_probs has shape {target_probs.shape}, "
                f"expected {(len(settings), d)}."
            )


# -----------------------------------------------------------------------------
# Batched setting cache
# -----------------------------------------------------------------------------

def prepare_setting_cache(
    settings: list[tuple[str, ...]],
    Nj: np.ndarray,
):
    Nj = np.asarray(Nj, dtype=float)
    Ntot = np.sum(Nj)

    if Ntot <= 0:
        raise ValueError("Sum of Nj must be positive.")

    local = {
        "X": qubit_basis_unitary("X"),
        "Y": qubit_basis_unitary("Y"),
        "Z": qubit_basis_unitary("Z"),
    }

    U_stack = np.stack(
        [
            kron_all([local[lbl] for lbl in setting])
            for setting in settings
        ],
        axis=0,
    )

    Udag_stack = np.conjugate(np.swapaxes(U_stack, -1, -2))
    weights = Nj / Ntot

    return settings, U_stack, Udag_stack, weights


# -----------------------------------------------------------------------------
# Batched probabilities
# -----------------------------------------------------------------------------

def probabilities_all_bases_from_cache(
    rho: np.ndarray,
    U_stack: np.ndarray,
    Udag_stack: np.ndarray,
) -> np.ndarray:
    tmp = Udag_stack @ rho @ U_stack
    return np.real(np.diagonal(tmp, axis1=1, axis2=2))


def precompute_target_probabilities(
    R: np.ndarray,
    U_stack: np.ndarray,
    Udag_stack: np.ndarray,
) -> np.ndarray:
    return probabilities_all_bases_from_cache(R, U_stack, Udag_stack)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_sigma_from_probs(
    sigma: np.ndarray,
    probs_all: np.ndarray,
    weights: np.ndarray,
    target_probs: np.ndarray,
    sqrt_R: np.ndarray | None = None,
    epsp: float = EPSP,
    compute_infidelity: bool = True,
) -> dict:
    p_safe = np.maximum(probs_all, epsp)

    ll = np.sum(weights[:, None] * target_probs * np.log(p_safe))

    diff = np.abs(probs_all - target_probs)
    err_l1 = np.sum(weights[:, None] * diff)
    err_max = np.max(diff)

    infid = None
    if sqrt_R is not None and compute_infidelity:
        infid = infidelity_from_sqrt_rho(sqrt_R, sigma)

    return {
        "log_likelihood": float(ll),
        "prob_l1_error": float(err_l1),
        "prob_max_error": float(err_max),
        "infidelity": infid,
    }


# -----------------------------------------------------------------------------
# Batched Petz update
# -----------------------------------------------------------------------------

def update_sigma_weighted_cached(
    sqrt_sigma: np.ndarray,
    U_stack: np.ndarray,
    weights: np.ndarray,
    target_probs: np.ndarray,
    probs_all: np.ndarray,
    epsp: float = EPSP,
    eig_floor: float = EPS_floor,
) -> np.ndarray:
    p_safe = np.maximum(probs_all, epsp)
    coeffs = target_probs / p_safe

    W = sqrt_sigma @ U_stack

    weighted_coeffs = weights[:, None] * coeffs

    sigma_plus = np.einsum(
        "jai,ji,jbi->ab",
        W,
        weighted_coeffs,
        np.conjugate(W),
        optimize=True,
    )

    d = sigma_plus.shape[0]
    sigma_plus = (1.0 - eig_floor) * sigma_plus + (eig_floor/ d) * np.eye(d, dtype=complex)

    return normalize_density_matrix(sigma_plus)


# -----------------------------------------------------------------------------
# Iteration driver
# -----------------------------------------------------------------------------

def iterate_weighted_update(
    sigma0: np.ndarray,
    settings: list[tuple[str, ...]],
    Nj: np.ndarray,
    R: np.ndarray | None = None,
    target_probs: np.ndarray | None = None,
    n_steps: int = 50,
    diagnostics_every: int = 10,
    store_probs_history: bool = False,
    eps: float = EPS,
    epsp: float = EPSP,
    verbose: bool = True,
):
    check_inputs(settings, Nj, target_probs)

    settings, U_stack, Udag_stack, weights = prepare_setting_cache(settings, Nj)

    if target_probs is None:
        if R is None:
            raise ValueError("Provide either R or target_probs.")
        R = normalize_density_matrix(np.array(R, dtype=complex))
        target_probs = precompute_target_probabilities(R, U_stack, Udag_stack)

    check_inputs(settings, Nj, target_probs)

    sqrt_R = matrix_sqrt(R) if R is not None else None

    sigma = normalize_density_matrix(np.array(sigma0, dtype=complex))

    vals_sigma, vecs_sigma = safe_eigh(sigma)
    sqrt_sigma = matrix_sqrt_from_eigh(vals_sigma, vecs_sigma)

    probs_all = probabilities_all_bases_from_cache(sigma, U_stack, Udag_stack)

    eval_current = evaluate_sigma_from_probs(
        sigma=sigma,
        probs_all=probs_all,
        weights=weights,
        target_probs=target_probs,
        sqrt_R=sqrt_R,
        epsp=epsp,
        compute_infidelity=True,
    )

    history = {
        "log_likelihood": [eval_current["log_likelihood"]],
        "delta": [],
        "step_change": [],
        "prob_l1_error": [eval_current["prob_l1_error"]],
        "prob_max_error": [eval_current["prob_max_error"]],
        "infidelity": [],
        "infidelity_iters": [],
    }

    if store_probs_history:
        history["probs"] = [probs_all.copy()]

    if eval_current["infidelity"] is not None:
        history["infidelity"].append(eval_current["infidelity"])
        history["infidelity_iters"].append(0)

    if verbose:
        msg = (
            f"step=  0 | "
            f"L = {eval_current['log_likelihood']:+.12f} | "
            f"L1err = {eval_current['prob_l1_error']:.12e} | "
            f"maxerr = {eval_current['prob_max_error']:.12e}"
        )
        if eval_current["infidelity"] is not None:
            msg += f" | infid = {eval_current['infidelity']:.12e}"
        print(msg)

    ll_prev = eval_current["log_likelihood"]

    for step in range(1, n_steps + 1):
        sigma_new = update_sigma_weighted_cached(
            sigma=sigma,
            sqrt_sigma=sqrt_sigma,
            vals_sigma=vals_sigma,
            vecs_sigma=vecs_sigma,
            U_stack=U_stack,
            weights=weights,
            target_probs=target_probs,
            probs_all=probs_all,
            eps=eps,
            epsp=epsp,
        )

        step_change = np.linalg.norm(sigma_new - sigma, ord="fro")

        vals_sigma_new, vecs_sigma_new = safe_eigh(sigma_new)
        sqrt_sigma_new = matrix_sqrt_from_eigh(vals_sigma_new, vecs_sigma_new)

        probs_all_new = probabilities_all_bases_from_cache(
            sigma_new,
            U_stack,
            Udag_stack,
        )

        compute_heavy = (step % diagnostics_every == 0) or (step == n_steps)

        eval_new = evaluate_sigma_from_probs(
            sigma=sigma_new,
            probs_all=probs_all_new,
            weights=weights,
            target_probs=target_probs,
            sqrt_R=sqrt_R,
            epsp=epsp,
            compute_infidelity=compute_heavy,
        )

        delta = eval_new["log_likelihood"] - ll_prev

        history["log_likelihood"].append(eval_new["log_likelihood"])
        history["delta"].append(delta)
        history["step_change"].append(step_change)
        history["prob_l1_error"].append(eval_new["prob_l1_error"])
        history["prob_max_error"].append(eval_new["prob_max_error"])

        if store_probs_history:
            history["probs"].append(probs_all_new.copy())

        if eval_new["infidelity"] is not None:
            history["infidelity"].append(eval_new["infidelity"])
            history["infidelity_iters"].append(step)

        if verbose:
            msg = (
                f"step={step:3d} | "
                f"L = {eval_new['log_likelihood']:+.12f} | "
                f"Delta = {delta:+.12e} | "
                f"dSigma = {step_change:.12e} | "
                f"L1err = {eval_new['prob_l1_error']:.12e} | "
                f"maxerr = {eval_new['prob_max_error']:.12e}"
            )
            if eval_new["infidelity"] is not None:
                msg += f" | infid = {eval_new['infidelity']:.12e}"
            print(msg)

        sigma = sigma_new
        vals_sigma = vals_sigma_new
        vecs_sigma = vecs_sigma_new
        sqrt_sigma = sqrt_sigma_new
        probs_all = probs_all_new
        ll_prev = eval_new["log_likelihood"]

    return sigma, history


# -----------------------------------------------------------------------------
# Batch simulation
# -----------------------------------------------------------------------------

def run_single_case_infidelity(
    seed: int,
    n_qubits: int,
    n_steps: int,
    diagnostics_every: int = 10,
    Nj_value: float = 700.0,
    eps: float = EPS,
    epsp: float = EPSP,
) -> dict:
    rng = np.random.default_rng(seed)

    d = 2**n_qubits
    settings = all_pauli_settings(n_qubits)
    Nj = np.full(len(settings), Nj_value, dtype=float)

    R = random_density_matrix(d, rng, eps=eps)
    sigma0 = random_density_matrix(d, rng, eps=eps)

    _, history = iterate_weighted_update(
        sigma0=sigma0,
        settings=settings,
        Nj=Nj,
        R=R,
        n_steps=n_steps,
        diagnostics_every=diagnostics_every,
        store_probs_history=False,
        eps=eps,
        epsp=epsp,
        verbose=False,
    )

    return {
        "seed": seed,
        "infidelity": np.array(history["infidelity"], dtype=float),
        "iters": np.array(history["infidelity_iters"], dtype=int),
    }


def run_many_cases_infidelity(
    n_cases: int = 50,
    base_seed: int = 1234,
    n_qubits: int = 4,
    n_steps: int = 10000,
    diagnostics_every: int = 10,
    Nj_value: float = 700.0,
    max_workers: int | None = None,
    parallel: bool = True,
) -> list[dict]:

    seeds = [base_seed + i for i in range(n_cases)]

    if not parallel:
        results = []
        for i, seed in enumerate(seeds):
            result = run_single_case_infidelity(
                seed=seed,
                n_qubits=n_qubits,
                n_steps=n_steps,
                diagnostics_every=diagnostics_every,
                Nj_value=Nj_value,
            )
            results.append(result)
            print(
                f"case {i + 1:3d}/{n_cases} | "
                f"seed={seed} | "
                f"final infid={result['infidelity'][-1]:.6e}"
            )
        return results

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                run_single_case_infidelity,
                seed,
                n_qubits,
                n_steps,
                diagnostics_every,
                Nj_value,
            ): seed
            for seed in seeds
        }

        for k, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)

            print(
                f"case {k:3d}/{n_cases} | "
                f"seed={result['seed']} | "
                f"final infid={result['infidelity'][-1]:.6e}"
            )

    results.sort(key=lambda r: r["seed"])
    return results


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------

def save_infidelity_results(
    results: list[dict],
    filename: str = "infidelity_runs.npz",
) -> None:
    infidelity_list = np.array([r["infidelity"] for r in results], dtype=object)
    iters_list = np.array([r["iters"] for r in results], dtype=object)
    seeds = np.array([r["seed"] for r in results], dtype=int)

    np.savez(
        filename,
        infidelity=infidelity_list,
        iters=iters_list,
        seeds=seeds,
    )

    print(f"Saved results to {filename}")


def load_infidelity_results(
    filename: str = "infidelity_runs.npz",
) -> list[dict]:
    data = np.load(filename, allow_pickle=True)

    results = []

    for seed, infid, iters in zip(
        data["seeds"],
        data["infidelity"],
        data["iters"],
    ):
        results.append({
            "seed": int(seed),
            "infidelity": np.array(infid, dtype=float),
            "iters": np.array(iters, dtype=int),
        })

    return results


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_all_infidelity(
    results: list[dict],
    logscale: bool = True,
    filename: str = "qc_infidelity.pdf",
) -> None:
    plt.figure(figsize=(7, 4))

    for r in results:
        x = r["iters"]
        y = r["infidelity"]

        if logscale:
            y = np.maximum(y, 1e-16)
            plt.semilogy(x, y, alpha=0.35)
        else:
            plt.plot(x, y, alpha=0.35)

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.title("Infidelity trajectories")
    plt.grid(True, which="both" if logscale else "major")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_mean_infidelity(
    results: list[dict],
    logscale: bool = True,
) -> None:
    iters = results[0]["iters"]
    Y = np.array([r["infidelity"] for r in results], dtype=float)

    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)

    plt.figure(figsize=(7, 4))

    if logscale:
        mean_plot = np.maximum(mean, 1e-16)
        lower = np.maximum(mean - std, 1e-16)
        upper = np.maximum(mean + std, 1e-16)

        plt.semilogy(iters, mean_plot, label="mean")
        plt.fill_between(iters, lower, upper, alpha=0.2, label="std")
    else:
        plt.plot(iters, mean, label="mean")
        plt.fill_between(iters, mean - std, mean + std, alpha=0.2, label="std")

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.title("Mean infidelity")
    plt.legend()
    plt.grid(True, which="both" if logscale else "major")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    n_cases = 20
    n_qubits = 6
    n_steps = 10000
    diagnostics_every = 10
    output_file = "infidelity_runs_q6_10cases.npz"

    results = run_many_cases_infidelity(
        n_cases=n_cases,
        base_seed=1234,
        n_qubits=n_qubits,
        n_steps=n_steps,
        diagnostics_every=diagnostics_every,
        Nj_value=700.0,
        max_workers=None,
        parallel=True,
    )

    save_infidelity_results(results, filename=output_file)

    plot_all_infidelity(results, logscale=True)
    plot_mean_infidelity(results, logscale=True)