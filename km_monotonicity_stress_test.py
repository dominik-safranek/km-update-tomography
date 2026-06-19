import copy
import numpy as np
from joblib import Parallel, delayed

np.set_printoptions(precision=12, suppress=False)

EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def hermitian_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def safe_eigh(a: np.ndarray, eps: float = EPS):
    vals, vecs = np.linalg.eigh(hermitian_part(a))
    vals = np.maximum(vals, eps)
    return vals, vecs


def matrix_log_from_eig(
    vals: np.ndarray,
    vecs: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.conj().T


def matrix_sqrt(a: np.ndarray, eps: float = EPS) -> np.ndarray:
    vals, vecs = safe_eigh(a, eps=eps)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T


def normalize_density_matrix(rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    rho = hermitian_part(rho)
    tr = np.real(np.trace(rho))
    if tr < eps:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    return rho / tr


# -----------------------------------------------------------------------------
# Random generation
# -----------------------------------------------------------------------------

def random_complex_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))


def random_density_matrix(
    d: int,
    rng: np.random.Generator,
    eps: float = 1e-8,
) -> np.ndarray:
    a = random_complex_matrix(d, rng)
    rho = a @ a.conj().T + eps * np.eye(d)
    return normalize_density_matrix(rho)


def random_kraus_channel(
    d: int,
    n_kraus: int,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = np.stack([random_complex_matrix(d, rng) for _ in range(n_kraus)], axis=0)
    m = np.einsum("aji,ajk->ik", raw.conj(), raw)
    vals, vecs = safe_eigh(m)
    m_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T
    kraus = np.einsum("aij,jk->aik", raw, m_inv_sqrt)
    return kraus


# -----------------------------------------------------------------------------
# Channel-dependent operations
# -----------------------------------------------------------------------------

def channel(rho: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    return np.einsum("aij,jk,alk->il", kraus, rho, kraus.conj())


def channel_adjoint(a: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    return np.einsum("aji,jk,akl->il", kraus.conj(), a, kraus)


def log_likelihood_from_y(x: np.ndarray, y: np.ndarray) -> float:
    vals_y, vecs_y = safe_eigh(y)
    return float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))


def log_likelihood(sigma: np.ndarray, x: np.ndarray, kraus: np.ndarray) -> float:
    y = channel(sigma, kraus)
    return log_likelihood_from_y(x, y)


# -----------------------------------------------------------------------------
# Exact KM score S_Y = Omega_Y^{-1}
# -----------------------------------------------------------------------------

def omega_inverse_from_eig(
    vals: np.ndarray,
    vecs: np.ndarray,
    a: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    vals = np.maximum(vals, eps)
    b = vecs.conj().T @ a @ vecs

    lam_i = vals[:, None]
    lam_j = vals[None, :]

    diff = lam_i - lam_j
    log_diff = np.log(lam_i) - np.log(lam_j)

    m = lam_i * np.ones_like(diff)
    mask = np.abs(diff) >= 1e-14
    m[mask] = diff[mask] / log_diff[mask]

    m = np.maximum(m, eps)
    out = b / m
    return vecs @ out @ vecs.conj().T


# -----------------------------------------------------------------------------
# KM update
# -----------------------------------------------------------------------------

def km_update(
    sigma: np.ndarray,
    x: np.ndarray,
    kraus: np.ndarray,
    vals_y: np.ndarray | None = None,
    vecs_y: np.ndarray | None = None,
) -> np.ndarray:
    if vals_y is None or vecs_y is None:
        y = channel(sigma, kraus)
        vals_y, vecs_y = safe_eigh(y)

    g = omega_inverse_from_eig(vals_y, vecs_y, x)
    s = matrix_sqrt(sigma)
    sigma_plus = s @ channel_adjoint(g, kraus) @ s
    sigma_plus = hermitian_part(sigma_plus)
    return normalize_density_matrix(sigma_plus)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def frobenius_commutator_norm(a: np.ndarray, b: np.ndarray) -> float:
    comm = a @ b - b @ a
    return float(np.linalg.norm(comm, ord="fro"))


def format_matrix(name: str, a: np.ndarray) -> None:
    print(f"{name} =")
    print(a)
    print()


# -----------------------------------------------------------------------------
# Single trial
# -----------------------------------------------------------------------------

def run_single_trial(
    seed: int,
    d: int,
    n_kraus: int,
    verbose: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)

    kraus = random_kraus_channel(d=d, n_kraus=n_kraus, rng=rng)
    sigma = random_density_matrix(d=d, rng=rng)
    x = random_density_matrix(d=d, rng=rng)

    y = channel(sigma, kraus)
    vals_y, vecs_y = safe_eigh(y)

    sigma_km = km_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)
    y_km = channel(sigma_km, kraus)

    ll_sigma = float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))
    ll_km = log_likelihood_from_y(x, y_km)

    delta_km = ll_km - ll_sigma

    cptp_defect = np.linalg.norm(
        np.einsum("aji,ajk->ik", kraus.conj(), kraus) - np.eye(d),
        ord="fro",
    )

    result = {
        "kraus": kraus,
        "sigma": sigma,
        "x": x,
        "y": y,
        "sigma_km": sigma_km,
        "y_km": y_km,
        "ll_sigma": ll_sigma,
        "ll_km": ll_km,
        "delta_km": delta_km,
        "cptp_defect": cptp_defect,
        "comm_norm": frobenius_commutator_norm(x, y),
        "eig_sigma": np.linalg.eigvalsh(hermitian_part(sigma)),
        "eig_x": np.linalg.eigvalsh(hermitian_part(x)),
        "eig_y": np.linalg.eigvalsh(hermitian_part(y)),
        "eig_sigma_km": np.linalg.eigvalsh(hermitian_part(sigma_km)),
    }

    if verbose:
        print("=== CPTP check ===")
        print("|| sum_a K_a^dagger K_a - I ||_F =", cptp_defect)
        print()

        format_matrix("sigma", sigma)
        format_matrix("X", x)
        format_matrix("Y = E(sigma)", y)
        format_matrix("sigma_km", sigma_km)
        format_matrix("Y_km = E(sigma_km)", y_km)

        print("eig(sigma)    =", result["eig_sigma"])
        print("eig(X)        =", result["eig_x"])
        print("eig(Y)        =", result["eig_y"])
        print("eig(sigma_km) =", result["eig_sigma_km"])
        print()

        print("|| [X, Y] ||_F =", result["comm_norm"])
        print()

        print("L_X(sigma)    =", ll_sigma)
        print("L_X(sigma_km) =", ll_km)
        print()

        print("Delta_KM =", delta_km)
        print()

        if delta_km > 0.0:
            print("The exact KM update increases the log-likelihood in this trial.")
        else:
            print("The exact KM update does not increase the log-likelihood in this trial.")
        print("-" * 80)

    return result


# -----------------------------------------------------------------------------
# Utilities for local refinement
# -----------------------------------------------------------------------------

def density_from_generator(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    rho = a @ a.conj().T + eps * np.eye(a.shape[0], dtype=complex)
    return normalize_density_matrix(rho)


def kraus_from_raw(raw: np.ndarray) -> np.ndarray:
    m = np.einsum("aji,ajk->ik", raw.conj(), raw)
    vals, vecs = safe_eigh(m)
    m_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T
    return np.einsum("aij,jk->aik", raw, m_inv_sqrt)


def evaluate_case(kraus: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> dict:
    y = channel(sigma, kraus)
    vals_y, vecs_y = safe_eigh(y)

    sigma_km = km_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)
    y_km = channel(sigma_km, kraus)

    ll_sigma = float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))
    ll_km = log_likelihood_from_y(x, y_km)

    delta_km = ll_km - ll_sigma

    cptp_defect = np.linalg.norm(
        np.einsum("aji,ajk->ik", kraus.conj(), kraus) - np.eye(sigma.shape[0]),
        ord="fro",
    )

    return {
        "kraus": kraus,
        "sigma": sigma,
        "x": x,
        "y": y,
        "sigma_km": sigma_km,
        "y_km": y_km,
        "ll_sigma": ll_sigma,
        "ll_km": ll_km,
        "delta_km": delta_km,
        "cptp_defect": cptp_defect,
        "comm_norm": frobenius_commutator_norm(x, y),
        "eig_sigma": np.linalg.eigvalsh(hermitian_part(sigma)),
        "eig_x": np.linalg.eigvalsh(hermitian_part(x)),
        "eig_y": np.linalg.eigvalsh(hermitian_part(y)),
        "eig_sigma_km": np.linalg.eigvalsh(hermitian_part(sigma_km)),
    }


def mutate_density_matrix(
    rho: np.ndarray,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    d = rho.shape[0]
    a = matrix_sqrt(rho)
    noise = scale * (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    a_new = a + noise
    return density_from_generator(a_new)


def mutate_kraus(
    kraus: np.ndarray,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    noise = scale * (
        rng.normal(size=kraus.shape) + 1j * rng.normal(size=kraus.shape)
    )
    raw_new = kraus + noise
    return kraus_from_raw(raw_new)


def print_result_summary(label: str, result: dict) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    print("Delta_KM      =", result["delta_km"])
    print("||[X,Y]||_F   =", result["comm_norm"])
    print("CPTP defect   =", result["cptp_defect"])
    print("eig(sigma)    =", result["eig_sigma"])
    print("eig(X)        =", result["eig_x"])
    print("eig(Y)        =", result["eig_y"])
    print("eig(sigma_km) =", result["eig_sigma_km"])


def refine_best_result(
    best_result: dict,
    n_steps: int = 200000,
    seed: int = 987654321,
    sigma_scale_start: float = 0.05,
    sigma_scale_end: float = 0.002,
    x_scale_start: float = 0.05,
    x_scale_end: float = 0.002,
    kraus_scale_start: float = 0.05,
    kraus_scale_end: float = 0.002,
    p_mutate_sigma: float = 0.35,
    p_mutate_x: float = 0.35,
    p_mutate_kraus: float = 0.20,
    p_mutate_two: float = 0.10,
    use_annealing: bool = True,
    temperature_start: float = 1e-2,
    temperature_end: float = 1e-6,
    restart_after: int = 5000,
    print_every: int = 5000,
) -> dict:
    """
    Local stochastic refinement of the strongest negative Delta_KM case.

    Objective: minimize delta_km (make it as negative as possible).
    """
    probs_sum = p_mutate_sigma + p_mutate_x + p_mutate_kraus + p_mutate_two
    if not np.isclose(probs_sum, 1.0):
        raise ValueError(
            f"Mutation probabilities must sum to 1. Got {probs_sum}."
        )

    rng = np.random.default_rng(seed)

    current = evaluate_case(
        kraus=np.array(best_result["kraus"], copy=True),
        sigma=np.array(best_result["sigma"], copy=True),
        x=np.array(best_result["x"], copy=True),
    )
    best = copy.deepcopy(current)

    n_accept = 0
    n_improve = 0
    stagnation_counter = 0
    last_improvement_step = 0

    window_accept = 0
    window_total = 0

    for step in range(1, n_steps + 1):
        t = (step - 1) / max(1, n_steps - 1)

        if use_annealing:
            temperature = temperature_start * (temperature_end / temperature_start) ** t
        else:
            temperature = 0.0

        sigma_scale_t = sigma_scale_start * (sigma_scale_end / sigma_scale_start) ** t
        x_scale_t = x_scale_start * (x_scale_end / x_scale_start) ** t
        kraus_scale_t = kraus_scale_start * (kraus_scale_end / kraus_scale_start) ** t

        candidate_kraus = current["kraus"]
        candidate_sigma = current["sigma"]
        candidate_x = current["x"]

        # mixture of local / medium / large jumps
        u_scale = rng.random()
        if u_scale < 0.70:
            jump_factor = 0.3
        elif u_scale < 0.95:
            jump_factor = 1.0
        else:
            jump_factor = 3.0

        u = rng.random()
        if u < p_mutate_sigma:
            candidate_sigma = mutate_density_matrix(
                candidate_sigma, rng, sigma_scale_t * jump_factor
            )
        elif u < p_mutate_sigma + p_mutate_x:
            candidate_x = mutate_density_matrix(
                candidate_x, rng, x_scale_t * jump_factor
            )
        elif u < p_mutate_sigma + p_mutate_x + p_mutate_kraus:
            candidate_kraus = mutate_kraus(
                candidate_kraus, rng, kraus_scale_t * jump_factor
            )
        else:
            move_type = rng.integers(0, 3)
            if move_type == 0:
                candidate_sigma = mutate_density_matrix(
                    candidate_sigma, rng, sigma_scale_t * jump_factor
                )
                candidate_x = mutate_density_matrix(
                    candidate_x, rng, x_scale_t * jump_factor
                )
            elif move_type == 1:
                candidate_sigma = mutate_density_matrix(
                    candidate_sigma, rng, sigma_scale_t * jump_factor
                )
                candidate_kraus = mutate_kraus(
                    candidate_kraus, rng, kraus_scale_t * jump_factor
                )
            else:
                candidate_x = mutate_density_matrix(
                    candidate_x, rng, x_scale_t * jump_factor
                )
                candidate_kraus = mutate_kraus(
                    candidate_kraus, rng, kraus_scale_t * jump_factor
                )

        candidate = evaluate_case(candidate_kraus, candidate_sigma, candidate_x)

        old_obj = current["delta_km"]
        new_obj = candidate["delta_km"]

        accept = False
        if new_obj < old_obj:
            accept = True
            n_improve += 1
        elif use_annealing and temperature > 0.0:
            prob = np.exp(-(new_obj - old_obj) / temperature)
            if rng.random() < prob:
                accept = True

        window_total += 1

        if accept:
            current = candidate
            n_accept += 1
            window_accept += 1

        if current["delta_km"] < best["delta_km"]:
            best = copy.deepcopy(current)
            stagnation_counter = 0
            last_improvement_step = step
        else:
            stagnation_counter += 1

        if stagnation_counter >= restart_after:
            candidate_sigma = mutate_density_matrix(
                best["sigma"], rng, 3.0 * sigma_scale_t
            )
            candidate_x = mutate_density_matrix(
                best["x"], rng, 3.0 * x_scale_t
            )
            candidate_kraus = mutate_kraus(
                best["kraus"], rng, 3.0 * kraus_scale_t
            )
            current = evaluate_case(candidate_kraus, candidate_sigma, candidate_x)
            stagnation_counter = 0

        if print_every > 0 and (step % print_every == 0 or step == 1):
            accept_rate = window_accept / max(window_total, 1)
            print(
                f"step={step:8d} | "
                f"current Delta_KM={current['delta_km']:+.12f} | "
                f"best Delta_KM={best['delta_km']:+.12f} | "
                f"T={temperature:.3e} | "
                f"acc_rate={accept_rate:.3f} | "
                f"since_improve={step - last_improvement_step}"
            )
            window_accept = 0
            window_total = 0

    print("\nRefinement finished.")
    print(f"Accepted moves: {n_accept}/{n_steps}")
    print(f"Improving moves: {n_improve}/{n_steps}")
    print_result_summary("Refined best result", best)

    return best


def multi_start_refinement(
    best_result: dict,
    n_runs: int = 5,
    base_seed: int = 987654321,
    n_steps: int = 200000,
    print_each_run: bool = True,
    **refine_kwargs,
) -> dict:
    """
    Run several independent refinement runs and keep the best one.
    """
    best_overall = None

    for run_idx in range(n_runs):
        seed = base_seed + run_idx

        if print_each_run:
            print("\n" + "#" * 80)
            print(f"REFINEMENT RUN {run_idx + 1}/{n_runs}  (seed={seed})")
            print("#" * 80)

        candidate = refine_best_result(
            best_result=best_result,
            n_steps=n_steps,
            seed=seed,
            **refine_kwargs,
        )

        if best_overall is None or candidate["delta_km"] < best_overall["delta_km"]:
            best_overall = copy.deepcopy(candidate)

    print("\n" + "=" * 80)
    print("BEST OVER ALL REFINEMENT RUNS")
    print("=" * 80)
    print_result_summary("Best overall refined result", best_overall)

    return best_overall


# -----------------------------------------------------------------------------
# Main random search
# -----------------------------------------------------------------------------

def main():
    N = 10**4
    d = 2
    n_kraus = 2
    seed = 12343

    n_jobs = -1
    chunk_size = 50_000

    def worker(start_seed, n_trials):
        n_km_increase = 0

        sum_km = 0.0
        min_km = np.inf
        max_km = -np.inf

        best_example = None
        best_delta_km = np.inf

        for i in range(n_trials):
            result = run_single_trial(
                seed=start_seed + i,
                d=d,
                n_kraus=n_kraus,
                verbose=False,
            )

            delta_km = result["delta_km"]

            sum_km += delta_km
            min_km = min(min_km, delta_km)
            max_km = max(max_km, delta_km)

            if delta_km > 0.0:
                n_km_increase += 1

            if delta_km < best_delta_km:
                best_delta_km = delta_km
                best_example = result

        return (
            n_km_increase,
            sum_km,
            min_km,
            max_km,
            best_delta_km,
            best_example,
        )

    chunks = []
    for start in range(0, N, chunk_size):
        size = min(chunk_size, N - start)
        chunks.append((seed + start, size))

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
    )(
        delayed(worker)(start_seed, size)
        for (start_seed, size) in chunks
    )

    n_km_increase = 0
    sum_km = 0.0
    min_km = np.inf
    max_km = -np.inf

    best_example = None
    best_delta_km = np.inf

    for r in results:
        (
            n_km_i,
            s_km,
            min_k,
            max_k,
            chunk_best_delta_km,
            example,
        ) = r

        n_km_increase += n_km_i
        sum_km += s_km

        min_km = min(min_km, min_k)
        max_km = max(max_km, max_k)

        if chunk_best_delta_km < best_delta_km:
            best_delta_km = chunk_best_delta_km
            best_example = example

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of trials = {N}")
    print(f"KM increases     = {n_km_increase}")
    print()
    print(f"Mean Delta_KM = {sum_km / N}")
    print(f"Min  Delta_KM = {min_km}")
    print(f"Max  Delta_KM = {max_km}")
    print()

    if best_example is not None:
        print("Best example from random search:")
        print("Delta_KM   =", best_example["delta_km"])
        print("||[X,Y]||_F =", best_example["comm_norm"])
        print(f"Kraus:\n{best_example['kraus']}")
        print(f"X:\n{best_example['x']}")
        print(f"Sigma:\n{best_example['sigma']}")

    return best_example


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    best_result = main()

    if best_result is not None:
        # broad multi-start refinement
        refined_best = multi_start_refinement(
            best_result=best_result,
            n_runs=5,
            base_seed=987654321,
            n_steps=200000,
            sigma_scale_start=0.05,
            sigma_scale_end=0.005,
            x_scale_start=0.05,
            x_scale_end=0.005,
            kraus_scale_start=0.05,
            kraus_scale_end=0.005,
            p_mutate_sigma=0.35,
            p_mutate_x=0.35,
            p_mutate_kraus=0.20,
            p_mutate_two=0.10,
            use_annealing=True,
            temperature_start=1e-2,
            temperature_end=1e-5,
            restart_after=5000,
            print_every=5000,
        )

        # final polish
        polished_best = refine_best_result(
            best_result=refined_best,
            n_steps=100000,
            seed=987654999,
            sigma_scale_start=0.005,
            sigma_scale_end=0.0005,
            x_scale_start=0.005,
            x_scale_end=0.0005,
            kraus_scale_start=0.005,
            kraus_scale_end=0.0005,
            p_mutate_sigma=0.40,
            p_mutate_x=0.40,
            p_mutate_kraus=0.15,
            p_mutate_two=0.05,
            use_annealing=False,
            temperature_start=0.0,
            temperature_end=0.0,
            restart_after=10000,
            print_every=5000,
        )

        print("\n" + "=" * 80)
        print("FINAL BEST EXAMPLE")
        print("=" * 80)
        print("Delta_KM   =", polished_best["delta_km"])
        print("||[X,Y]||_F =", polished_best["comm_norm"])
        print("CPTP defect =", polished_best["cptp_defect"])
        print(f"Kraus:\n{polished_best['kraus']}")
        print(f"X:\n{polished_best['x']}")
        print(f"Sigma:\n{polished_best['sigma']}")
        print(f"Y:\n{polished_best['y']}")
        print(f"Sigma_KM:\n{polished_best['sigma_km']}")
        print(f"Y_KM:\n{polished_best['y_km']}")
    else:
        print("No best result to refine.")