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


def matrix_log_from_eig(vals: np.ndarray, vecs: np.ndarray, eps: float = EPS) -> np.ndarray:
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.conj().T


def matrix_sqrt(a: np.ndarray, eps: float = EPS) -> np.ndarray:
    vals, vecs = safe_eigh(a, eps=eps)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T


def matrix_inv_sqrt_from_eig(vals: np.ndarray, vecs: np.ndarray, eps: float = EPS) -> np.ndarray:
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T


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
    # M = sum_a L_a^\dagger L_a
    m = np.einsum("aji,ajk->ik", raw.conj(), raw)
    vals, vecs = safe_eigh(m)
    m_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T
    # K_a = L_a M^{-1/2}
    kraus = np.einsum("aij,jk->aik", raw, m_inv_sqrt)
    return kraus


# -----------------------------------------------------------------------------
# Channel-dependent operations
# -----------------------------------------------------------------------------

def channel(rho: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    # (K rho K^\dag)_{il} = sum_{a,j,k} K_{aij} rho_{jk} conj(K_{alk})
    return np.einsum("aij,jk,alk->il", kraus, rho, kraus.conj())


def channel_adjoint(a: np.ndarray, kraus: np.ndarray) -> np.ndarray:
    # (K^\dag a K)_{il} = sum_{a,j,k} conj(K_{aji}) a_{jk} K_{akl}
    return np.einsum("aji,jk,akl->il", kraus.conj(), a, kraus)


def log_likelihood_from_y(x: np.ndarray, y: np.ndarray) -> float:
    vals_y, vecs_y = safe_eigh(y)
    return float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))


def log_likelihood(sigma: np.ndarray, x: np.ndarray, kraus: np.ndarray) -> float:
    y = channel(sigma, kraus)
    return log_likelihood_from_y(x, y)


# -----------------------------------------------------------------------------
# Petz score and exact KM score S_Y = Omega_Y^{-1}
# -----------------------------------------------------------------------------

def petz_score_from_eig(vals_y: np.ndarray, vecs_y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y_inv_sqrt = matrix_inv_sqrt_from_eig(vals_y, vecs_y)
    return y_inv_sqrt @ x @ y_inv_sqrt


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

    # Initialize with diagonal-limit value m_ii = lambda_i
    m = lam_i * np.ones_like(diff)

    # Off-diagonal entries use the logarithmic mean formula
    mask = np.abs(diff) >= 1e-14
    m[mask] = diff[mask] / log_diff[mask]

    m = np.maximum(m, eps)
    out = b / m
    return vecs @ out @ vecs.conj().T


# -----------------------------------------------------------------------------
# Updates
# -----------------------------------------------------------------------------

def petz_update(
    sigma: np.ndarray,
    x: np.ndarray,
    kraus: np.ndarray,
    vals_y: np.ndarray | None = None,
    vecs_y: np.ndarray | None = None,
) -> np.ndarray:
    if vals_y is None or vecs_y is None:
        y = channel(sigma, kraus)
        vals_y, vecs_y = safe_eigh(y)

    g = petz_score_from_eig(vals_y, vecs_y, x)
    s = matrix_sqrt(sigma)
    sigma_plus = s @ channel_adjoint(g, kraus) @ s
    sigma_plus = hermitian_part(sigma_plus)
    return normalize_density_matrix(sigma_plus)


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

    sigma_petz = petz_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)
    sigma_km = km_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)

    y_petz = channel(sigma_petz, kraus)
    y_km = channel(sigma_km, kraus)

    ll_sigma = float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))
    ll_petz = log_likelihood_from_y(x, y_petz)
    ll_km = log_likelihood_from_y(x, y_km)

    delta_petz = ll_petz - ll_sigma
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
        "sigma_petz": sigma_petz,
        "sigma_km": sigma_km,
        "y_petz": y_petz,
        "y_km": y_km,
        "ll_sigma": ll_sigma,
        "ll_petz": ll_petz,
        "ll_km": ll_km,
        "delta_petz": delta_petz,
        "delta_km": delta_km,
        "cptp_defect": cptp_defect,
        "comm_norm": frobenius_commutator_norm(x, y),
        "eig_sigma": np.linalg.eigvalsh(hermitian_part(sigma)),
        "eig_x": np.linalg.eigvalsh(hermitian_part(x)),
        "eig_y": np.linalg.eigvalsh(hermitian_part(y)),
        "eig_sigma_petz": np.linalg.eigvalsh(hermitian_part(sigma_petz)),
        "eig_sigma_km": np.linalg.eigvalsh(hermitian_part(sigma_km)),
    }

    if verbose:
        print("=== CPTP check ===")
        print("|| sum_a K_a^dagger K_a - I ||_F =", cptp_defect)
        print()

        format_matrix("sigma", sigma)
        format_matrix("X", x)
        format_matrix("Y = E(sigma)", y)
        format_matrix("sigma_petz", sigma_petz)
        format_matrix("sigma_km", sigma_km)
        format_matrix("Y_petz = E(sigma_petz)", y_petz)
        format_matrix("Y_km = E(sigma_km)", y_km)

        print("eig(sigma)      =", result["eig_sigma"])
        print("eig(X)          =", result["eig_x"])
        print("eig(Y)          =", result["eig_y"])
        print("eig(sigma_petz) =", result["eig_sigma_petz"])
        print("eig(sigma_km)   =", result["eig_sigma_km"])
        print()

        print("|| [X, Y] ||_F =", result["comm_norm"])
        print()

        print("L_X(sigma)      =", ll_sigma)
        print("L_X(sigma_petz) =", ll_petz)
        print("L_X(sigma_km)   =", ll_km)
        print()

        print("Delta_Petz =", delta_petz)
        print("Delta_KM   =", delta_km)
        print()

        if delta_petz < 0.0:
            print("Petz decreases the log-likelihood in this trial.")
        else:
            print("Petz does not decrease the log-likelihood in this trial.")

        if delta_km > 0.0:
            print("The exact KM update increases the log-likelihood in this trial.")
        else:
            print("The exact KM update does not increase the log-likelihood in this trial.")
        print("-" * 80)

    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    N = 10**5   # change to 10**8 for better results
    d = 6
    n_kraus = 2
    seed = 12347

    n_jobs = -1
    chunk_size = 50_000

    def worker(start_seed, n_trials):
        n_petz_decrease = 0
        n_km_increase = 0
        n_both = 0

        sum_petz = 0.0
        sum_km = 0.0
        min_petz = np.inf
        max_petz = -np.inf
        min_km = np.inf
        max_km = -np.inf

        best_example = None
        best_delta_petz = np.inf

        for i in range(n_trials):
            result = run_single_trial(
                seed=start_seed + i,
                d=d,
                n_kraus=n_kraus,
                verbose=False,
            )

            delta_petz = result["delta_petz"]
            delta_km = result["delta_km"]

            sum_petz += delta_petz
            sum_km += delta_km

            min_petz = min(min_petz, delta_petz)
            max_petz = max(max_petz, delta_petz)
            min_km = min(min_km, delta_km)
            max_km = max(max_km, delta_km)

            petz_decreases = delta_petz < 0.0
            km_increases = delta_km > 0.0

            if petz_decreases:
                n_petz_decrease += 1
            if km_increases:
                n_km_increase += 1
            if petz_decreases and km_increases:
                n_both += 1

            if delta_petz < best_delta_petz:
                best_delta_petz = delta_petz
                best_example = result

        return (
            n_petz_decrease,
            n_km_increase,
            n_both,
            sum_petz,
            sum_km,
            min_petz,
            max_petz,
            min_km,
            max_km,
            best_delta_petz,
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

    n_petz_decrease = 0
    n_km_increase = 0
    n_both = 0

    sum_petz = 0.0
    sum_km = 0.0
    min_petz = np.inf
    max_petz = -np.inf
    min_km = np.inf
    max_km = -np.inf

    best_example = None
    best_delta_petz = np.inf

    for r in results:
        (
            n_petz_d,
            n_km_i,
            n_b,
            s_petz,
            s_km,
            min_p,
            max_p,
            min_k,
            max_k,
            chunk_best_delta_petz,
            example,
        ) = r

        n_petz_decrease += n_petz_d
        n_km_increase += n_km_i
        n_both += n_b

        sum_petz += s_petz
        sum_km += s_km

        min_petz = min(min_petz, min_p)
        max_petz = max(max_petz, max_p)
        min_km = min(min_km, min_k)
        max_km = max(max_km, max_k)

        if chunk_best_delta_petz < best_delta_petz:
            best_delta_petz = chunk_best_delta_petz
            best_example = example

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of trials               = {N}")
    print(f"Petz decreases                 = {n_petz_decrease}")
    print(f"KM increases                   = {n_km_increase}")
    print(f"Petz decreases & KM increases  = {n_both}")
    print()
    print(f"Mean Delta_Petz = {sum_petz / N}")
    print(f"Mean Delta_KM   = {sum_km / N}")
    print(f"Min  Delta_Petz = {min_petz}")
    print(f"Max  Delta_Petz = {max_petz}")
    print(f"Min  Delta_KM   = {min_km}")
    print(f"Max  Delta_KM   = {max_km}")
    print()

    if best_example is not None:
        print("Best example:")
        print("Delta_Petz =", best_example["delta_petz"])
        print("Delta_KM   =", best_example["delta_km"])
        print("||[X,Y]||_F =", best_example["comm_norm"])
        print(f"Kraus:\n{best_example['kraus']}")
        print(f"X:\n{best_example['x']}")
        print(f"Sigma:\n{best_example['sigma']}")

    return best_example

    # -------------------------------------------------------------------------
    # Create chunks
    # -------------------------------------------------------------------------
    chunks = []
    for start in range(0, N, chunk_size):
        size = min(chunk_size, N - start)
        chunks.append((seed + start, size))

    # -------------------------------------------------------------------------
    # Run in parallel (TRUE multiprocessing)
    # -------------------------------------------------------------------------
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
    )(
        delayed(worker)(start_seed, size)
        for (start_seed, size) in chunks
    )

    # -------------------------------------------------------------------------
    # Reduce results
    # -------------------------------------------------------------------------
    n_petz_decrease = 0
    n_km_increase = 0
    n_both = 0

    sum_petz = 0.0
    sum_km = 0.0
    min_petz = np.inf
    max_petz = -np.inf
    min_km = np.inf
    max_km = -np.inf

    best_example = None
    best_score = -np.inf

    for r in results:
        (
            n_petz_d,
            n_km_i,
            n_b,
            s_petz,
            s_km,
            min_p,
            max_p,
            min_k,
            max_k,
            score,
            example,
        ) = r

        n_petz_decrease += n_petz_d
        n_km_increase += n_km_i
        n_both += n_b

        sum_petz += s_petz
        sum_km += s_km

        min_petz = min(min_petz, min_p)
        max_petz = max(max_petz, max_p)
        min_km = min(min_km, min_k)
        max_km = max(max_km, max_k)

        if score > best_score:
            best_score = score
            best_example = example

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of trials               = {N}")
    print(f"Petz decreases                 = {n_petz_decrease}")
    print(f"KM increases                   = {n_km_increase}")
    print(f"Petz decreases & KM increases  = {n_both}")
    print()
    print(f"Mean Delta_Petz = {sum_petz / N}")
    print(f"Mean Delta_KM   = {sum_km / N}")
    print(f"Min  Delta_Petz = {min_petz}")
    print(f"Max  Delta_Petz = {max_petz}")
    print(f"Min  Delta_KM   = {min_km}")
    print(f"Max  Delta_KM   = {max_km}")
    print()

    if best_example is not None:
        print("Best example:")
        print("Delta_Petz =", best_example["delta_petz"])
        print("Delta_KM   =", best_example["delta_km"])
        print("||[X,Y]||_F =", best_example["comm_norm"])
        print(f"Kraus:\n{best_example['kraus']}")
        print(f"X:\n{best_example['x']}")
        print(f"Sigma:\n{best_example['sigma']}")

    return best_example


import copy
import numpy as np


# -----------------------------------------------------------------------------
# Utilities for local refinement
# -----------------------------------------------------------------------------

def density_from_generator(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    rho = a @ a.conj().T + eps * np.eye(a.shape[0], dtype=complex)
    return normalize_density_matrix(rho)


def kraus_from_raw(raw: np.ndarray) -> np.ndarray:
    """
    raw shape: (n_kraus, d, d)
    Returns CPTP-normalized Kraus operators K_a = L_a M^{-1/2},
    where M = sum_a L_a^\dagger L_a.
    """
    m = np.einsum("aji,ajk->ik", raw.conj(), raw)
    vals, vecs = safe_eigh(m)
    m_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T
    return np.einsum("aij,jk->aik", raw, m_inv_sqrt)


def evaluate_case(kraus: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> dict:
    y = channel(sigma, kraus)
    vals_y, vecs_y = safe_eigh(y)

    sigma_petz = petz_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)
    sigma_km = km_update(sigma, x, kraus, vals_y=vals_y, vecs_y=vecs_y)

    y_petz = channel(sigma_petz, kraus)
    y_km = channel(sigma_km, kraus)

    ll_sigma = float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))
    ll_petz = log_likelihood_from_y(x, y_petz)
    ll_km = log_likelihood_from_y(x, y_km)

    delta_petz = ll_petz - ll_sigma
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
        "sigma_petz": sigma_petz,
        "sigma_km": sigma_km,
        "y_petz": y_petz,
        "y_km": y_km,
        "ll_sigma": ll_sigma,
        "ll_petz": ll_petz,
        "ll_km": ll_km,
        "delta_petz": delta_petz,
        "delta_km": delta_km,
        "cptp_defect": cptp_defect,
        "comm_norm": frobenius_commutator_norm(x, y),
        "eig_sigma": np.linalg.eigvalsh(hermitian_part(sigma)),
        "eig_x": np.linalg.eigvalsh(hermitian_part(x)),
        "eig_y": np.linalg.eigvalsh(hermitian_part(y)),
        "eig_sigma_petz": np.linalg.eigvalsh(hermitian_part(sigma_petz)),
        "eig_sigma_km": np.linalg.eigvalsh(hermitian_part(sigma_km)),
    }


def mutate_density_matrix(rho: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    d = rho.shape[0]
    a = matrix_sqrt(rho)
    noise = scale * (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    a_new = a + noise
    return density_from_generator(a_new)


def mutate_kraus(kraus: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    noise = scale * (
        rng.normal(size=kraus.shape) + 1j * rng.normal(size=kraus.shape)
    )
    raw_new = kraus + noise
    return kraus_from_raw(raw_new)


def print_result_summary(label: str, result: dict) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    print("Delta_Petz =", result["delta_petz"])
    print("Delta_KM   =", result["delta_km"])
    print("||[X,Y]||_F =", result["comm_norm"])
    print("CPTP defect =", result["cptp_defect"])
    print("eig(sigma)      =", result["eig_sigma"])
    print("eig(X)          =", result["eig_x"])
    print("eig(Y)          =", result["eig_y"])
    print("eig(sigma_petz) =", result["eig_sigma_petz"])
    print("eig(sigma_km)   =", result["eig_sigma_km"])


def refine_best_result(
    best_result: dict,
    n_steps: int = 20000,
    seed: int = 987654321,
    sigma_scale: float = 0.02,
    x_scale: float = 0.02,
    kraus_scale: float = 0.02,
    p_mutate_sigma: float = 0.4,
    p_mutate_x: float = 0.4,
    p_mutate_kraus: float = 0.2,
    use_annealing: bool = True,
    temperature_start: float = 1e-2,
    temperature_end: float = 1e-5,
    print_every: int = 1000,
) -> dict:
    """
    Local stochastic refinement of the strongest negative Delta_Petz case.

    Objective: minimize delta_petz (make it as negative as possible).
    """
    rng = np.random.default_rng(seed)

    current = evaluate_case(
        kraus=np.array(best_result["kraus"], copy=True),
        sigma=np.array(best_result["sigma"], copy=True),
        x=np.array(best_result["x"], copy=True),
    )
    best = copy.deepcopy(current)

    n_accept = 0
    n_improve = 0

    for step in range(1, n_steps + 1):
        t = (step - 1) / max(1, n_steps - 1)
        if use_annealing:
            temperature = temperature_start * (temperature_end / temperature_start) ** t
        else:
            temperature = 0.0

        candidate_kraus = current["kraus"]
        candidate_sigma = current["sigma"]
        candidate_x = current["x"]

        u = rng.random()
        if u < p_mutate_sigma:
            candidate_sigma = mutate_density_matrix(candidate_sigma, rng, sigma_scale)
        elif u < p_mutate_sigma + p_mutate_x:
            candidate_x = mutate_density_matrix(candidate_x, rng, x_scale)
        else:
            candidate_kraus = mutate_kraus(candidate_kraus, rng, kraus_scale)

        candidate = evaluate_case(candidate_kraus, candidate_sigma, candidate_x)

        old_obj = current["delta_petz"]
        new_obj = candidate["delta_petz"]

        accept = False
        if new_obj < old_obj:
            accept = True
            n_improve += 1
        elif use_annealing and temperature > 0.0:
            # accept some uphill moves
            prob = np.exp(-(new_obj - old_obj) / temperature)
            if rng.random() < prob:
                accept = True

        if accept:
            current = candidate
            n_accept += 1

            if current["delta_petz"] < best["delta_petz"]:
                best = copy.deepcopy(current)

        if print_every > 0 and (step % print_every == 0 or step == 1):
            print(
                f"step={step:7d} | "
                f"current Delta_Petz={current['delta_petz']:+.12f} | "
                f"best Delta_Petz={best['delta_petz']:+.12f} | "
                f"Delta_KM(best)={best['delta_km']:+.12f} | "
                f"accept={n_accept} | improve={n_improve}"
            )

    print("\nRefinement finished.")
    print(f"Accepted moves: {n_accept}/{n_steps}")
    print(f"Improving moves: {n_improve}/{n_steps}")
    print_result_summary("Refined best result", best)

    return best

if __name__ == "__main__":
    best_result = main()

    if best_result is not None:
        refined_best = refine_best_result(best_result)

        print("\n" + "=" * 80)
        print("REFINED BEST EXAMPLE")
        print("=" * 80)
        print("Delta_Petz =", refined_best["delta_petz"])
        print("Delta_KM   =", refined_best["delta_km"])
        print("||[X,Y]||_F =", refined_best["comm_norm"])
        print(f"Kraus:\n{refined_best['kraus']}")
        print(f"X:\n{refined_best['x']}")
        print(f"Sigma:\n{refined_best['sigma']}")
    else:
        print("No best result to refine.")