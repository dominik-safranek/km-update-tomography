import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter


# just for the initial state regularization
EPS = 1e-2  # state regularization
EPS_floor = 0  # EPS / 10000   # EPS / 10000     # Petz update stabilization. Set 0 for not stabilized evolution
EPSP = 1e-16    # approximate zero for expressions of type p_i log p_i


# =============================================================================
# Basic helpers
# =============================================================================

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


def normalize_density_matrix(rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    rho = hermitian_part(rho)
    tr = np.real(np.trace(rho))
    if tr < eps:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    return rho / tr


def random_complex_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))


def random_density_matrix(
    d: int,
    rng: np.random.Generator,
    eps: float = EPS,
) -> np.ndarray:
    """
    Mixed sampler:
      - sometimes pure
      - sometimes low-rank
      - sometimes full-rank
    """
    u = rng.random()

    if u < 0.2:
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
    else:
        if u < 0.5:
            k = min(d, int(rng.choice([2, 2, 2, d])))
        elif u < 0.85:
            k = max(2, d)
        else:
            k = d

        a = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
        rho = a @ a.conj().T
        rho = normalize_density_matrix(rho, eps=eps)

    rho = (1.0 - eps) * rho + ( eps / d ) * np.eye(d, dtype=complex)
    return normalize_density_matrix(rho, eps=eps)


def random_unitary_haar(d: int, rng: np.random.Generator) -> np.ndarray:
    z = random_complex_matrix(d, rng)
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = phases / np.maximum(np.abs(phases), 1e-15)
    return q @ np.diag(np.conj(phases))


# =============================================================================
# Fidelity
# =============================================================================

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


# =============================================================================
# Qubit basis unitaries / Pauli-basis measurements
# =============================================================================

def qubit_basis_unitary(label: str) -> np.ndarray:
    label = label.upper()

    if label == "Z":
        return np.array(
            [[1.0, 0.0],
             [0.0, 1.0]],
            dtype=complex,
        )

    if label == "X":
        return np.array(
            [[1.0,  1.0],
             [1.0, -1.0]],
            dtype=complex,
        ) / np.sqrt(2.0)

    if label == "Y":
        return np.array(
            [[1.0,  1.0],
             [1.0j, -1.0j]],
            dtype=complex,
        ) / np.sqrt(2.0)

    raise ValueError(f"Unknown basis label: {label}")


def projector(idx: int, d: int = 2) -> np.ndarray:
    e = np.zeros(d, dtype=complex)
    e[idx] = 1.0
    return np.outer(e, e.conj())


def pauli_measurement_projectors(label: str) -> list[np.ndarray]:
    U = qubit_basis_unitary(label)
    return [U @ projector(0) @ U.conj().T, U @ projector(1) @ U.conj().T]


# =============================================================================
# Stinespring isometry from a 2-qubit global unitary
# =============================================================================

def stinespring_isometry_from_unitary(
    U: np.ndarray,
    env_init_index: int = 0,
) -> np.ndarray:
    """
    Input is 1 qubit (dim 2), environment is 1 qubit (dim 2), global output dim 4.
    V |psi> = U (|psi> \\otimes |env_init>)
    """
    if U.shape != (4, 4):
        raise ValueError("U must be 4x4 for a 2-qubit global unitary.")

    V = np.zeros((4, 2), dtype=complex)

    for a in range(2):
        basis = np.zeros(4, dtype=complex)
        basis[2 * a + env_init_index] = 1.0
        V[:, a] = U @ basis

    err = np.linalg.norm(V.conj().T @ V - np.eye(2), ord="fro")
    if err > 1e-8:
        raise ValueError(f"Constructed V is not an isometry: error={err:.3e}")

    return V


# =============================================================================
# Partial traces and outputs
# =============================================================================

def joint_output(V: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return hermitian_part(V @ rho @ V.conj().T)


def partial_trace_env(rho_se: np.ndarray) -> np.ndarray:
    rho4 = rho_se.reshape(2, 2, 2, 2)
    out = np.zeros((2, 2), dtype=complex)
    for e in range(2):
        out += rho4[:, e, :, e]
    return hermitian_part(out)


def partial_trace_sys(rho_se: np.ndarray) -> np.ndarray:
    rho4 = rho_se.reshape(2, 2, 2, 2)
    out = np.zeros((2, 2), dtype=complex)
    for s in range(2):
        out += rho4[s, :, s, :]
    return hermitian_part(out)


def system_output(V: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return partial_trace_env(joint_output(V, rho))


def environment_output(V: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return partial_trace_sys(joint_output(V, rho))


def correlation_tensor_from_joint(rho_se: np.ndarray) -> np.ndarray:
    """
    Returns the 3x3 matrix T_ab = Tr[(sigma_a \\otimes sigma_b) rho_se]
    in the order X,Y,Z.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [X, Y, Z]

    T = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            T[a, b] = np.real(np.trace(np.kron(paulis[a], paulis[b]) @ rho_se))
    return T


# =============================================================================
# Kubo-Mori inverse
# =============================================================================

def omega_inverse(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Omega_Y^{-1}(X), general matrix version.
    """
    vals, vecs = safe_eigh(Y)

    Xb = vecs.conj().T @ X @ vecs
    out = np.zeros_like(Xb, dtype=complex)

    n = len(vals)
    for i in range(n):
        for j in range(n):
            if abs(vals[i] - vals[j]) < 1e-14:
                m_ij = vals[i]
            else:
                m_ij = (vals[i] - vals[j]) / (np.log(vals[i]) - np.log(vals[j]))
            out[i, j] = Xb[i, j] / m_ij

    return hermitian_part(vecs @ out @ vecs.conj().T)


# =============================================================================
# Channels
# =============================================================================

def lift_system_operator(A_sys: np.ndarray) -> np.ndarray:
    return np.kron(A_sys, np.eye(2, dtype=complex))


def lift_environment_operator(A_env: np.ndarray) -> np.ndarray:
    return np.kron(np.eye(2, dtype=complex), A_env)


def build_system_channel(V: np.ndarray) -> dict:
    def forward(sigma: np.ndarray) -> np.ndarray:
        return system_output(V, sigma)

    def adjoint(A_out: np.ndarray) -> np.ndarray:
        return hermitian_part(V.conj().T @ lift_system_operator(A_out) @ V)

    return {"name": "system", "forward": forward, "adjoint": adjoint}


def build_environment_channel(V: np.ndarray) -> dict:
    def forward(sigma: np.ndarray) -> np.ndarray:
        return environment_output(V, sigma)

    def adjoint(A_out: np.ndarray) -> np.ndarray:
        return hermitian_part(V.conj().T @ lift_environment_operator(A_out) @ V)

    return {"name": "environment", "forward": forward, "adjoint": adjoint}


def build_correlation_pauli_channel(V: np.ndarray) -> dict:
    """
    One classical channel built from all 9 local Pauli settings on the 2-qubit output.
    Output space is 36-dimensional classical:
        9 settings x 4 outcomes,
    encoded as a diagonal 36x36 density matrix.

    Each effect is
        A_{s,o} = (1/9) V^\\dagger M_{s,o} V
    so the total channel is CPTP.
    """
    labels = ["X", "Y", "Z"]
    effects = []
    setting_info = []

    for a in labels:
        P_a = pauli_measurement_projectors(a)
        for b in labels:
            P_b = pauli_measurement_projectors(b)

            for oa in range(2):
                for ob in range(2):
                    M_out = np.kron(P_a[oa], P_b[ob])
                    A_in = (1.0 / 9.0) * hermitian_part(V.conj().T @ M_out @ V)
                    effects.append(A_in)
                    setting_info.append((a, b, oa, ob))

    total = np.zeros((2, 2), dtype=complex)
    for A in effects:
        total += A
    err = np.linalg.norm(total - np.eye(2), ord="fro")
    if err > 1e-8:
        raise ValueError(f"Correlation channel effects do not sum to I: error={err:.3e}")

    def forward(sigma: np.ndarray) -> np.ndarray:
        probs = np.array([np.real(np.trace(A @ sigma)) for A in effects], dtype=float)
        probs = np.maximum(probs, 0.0)
        probs /= np.sum(probs)
        return np.diag(probs)

    def adjoint(A_out: np.ndarray) -> np.ndarray:
        diag_vals = np.real(np.diag(A_out))
        out = np.zeros((2, 2), dtype=complex)
        for coeff, A in zip(diag_vals, effects):
            out += coeff * A
        return hermitian_part(out)

    return {
        "name": "correlation_pauli",
        "forward": forward,
        "adjoint": adjoint,
        "effects": effects,
        "setting_info": setting_info,
    }


# =============================================================================
# KM update
# =============================================================================

def km_update_weighted_channels(
    sigma: np.ndarray,
    sqrt_sigma: np.ndarray,
    channels: list[dict],
    weights: np.ndarray,
    X_targets: list[np.ndarray],
    eps: float = EPS,
    eig_floor: float = EPS_floor,
) -> np.ndarray:
    sigma_plus = np.zeros_like(sigma, dtype=complex)

    for w, ch, X in zip(weights, channels, X_targets):
        Y = ch["forward"](sigma)
        O = omega_inverse(Y, X)
        pulled = ch["adjoint"](O)
        sigma_plus += w * (sqrt_sigma @ pulled @ sqrt_sigma)

    d = sigma_plus.shape[0]
    sigma_plus = (1.0 - eig_floor) * sigma_plus + (eig_floor / d) * np.eye(d, dtype=complex)

    return normalize_density_matrix(sigma_plus, eps=eps)


# =============================================================================
# Diagnostics
# =============================================================================

def evaluate_current_state(
    sigma: np.ndarray,
    rho_true: np.ndarray,
    V: np.ndarray,
    corr_channel: dict,
    eps: float = EPS,
) -> dict:
    sqrt_rho = matrix_sqrt(rho_true)

    sigma_se = joint_output(V, sigma)
    rho_se = joint_output(V, rho_true)

    sigma_s = partial_trace_env(sigma_se)
    rho_s = partial_trace_env(rho_se)

    sigma_e = partial_trace_sys(sigma_se)
    rho_e = partial_trace_sys(rho_se)

    Xc_true = corr_channel["forward"](rho_true)
    Xc_sigma = corr_channel["forward"](sigma)

    p_true = np.real(np.diag(Xc_true))
    p_sigma = np.real(np.diag(Xc_sigma))

    return {
        "infidelity": infidelity_from_sqrt_rho(sqrt_rho, sigma),
        "system_fro": float(np.linalg.norm(sigma_s - rho_s, ord="fro")),
        "environment_fro": float(np.linalg.norm(sigma_e - rho_e, ord="fro")),
        "correlation_l1": float(np.sum(np.abs(p_sigma - p_true))),
        "correlation_max": float(np.max(np.abs(p_sigma - p_true))),
        "corr_tensor_fro": float(
            np.linalg.norm(
                correlation_tensor_from_joint(sigma_se) - correlation_tensor_from_joint(rho_se),
                ord="fro"
            )
        ),
    }


# =============================================================================
# Main iteration driver
# =============================================================================

def run_single_experiment(
    seed: int = 1234,
    n_steps: int = 200,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    verbose: bool = True,
    eps: float = EPS,
):
    rng = np.random.default_rng(seed)

    rho_true = random_density_matrix(2, rng, eps=eps)
    U = random_unitary_haar(4, rng)
    V = stinespring_isometry_from_unitary(U, env_init_index=0)

    ch_sys = build_system_channel(V)
    ch_env = build_environment_channel(V)
    ch_corr = build_correlation_pauli_channel(V)
    channels = [ch_sys, ch_env, ch_corr]

    w = np.array(weights, dtype=float)
    w /= np.sum(w)

    X_targets = [ch["forward"](rho_true) for ch in channels]

    rho_se = joint_output(V, rho_true)
    rho_s = partial_trace_env(rho_se)
    rho_e = partial_trace_sys(rho_se)
    corr_true = correlation_tensor_from_joint(rho_se)

    sigma = np.eye(2, dtype=complex) / 2.0

    history = {
        "infidelity": [],
        "system_fro": [],
        "environment_fro": [],
        "correlation_l1": [],
        "correlation_max": [],
        "corr_tensor_fro": [],
    }

    for step in range(n_steps + 1):
        evals = evaluate_current_state(
            sigma=sigma,
            rho_true=rho_true,
            V=V,
            corr_channel=ch_corr,
            eps=eps,
        )

        for k, v in evals.items():
            history[k].append(v)

        if verbose:
            print(
                f"step={step:4d} | "
                f"infid={evals['infidelity']:.12e} | "
                f"sys_fro={evals['system_fro']:.12e} | "
                f"env_fro={evals['environment_fro']:.12e} | "
                f"corr_L1={evals['correlation_l1']:.12e} | "
                f"corr_tensor_fro={evals['corr_tensor_fro']:.12e}"
            )

        if step == n_steps:
            break

        vals_sigma, vecs_sigma = safe_eigh(sigma)
        sqrt_sigma = matrix_sqrt_from_eigh(vals_sigma, vecs_sigma)

        sigma = km_update_weighted_channels(
            sigma=sigma,
            sqrt_sigma=sqrt_sigma,
            channels=channels,
            weights=w,
            X_targets=X_targets,
            eps=eps,
        )

    results = {
        "seed": seed,
        "rho_true": rho_true,
        "U": U,
        "V": V,
        "rho_sys_true": rho_s,
        "rho_env_true": rho_e,
        "corr_tensor_true": corr_true,
        "sigma_final": sigma,
        "history": history,
        "weights": w,
    }
    return results


# =============================================================================
# Batch runs in compact infidelity format
# =============================================================================

def run_single_case_infidelity(
    seed: int,
    n_steps: int,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = EPS,
) -> dict:
    res = run_single_experiment(
        seed=seed,
        n_steps=n_steps,
        weights=weights,
        verbose=False,
        eps=eps,
    )

    infidelity = np.array(res["history"]["infidelity"], dtype=float)
    iters = np.arange(len(infidelity), dtype=int)

    return {
        "seed": seed,
        "infidelity": infidelity,
        "iters": iters,
    }


def run_many_cases_infidelity(
    n_cases: int = 50,
    base_seed: int = 1234,
    n_steps: int = 150,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[dict]:
    results = []

    for i in range(n_cases):
        seed = base_seed + i

        result = run_single_case_infidelity(
            seed=seed,
            n_steps=n_steps,
            weights=weights,
        )

        results.append(result)

        print(
            f"case {i+1:3d}/{n_cases} | "
            f"seed={seed} | "
            f"final infid={result['infidelity'][-1]:.6e}"
        )

    return results


def save_infidelity_results(results: list[dict], filename: str = "infidelity_runs.npz") -> None:
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


def load_infidelity_results(filename: str = "infidelity_runs.npz") -> list[dict]:
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


# =============================================================================
# Plot
# =============================================================================

def plot_all_infidelity(
    results: list[dict],
    logscale: bool = True,
    xmax: int = 1000,
    ymax: float = 1e-0,
    ymin: float = 1e-16,
) -> None:
    plt.figure(figsize=(7, 4))

    for r in results:
        x = r["iters"]
        y = r["infidelity"]

        if logscale:
            y = np.maximum(y, ymin)
            plt.semilogy(x, y, alpha=0.35)
        else:
            plt.plot(x, y, alpha=0.35)

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.title("Infidelity trajectories")

    plt.xlim(0, xmax)
    if logscale:
        plt.ylim(ymin, ymax)
        ax = plt.gca()
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        plt.ylim(0, ymax)

    plt.grid(True, which="both" if logscale else "major")
    plt.tight_layout()
    plt.show()


def plot_mean_infidelity(
    results: list[dict],
    logscale: bool = True,
    xmax: int = 500,
    ymax: float = 1e-0,
    ymin: float = 1e-4,
) -> None:
    iters = results[0]["iters"]
    Y = np.array([r["infidelity"] for r in results], dtype=float)

    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)

    plt.figure(figsize=(7, 4))
    if logscale:
        mean_plot = np.maximum(mean, ymin)
        lower = np.maximum(mean - std, ymin)
        upper = np.maximum(mean + std, ymin)
        plt.semilogy(iters, mean_plot, label="mean")
        plt.fill_between(iters, lower, upper, alpha=0.2, label="std")
    else:
        plt.plot(iters, mean, label="mean")
        plt.fill_between(iters, mean - std, mean + std, alpha=0.2, label="std")

    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.title("Mean infidelity")
    plt.legend()

    plt.xlim(0, xmax)
    if logscale:
        plt.ylim(ymin, ymax)
        ax = plt.gca()
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        plt.ylim(0, ymax)

    plt.grid(True, which="both" if logscale else "major")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    n_cases = 20
    n_steps = 500
    output_file = "infidelity_runs_q1_KM20cases.npz"

    # Optional single run with detailed output
    result = run_single_experiment(
        seed=1234,
        n_steps=n_steps,
        weights=(1.0, 1.0, 1.0),
        verbose=True,
    )

    print("\nTrue input state rho:")
    print(result["rho_true"])

    print("\nFinal estimate sigma:")
    print(result["sigma_final"])

    print("\nTrue system marginal:")
    print(result["rho_sys_true"])

    print("\nTrue environment marginal:")
    print(result["rho_env_true"])

    print("\nTrue 3x3 Pauli correlation tensor:")
    print(result["corr_tensor_true"])

    # Batch runs in the same compact format as your old file
    results = run_many_cases_infidelity(
        n_cases=n_cases,
        base_seed=1234,
        n_steps=n_steps,
        weights=(1.0, 1.0, 1.0),
    )

    save_infidelity_results(results, filename=output_file)

    plot_all_infidelity(results, logscale=True, xmax=500, ymax=1e-0)
    plot_mean_infidelity(results, logscale=True, xmax=500, ymax=1e-0)