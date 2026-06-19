import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=12, suppress=False)

EPS = 1e-12

SIGMA = np.array(
    [
        [2/3, 0.0],
        [0.0, 1/3],
    ],
    dtype=complex,
)

X = np.array(
    [
        [0.99, 0.0],
        [0.0, 0.01],
    ],
    dtype=complex,
)

KRAUS_SEED = [
    np.array(
        [
            [-0.231935612603 + 0.460957590667j,  0.335251798321 - 0.066849316442j],
            [ 0.308987578760 - 0.652477352426j, -0.313255130368 - 0.003543001473j],
        ],
        dtype=complex,
    ),
    np.array(
        [
            [-0.143196738930 + 0.195937761953j, -0.577790445576 - 0.135924385817j],
            [ 0.281268103827 - 0.272974899364j,  0.645455974032 + 0.126749427626j],
        ],
        dtype=complex,
    ),
]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def hermitian_part(a):
    return 0.5 * (a + a.conj().T)


def safe_eigh(a, eps=EPS):
    vals, vecs = np.linalg.eigh(hermitian_part(a))
    vals = np.maximum(vals, eps)
    return vals, vecs


def matrix_log_from_eig(vals, vecs):
    vals = np.maximum(vals, EPS)
    return vecs @ np.diag(np.log(vals)) @ vecs.conj().T


def matrix_sqrt(a):
    vals, vecs = safe_eigh(a)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T


def matrix_inv_sqrt(a):
    vals, vecs = safe_eigh(a)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T


def normalize_density_matrix(rho):
    rho = hermitian_part(rho)
    tr = np.real(np.trace(rho))
    if tr < EPS:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    return rho / tr


# -----------------------------------------------------------------------------
# Channel operations
# -----------------------------------------------------------------------------

def channel(rho, kraus):
    return np.einsum("aij,jk,alk->il", kraus, rho, kraus.conj())


def channel_adjoint(a, kraus):
    return np.einsum("aji,jk,akl->il", kraus.conj(), a, kraus)


# -----------------------------------------------------------------------------
# Petz update
# -----------------------------------------------------------------------------

def log_likelihood_from_y(x, y):
    vals_y, vecs_y = safe_eigh(y)
    return float(np.real(np.trace(x @ matrix_log_from_eig(vals_y, vecs_y))))


def petz_update(sigma, x, kraus):
    y = channel(sigma, kraus)
    vals_y, vecs_y = safe_eigh(y)

    y_inv_sqrt = vecs_y @ np.diag(1.0 / np.sqrt(vals_y)) @ vecs_y.conj().T
    g = y_inv_sqrt @ x @ y_inv_sqrt

    s = matrix_sqrt(sigma)
    sigma_plus = s @ channel_adjoint(g, kraus) @ s
    sigma_plus = hermitian_part(sigma_plus)
    return normalize_density_matrix(sigma_plus)


# -----------------------------------------------------------------------------
# Raw-Kraus parametrization
# -----------------------------------------------------------------------------

def kraus_to_vector(kraus):
    """
    kraus shape (2,2,2) complex -> real vector of length 16
    """
    return np.concatenate([kraus.real.ravel(), kraus.imag.ravel()])


def vector_to_raw_kraus(vec):
    """
    real vector of length 16 -> raw complex kraus shape (2,2,2)
    """
    vec = np.asarray(vec, dtype=float)
    n = 8
    real = vec[:n].reshape(2, 2, 2)
    imag = vec[n:].reshape(2, 2, 2)
    return real + 1j * imag


def raw_to_cptp_kraus(raw):
    """
    K_a = L_a M^{-1/2}, where M = sum_a L_a^\dagger L_a
    """
    m = np.einsum("aji,ajk->ik", raw.conj(), raw)
    m_inv_sqrt = matrix_inv_sqrt(m)
    return np.einsum("aij,jk->aik", raw, m_inv_sqrt)


def cptp_defect(kraus):
    return float(
        np.linalg.norm(
            np.einsum("aji,ajk->ik", kraus.conj(), kraus) - np.eye(2),
            ord="fro",
        )
    )


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------

def evaluate_channel_only(kraus, sigma=SIGMA, x=X):
    y = channel(sigma, kraus)
    sigma_petz = petz_update(sigma, x, kraus)
    y_petz = channel(sigma_petz, kraus)

    ll_sigma = log_likelihood_from_y(x, y)
    ll_petz = log_likelihood_from_y(x, y_petz)

    delta_petz = ll_petz - ll_sigma

    return {
        "kraus": kraus,
        "sigma": sigma,
        "x": x,
        "y": y,
        "sigma_petz": sigma_petz,
        "y_petz": y_petz,
        "ll_sigma": ll_sigma,
        "ll_petz": ll_petz,
        "delta_petz": delta_petz,
        "cptp_defect": cptp_defect(kraus),
        "eig_y": np.linalg.eigvalsh(hermitian_part(y)),
        "eig_sigma_petz": np.linalg.eigvalsh(hermitian_part(sigma_petz)),
    }


def objective_raw(vec, sigma=SIGMA, x=X):
    try:
        raw = vector_to_raw_kraus(vec)
        kraus = raw_to_cptp_kraus(raw)
        result = evaluate_channel_only(kraus, sigma=sigma, x=x)
        return result["delta_petz"]
    except Exception:
        return 1e12


# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------

def print_result(label, result):
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print("Delta_Petz =", result["delta_petz"])
    print("CPTP defect =", result["cptp_defect"])
    print("L_X(sigma)  =", result["ll_sigma"])
    print("L_X(Petz)   =", result["ll_petz"])
    print("eig(Y)          =", result["eig_y"])
    print("eig(sigma_petz) =", result["eig_sigma_petz"])
    print()
    print("K0 =")
    print(result["kraus"][0])
    print()
    print("K1 =")
    print(result["kraus"][1])
    print()
    print("Y = E(sigma) =")
    print(result["y"])
    print()
    print("sigma_petz =")
    print(result["sigma_petz"])
    print()
    print("Y_petz = E(sigma_petz) =")
    print(result["y_petz"])


# -----------------------------------------------------------------------------
# Powell optimization
# -----------------------------------------------------------------------------

def run_powell_once(
    kraus_seed,
    sigma=SIGMA,
    x=X,
    maxiter=2000,
    xtol=1e-8,
    ftol=1e-8,
):
    kraus_seed = np.stack(kraus_seed, axis=0)
    x0 = kraus_to_vector(kraus_seed)

    initial_result = evaluate_channel_only(raw_to_cptp_kraus(kraus_seed), sigma=sigma, x=x)
    print_result("INITIAL SEED", initial_result)

    opt = minimize(
        objective_raw,
        x0=x0,
        args=(sigma, x),
        method="Powell",
        options={
            "maxiter": maxiter,
            "xtol": xtol,
            "ftol": ftol,
            "disp": True,
        },
    )

    best_raw = vector_to_raw_kraus(opt.x)
    best_kraus = raw_to_cptp_kraus(best_raw)
    best_result = evaluate_channel_only(best_kraus, sigma=sigma, x=x)

    print("\nOptimization finished.")
    print("success =", opt.success)
    print("message =", opt.message)
    print("objective =", opt.fun)

    print_result("BEST POWELL RESULT", best_result)
    return best_result, opt


# -----------------------------------------------------------------------------
# Multi-start Powell around seed
# -----------------------------------------------------------------------------

def perturb_seed(kraus_seed, rng, scale=0.02):
    kraus_seed = np.stack(kraus_seed, axis=0)
    noise = scale * (
        rng.normal(size=kraus_seed.shape) + 1j * rng.normal(size=kraus_seed.shape)
    )
    raw = kraus_seed + noise
    return raw_to_cptp_kraus(raw)


def run_powell_multiple(
    kraus_seed,
    M=10,
    perturb_scale=0.02,
    sigma=SIGMA,
    x=X,
    maxiter=2000,
    xtol=1e-8,
    ftol=1e-8,
    random_seed=12345,
):
    rng = np.random.default_rng(random_seed)

    overall_best = None
    overall_obj = np.inf

    # include the exact seed as run 1
    seeds = [np.stack(kraus_seed, axis=0)]
    for _ in range(M - 1):
        seeds.append(perturb_seed(kraus_seed, rng, scale=perturb_scale))

    for i, seed_kraus in enumerate(seeds, start=1):
        print("\n" + "#" * 80)
        print(f"POWELL RUN {i} / {M}")
        print("#" * 80)

        x0 = kraus_to_vector(seed_kraus)

        opt = minimize(
            objective_raw,
            x0=x0,
            args=(sigma, x),
            method="Powell",
            options={
                "maxiter": maxiter,
                "xtol": xtol,
                "ftol": ftol,
                "disp": True,
            },
        )

        best_raw = vector_to_raw_kraus(opt.x)
        best_kraus = raw_to_cptp_kraus(best_raw)
        best_result = evaluate_channel_only(best_kraus, sigma=sigma, x=x)

        print("Run objective =", opt.fun)
        print("Run success   =", opt.success)
        print("Run message   =", opt.message)
        print_result(f"RESULT OF RUN {i}", best_result)

        if best_result["delta_petz"] < overall_obj:
            overall_obj = best_result["delta_petz"]
            overall_best = best_result

    print_result("OVERALL BEST POWELL RESULT", overall_best)
    return overall_best


if __name__ == "__main__":
    best = run_powell_multiple(
        kraus_seed=KRAUS_SEED,
        M=8,
        perturb_scale=0.01,
        maxiter=2500,
        xtol=1e-9,
        ftol=1e-9,
        random_seed=20260320,
    )