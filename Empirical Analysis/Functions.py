# ==========================================================
# FUNCTIONS FOR AML ESTIMATION, PREDICTIVE DENSITY CALCULATION, BIMODALITY MEASURE DETERMINATION AND PLOTS
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import gamma
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy import stats
from numdifftools import Hessian
from types import SimpleNamespace
import inspect
import pandas as pd
from PSY import cvPSYwmboot, locate, PSY

# =========================================================
# USEFUL FUNCTIONS
# =========================================================
# Integrate area under the curve
def integrate_area(y_vals, x_vals):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y_vals, x_vals)
    return np.trapz(y_vals, x_vals)

# Student-t PDF
def student_t_pdf(e, nu, sigma):
    c = gamma((nu + 1) / 2) / (gamma(nu / 2) * np.sqrt(nu * np.pi) * sigma)
    return c * ((1 + (e**2) / (nu * sigma**2)) ** (-(nu + 1) / 2))



# =========================================================
# RANDOM INITIALIZATION RULES
# =========================================================

def random_init(param_name, rng, scale):

    name = param_name.lower()

    if "phi" in name or "psi" in name:
        return rng.uniform(-.85, .85)

    elif "beta" in name or name.startswith("beta"):
        return rng.normal(0, .1)

    elif name == "sigma":
        return scale * rng.uniform(.5, 2.)

    elif name == "nu":
        return rng.choice([.8, 1., 1.8, 3., 6., 10.])

    else:
        return rng.uniform(-1, 1)




# =========================================================
# MLE DIAGNOSTICS (R^2, AIC, BIC, residuals)
# =========================================================

def _theta_dict(theta, param_names):
    return {name: theta[i] for i, name in enumerate(param_names)}


def _beta_from_theta(theta_dict):
    beta_keys = [
        name for name in theta_dict
        if name.lower().startswith("beta")
    ]
    if not beta_keys:
        return None
    return np.array([theta_dict[name] for name in beta_keys])


def _compute_mar_residuals(theta, y, X=None, param_names=None):
    td = _theta_dict(theta, param_names)
    phi = td["phi"]
    psi = td["psi"]
    beta = _beta_from_theta(td)

    e = (1 + phi * psi) * y[1:-1] - phi * y[:-2] - psi * y[2:]
    if beta is not None and X is not None:
        e = e - X[1:-1] @ beta
    return e



def _r_squared(residuals, y):
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    residuals = np.asarray(residuals, dtype=float)
    y = np.asarray(y, dtype=float)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0 or not np.isfinite(ss_res):
        return np.nan
    return 1.0 - ss_res / ss_tot



def _sample_acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([1.0])
    nlags = min(int(nlags), n - 1)
    denom = np.sum(x**2)
    if denom <= 0:
        return np.ones(nlags + 1)
    acf = np.empty(nlags + 1)
    acf[0] = 1.0
    for lag in range(1, nlags + 1):
        acf[lag] = np.sum(x[:-lag] * x[lag:]) / denom
    return acf


def _ljung_box_test(x, lags):
    """Ljung--Box Q test for residual autocorrelation."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    acf = _sample_acf(x, max(lags))
    stats_out = []
    pvals = []

    for h in lags:
        rho = acf[1 : h + 1]
        q = n * (n + 2) * np.sum((rho**2) / (n - np.arange(1, h + 1)))
        stats_out.append(float(q))
        pvals.append(float(1.0 - stats.chi2.cdf(q, df=h)))

    return stats_out, pvals


def _autocorrelation_tests(residuals, max_lag=None):
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    n = len(e)

    if n < 8:
        return {"n": n, "note": "Not enough observations for autocorrelation tests."}

    if max_lag is None:
        max_lag = min(20, n // 5)

    max_lag = max(1, min(max_lag, n - 2))
    lags = list(range(1, max_lag + 1))

    lb_stat, lb_p = _ljung_box_test(e, lags)

    return {
        "n": n,
        "max_lag": max_lag,
        "ljung_box": {
            "lags": lags,
            "statistic": lb_stat,
            "p_value": lb_p,
        },
        "acf": _sample_acf(e, max_lag).tolist(),
    }





def _print_autocorrelation_tests(tests, label="residuals"):
    print(f"\nAutocorrelation tests ({label}):")
    if "note" in tests:
        print(tests["note"])
        return

    lb = tests["ljung_box"]
    print(f"  n       = {tests['n']}")
    print("  Ljung-Box (H0: no autocorrelation up to lag h):")

    show_lags = sorted({
        1,
        5,
        10,
        tests["max_lag"],
    } & set(lb["lags"]))

    lag_to_idx = {lag: i for i, lag in enumerate(lb["lags"])}
    for lag in show_lags:
        i = lag_to_idx[lag]
        print(
            f"    lag {lag:2d}: statistic = {lb['statistic'][i]:.6f}, "
            f"p-value = {lb['p_value'][i]:.6g}"
        )


def _innovation_label(nu=None):
    if nu is None:
        return "Cauchy(0,1)"
    return f"Student-t (ν={float(nu):.2f})"


def _innovation_pdf(x, nu=None):
    """PDF of ε/σ under the fitted innovation (Cauchy or Student-t)."""
    x = np.asarray(x, dtype=float)
    if nu is None:
        return 1.0 / (np.pi * (1.0 + x**2))
    return stats.t.pdf(x, df=float(nu))


def _plot_acf_panel(ax, series, acf_lags, title):
    series = np.asarray(series, dtype=float)
    acf_vals = _sample_acf(series, acf_lags)
    lags = np.arange(len(acf_vals))
    n = len(series)
    conf = 1.96 / np.sqrt(n) if n > 0 else 0.0

    ax.stem(lags, acf_vals, basefmt=" ")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(conf, color="red", ls="--", lw=1)
    ax.axhline(-conf, color="red", ls="--", lw=1)
    ax.set_title(title)
    ax.set_xlabel("lag")
    ax.set_ylabel("acf")
    ax.grid(True, alpha=0.3)


def _plot_residual_diagnostics(residuals, standardized, title, acf_lags=20, nu=None):
    standardized = np.asarray(standardized, dtype=float)
    ref_label = _innovation_label(nu)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.ravel()

    ax[0].plot(residuals, lw=1)
    ax[0].axhline(0.0, color="black", lw=0.8)
    ax[0].set_title("Residuals")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$\hat{\varepsilon}_t$")
    ax[0].grid(True, alpha=0.3)

    ax[1].hist(
        standardized,
        bins=30,
        density=True,
        alpha=0.65,
        edgecolor="white",
        label=r"$\hat{\varepsilon}_t / \hat{\sigma}$",
    )
    x_grid = np.linspace(
        np.min(standardized) - 0.5,
        np.max(standardized) + 0.5,
        300,
    )
    ax[1].plot(
        x_grid,
        _innovation_pdf(x_grid, nu=nu),
        color="red",
        lw=2,
        label=ref_label,
    )
    ax[1].set_title("Histogram vs fitted innovation")
    ax[1].set_xlabel(r"$\hat{\varepsilon}_t / \hat{\sigma}$")
    ax[1].set_ylabel("density")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    _plot_acf_panel(
        ax[2],
        standardized,
        acf_lags,
        title="ACF of standardized residuals",
    )
    _plot_acf_panel(
        ax[3],
        standardized**2,
        acf_lags,
        title=r"ACF of $(\hat{\varepsilon}_t / \hat{\sigma})^2$",
    )

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _print_mle_diagnostics(
    label,
    loglik,
    n_params,
    n_obs,
    r2,
    residuals,
    sigma,
    nu=None,
):
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_obs) - 2 * loglik

    print(f"\n--- {label} diagnostics ---")
    print(f"Log-likelihood: {loglik:.6f}")
    print(f"AIC:            {aic:.6f}")
    print(f"BIC:            {bic:.6f}")
    print(f"R^2:            {r2:.6f}")

    std_residuals = residuals / sigma
    acf_std = _autocorrelation_tests(std_residuals)
    acf_std_sq = _autocorrelation_tests(std_residuals**2)

    _print_autocorrelation_tests(acf_std, label="standardized residuals")
    _print_autocorrelation_tests(
        acf_std_sq,
        label="squared standardized residuals",
    )

    _plot_residual_diagnostics(
        residuals,
        std_residuals,
        title=f"{label}: residual diagnostics",
        nu=nu,
    )

    return {
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "r2": r2,
        "residuals": residuals,
        "standardized_residuals": std_residuals,
        "autocorrelation_tests_standardized": acf_std,
        "autocorrelation_tests_standardized_squared": acf_std_sq,
    }


# =========================================================
# MULTISTART MLE
# =========================================================


def run_multistart(
    neg_loglik,
    param_names,
    bounds,
    y,
    X=None,
    n_starts=40,
    method="L-BFGS-B",
    seed=42
):

    rng = np.random.default_rng(seed)

    best_success = None
    best_fun_success = np.inf

    best_any = None
    best_fun_any = np.inf

    all_results = []

    # ---------------------------------------------
    # DATA-DEPENDENT SCALE
    # ---------------------------------------------

    scale = max(
        np.median(np.abs(np.diff(y))),
        .05
    )

    # ---------------------------------------------
    # FIXED INITIAL VALUES
    # ---------------------------------------------

    defaults = {
        "phi": .4,
        "psi": .5,
        "sigma": scale,
        "nu": 5.
    }

    theta_fixed = np.array([
        defaults[name.lower()]
        if name.lower() in defaults
        else 0.
        for name in param_names
    ], dtype=float)

    if len(bounds) != len(param_names):
        raise ValueError(
            "Number of bounds must match number of parameters."
        )

    # Check fixed initial values against bounds
    for value, (lower, upper) in zip(theta_fixed, bounds):

        if lower is not None and value < lower:
            raise ValueError("Fixed initial value below lower bound.")

        if upper is not None and value > upper:
            raise ValueError("Fixed initial value above upper bound.")

    # ---------------------------------------------
    # LIKELIHOOD ARGUMENTS
    # ---------------------------------------------

    if X is None:
        args = (y,)
    else:
        args = (y, X)

    # ---------------------------------------------
    # MULTISTART OPTIMIZATION
    # ---------------------------------------------

    for start in range(n_starts):

        # First start: fixed initial values
        if start == 0:
            theta0 = theta_fixed.copy()

        # Remaining starts: random initialization
        else:
            theta0 = np.array([
                random_init(name, rng, scale)
                for name in param_names
            ])

        # Check random initial values against bounds
        for j, (lower, upper) in enumerate(bounds):

            if lower is not None:
                theta0[j] = max(theta0[j], lower)

            if upper is not None:
                theta0[j] = min(theta0[j], upper)

        try:
            result = minimize(
                neg_loglik,
                theta0,
                args=args,
                method=method,
                bounds=bounds,
                options={
                    "maxiter": 5000,
                    "ftol": 1e-11,
                    "gtol": 1e-8,
                    "maxls": 50
                }
            )

        except (
            ValueError,
            FloatingPointError,
            OverflowError
        ):
            continue

        all_results.append(result)

        fun = result.fun

        if not np.isfinite(fun) or fun >= 1e90:
            continue

        # Best finite result
        if fun < best_fun_any:
            best_fun_any = fun
            best_any = result

        # Best successful result
        if result.success and fun < best_fun_success:
            best_fun_success = fun
            best_success = result

    # ---------------------------------------------
    # SELECT BEST OPTIMIZATION
    # ---------------------------------------------

    if best_success is not None:

        best_result = best_success

    elif best_any is not None:

        print(
            "Warning: no successful L-BFGS-B run; "
            "best finite result retained for diagnostics."
        )

        best_result = best_any

    else:
        raise ValueError(
            "No optimization result with a finite objective."
        )

    # ---------------------------------------------
    # HESSIAN + STANDARD ERRORS
    # ---------------------------------------------

    hess_fun = Hessian(
        lambda th: neg_loglik(th, *args),
        method="central",
        step=1e-5
    )

    Hess = hess_fun(best_result.x)

    cov = np.linalg.pinv(Hess)

    var_diag = np.diag(cov)

    if np.any(var_diag < 0):
        print(
            "Warning: negative variance estimates on diagonal; "
            "SEs clipped at 0."
        )

    se = np.sqrt(np.maximum(var_diag, 0.0))

    eigvals = np.linalg.eigvals(Hess)

    # ---------------------------------------------
    # CREATE *_hat PARAMETERS
    # ---------------------------------------------

    param_hats = {}

    for i, name in enumerate(param_names):

        clean_name = name.lower()

        clean_name = clean_name.replace("beta_", "b")

        param_hats[f"{clean_name}_hat"] = best_result.x[i]

    # Create beta_hat vector automatically
    beta_keys = [
        key for key in param_hats.keys()
        if key.startswith("b") and key != "beta_hat"
    ]

    if len(beta_keys) > 0:

        beta_hat = np.array([
            param_hats[key]
            for key in beta_keys
        ])

        param_hats["beta_hat"] = beta_hat

    # Convert to namespace object
    params = SimpleNamespace(**param_hats)

    # ---------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------

    print("Optimization success:",
          best_result.success)

    print("Successful starts:",
          sum(
              r.success and np.isfinite(r.fun) and r.fun < 1e90
              for r in all_results
          ),
          "/", n_starts)

    print("Log-likelihood:",
          -best_result.fun)

    print("\nEstimated parameters:")

    for i, name in enumerate(param_names):

        print(
            f"{name:15} = "
            f"{best_result.x[i]:>10.4f} ({se[i]:.4f})"
        )

    print("\nHessian condition number:")
    print(np.linalg.cond(Hess))

    print("\nEigenvalues of Hessian:")
    print(eigvals)

    # ---------------------------------------------
    # R^2, AIC, BIC, RESIDUAL DIAGNOSTICS
    # ---------------------------------------------

    has_nu = "nu" in param_names
    is_marx = X is not None

    n_obs = len(y) - 2
    n_params = len(param_names)

    theta_hat = best_result.x

    td_hat = _theta_dict(theta_hat, param_names)

    sigma_hat = td_hat["sigma"]

    residuals = _compute_mar_residuals(
        theta_hat, y, X, param_names
    )

    loglik_model = -float(best_result.fun)

    y_eff = y[1:-1]

    r2_mar = None
    r2_marx = None

    diagnostics = {}

    if is_marx:

        r2_marx = _r_squared(residuals, y_eff)

        print("\n" + "=" * 60)
        print("MLE diagnostics")
        print("=" * 60)

        diagnostics["marx"] = _print_mle_diagnostics(
            label="MARX model",
            loglik=loglik_model,
            n_params=n_params,
            n_obs=n_obs,
            r2=r2_marx,
            residuals=residuals,
            sigma=sigma_hat,
            nu=td_hat.get("nu") if has_nu else None,
        )

    else:

        r2_mar = _r_squared(residuals, y_eff)

        print("\n" + "=" * 60)
        print("MLE diagnostics")
        print("=" * 60)

        diagnostics["mar"] = _print_mle_diagnostics(
            label="MAR model",
            loglik=loglik_model,
            n_params=n_params,
            n_obs=n_obs,
            r2=r2_mar,
            residuals=residuals,
            sigma=sigma_hat,
            nu=td_hat.get("nu") if has_nu else None,
        )

    # ---------------------------------------------
    # RETURN EVERYTHING
    # ---------------------------------------------

    return {
        "params": params,
        "result": best_result,
        "success": best_success is not None,
        "se": se,
        "cov": cov,
        "Hessian": Hess,
        "eigenvalues": eigvals,
        "all_results": all_results,
        "n_converged": sum(
            r.success and np.isfinite(r.fun) and r.fun < 1e90
            for r in all_results
        ),
        "n_starts": n_starts,
        "r2_mar": r2_mar,
        "r2_marx": r2_marx if is_marx else None,
        "aic": (
            diagnostics["marx"]["aic"]
            if is_marx else diagnostics["mar"]["aic"]
        ),
        "bic": (
            diagnostics["marx"]["bic"]
            if is_marx else diagnostics["mar"]["bic"]
        ),
        "residuals": residuals,
        "diagnostics": diagnostics,
    }



# =========================================================
# UNIVERSAL PREDICTIVE DENSITY EVALUATOR
# =========================================================

def evaluate_predictive_density(
    predictive_density_fn,
    grid,
    y,
    t,
    params,
    X=None
):

    sig = inspect.signature(predictive_density_fn)

    kwargs = {}

    for name in sig.parameters.keys():

        # standard arguments
        if name == "y_next":
            kwargs[name] = grid

        elif name == "y":
            kwargs[name] = y

        elif name == "t":
            kwargs[name] = t

        elif name == "X":
            kwargs[name] = X

        # automatically inject *_hat parameters
        else:

            hat_name = f"{name}_hat"

            if hasattr(params, hat_name):

                kwargs[name] = getattr(params, hat_name)

    dens = predictive_density_fn(**kwargs)

    return np.asarray(dens, dtype=float)


# =========================================================
# PLOT PREDICTIVE DENSITY
# =========================================================

def plot_predictive_density(
    predictive_density_fn,
    grid,
    y,
    t,
    params,
    X=None,
    normalize=True,
    title="Predictive Density"
):

    # Compute density
    dens = evaluate_predictive_density(
        predictive_density_fn=predictive_density_fn,
        grid=grid,
        y=y,
        t=t,
        params=params,
        X=X
    )

    # Ensure positivity
    dens = np.maximum(dens, 0)

    # Normalize
    if normalize:

        area = integrate_area(dens, grid)

        if area > 0 and np.isfinite(area):

            dens = dens / area

    # Plot
    plt.figure(figsize=(10, 4))

    plt.plot(grid, dens, lw=2)

    plt.title(title)

    plt.xlabel(r"$y_{T+1}$")

    plt.ylabel("density")

    plt.grid(True, alpha=0.3)

    plt.show()

    return dens


# =========================================================
# PLOT PREDICTIVE DENSITY WITH CURRENT VALUE OF THE SERIES AND VALUES AT THE PEAKS
# =========================================================

def plot_predictive_density_ax(
    ax,
    predictive_density_fn,
    grid,
    y,
    t,
    params,
    X=None,
    normalize=True,
    title=None,
):
    # Compute density
    dens = evaluate_predictive_density(
        predictive_density_fn=predictive_density_fn,
        grid=grid,
        y=y,
        t=t,
        params=params,
        X=X,
    )

    # Ensure positivity
    dens = np.maximum(dens, 0)

    # Normalize
    if normalize:
        area = integrate_area(dens, grid)
        if area > 0 and np.isfinite(area):
            dens = dens / area

    # Plot density
    ax.plot(grid, dens, lw=1.5, color="black")

    # Current price
    current_price = y[t]

    # Red line
    ax.axvline(
        current_price,
        color="red",
        linewidth=1.5,
        label=r"$y_t$"
    )

    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.5
    )

    # Find density peaks
    peaks, _ = find_peaks(dens)

    # Sort peaks by height
    peaks = peaks[np.argsort(dens[peaks])[::-1]]

    # Keep maximum 2 peaks
    peaks = peaks[:2]

    # Blue lines
    for peak in peaks:
        peak_price = grid[peak]

        ax.axvline(
            peak_price,
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )

    # ---------------------------------------------------------
    # Labels ABOVE the plot
    # Shift labels only when they overlap
    # ---------------------------------------------------------

    current_x_offset = 0
    peak_x_offsets = {peak: 0 for peak in peaks}

    overlap_threshold = 30  # Screen pixels
    side_shift = 9       # Small shift in points

    # Current price position in screen coordinates
    current_x_screen = ax.transData.transform(
        (current_price, 0)
    )[0]

    # Check each peak against the current price
    for peak in peaks:
        peak_price = grid[peak]

        peak_x_screen = ax.transData.transform(
            (peak_price, 0)
        )[0]

        if abs(current_x_screen - peak_x_screen) < overlap_threshold:

            # Current price is left of the peak:
            # move red left and blue right to separate them.
            if current_price < peak_price:
                current_x_offset = -side_shift
                peak_x_offsets[peak] = side_shift

            # Current price is right of the peak:
            # move red right and blue left to separate them.
            else:
                current_x_offset = side_shift
                peak_x_offsets[peak] = -side_shift

    # Current price label (red)
    ax.annotate(
        f"{current_price:.2f}",
        xy=(current_price, 1),
        xycoords=("data", "axes fraction"),
        xytext=(current_x_offset, 5),
        textcoords="offset points",
        color="red",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

    # Peak labels (blue)
    for peak in peaks:
        peak_price = grid[peak]

        ax.annotate(
            f"{peak_price:.2f}",
            xy=(peak_price, 1),
            xycoords=("data", "axes fraction"),
            xytext=(peak_x_offsets[peak], 5),
            textcoords="offset points",
            color="blue",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(
        title or "",
        fontsize=12,
        pad=25,
    )

    ax.set_xlabel(r"$y_{t+1}$", fontsize=10)
    ax.set_ylabel("densities", fontsize=10)

    return dens


# =========================================================
# PLOT PREDICTIVE DENSITY AT AND AROUND POINTS OF INTEREST
# =========================================================

def plot_predictive_density_simple(
    ax,
    predictive_density_fn,
    grid,
    y,
    t,
    params,
    X=None,
    normalize=True,
    title=None,
    ):

    # Compute density

    dens = evaluate_predictive_density(
    predictive_density_fn=predictive_density_fn,
    grid=grid,
    y=y,
    t=t,
    params=params,
    X=X,
    )

    # Ensure positivity

    dens = np.maximum(dens, 0)

    # Normalize

    if normalize:

        area = integrate_area(dens, grid)

        if area > 0 and np.isfinite(area):

            dens = dens / area

    # Plot on existing axis

    ax.plot(grid, dens, lw=1.5)

    ax.set_title(title or f"$t={t}$", fontsize=9)

    ax.grid(True, alpha=0.3)

       # Current price
    current_price = y[t]

    # Red line
    ax.axvline(
        current_price,
        color="red",
        linewidth=1.5,
    )

        # Current price label
    ax.annotate(
        f"{current_price:.2f}",
        xy=(current_price, -0.07),
        xycoords=("data", "axes fraction"),
        xytext=(0, 5),
        textcoords="offset points",
        color="red",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    return dens




# =========================================================
# BIMODALITY MEASURE
# =========================================================

def bimodality_measure(
    t0,
    predictive_density_fn,
    grid,
    params,
    y,
    X=None,
    prominence_factor=0.02
):

    # Compute predictive density
    dens = evaluate_predictive_density(
        predictive_density_fn=predictive_density_fn,
        grid=grid,
        y=y,
        t=t0,
        params=params,
        X=X
    )

    dens = np.maximum(dens, 0)

    # Normalize
    area = integrate_area(dens, grid)

    if area <= 0 or not np.isfinite(area):

        return 0.0

    dens /= area

    # Find peaks
    peaks, _ = find_peaks(
        dens,
        prominence=prominence_factor * dens.max()
    )

    if len(peaks) < 2:

        return 0.0

    # Two highest peaks
    top2 = peaks[np.argsort(dens[peaks])[-2:]]

    p1, p2 = np.sort(top2)

    h1, h2 = dens[p1], dens[p2]

    # Valley between peaks
    valley_idx = p1 + np.argmin(dens[p1:p2+1])

    valley_h = dens[valley_idx]

    # Bimodality score
    b = (
        min(h1, h2) / max(h1, h2)
    ) * (
        1 - valley_h / min(h1, h2)
    )

    return np.clip(b, 0, 1)


# =========================================================
# COMPUTE BIMODALITY SERIES
# =========================================================

def compute_bimodality_series(
    t_range,
    predictive_density_fn,
    grid,
    params,
    y,
    X=None,
    prominence_factor=0.02
):

    return np.array([

        bimodality_measure(
            t0=t,
            predictive_density_fn=predictive_density_fn,
            grid=grid,
            params=params,
            y=y,
            X=X,
            prominence_factor=prominence_factor
        )

        for t in t_range
    ])


# =========================================================
# PLOT BIMODALITY
# =========================================================

def plot_bimodality(
    times,
    bimodality_series,
    title="Bimodality Measure"
):

    plt.figure(figsize=(10, 4))

    plt.plot(times, bimodality_series)

    plt.xlabel("t")

    plt.ylabel("Bimodality")

    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.show()


# =========================================================
# PLOT BIMODALITY + OBSERVED SERIES
# =========================================================
def plot_bimodality_and_series(
    times,
    y_values,
    bimodality_series,
    y_limits,
    title=None,
    psy=None
):
    fig, ax1 = plt.subplots(figsize=(9, 4))

    if psy is not None:
        for start, end in psy["bubbles"][0.95]["episodes"]:
            ax1.axvspan(
                start,
                end,
                color="orange",
                alpha=0.25,
                zorder=0
            )

    ax1.plot(
        times,
        y_values,
        color="black",
        zorder=2
    )

    ax1.set_ylabel(
        r"$y_t$",
        color="black"
    )
    ax1.set_ylim(*y_limits)

    ax1.set_xlim(times[0], times[-1])

    ax2 = ax1.twinx()

    ax2.plot(
        times,
        bimodality_series,
        color="red",
        zorder=3
    )

    ax2.set_ylabel(
        "Bimodality",
        color="red"
    )
    ax2.tick_params(axis="y", colors="red")


    y_min, y_max = y_limits

    ratio = abs(y_min) / (y_max - y_min)

    bmax = np.nanmax(
        bimodality_series
    ) * 1.2

    if not np.isfinite(bmax) or bmax <= 0:
        bmax = 1.0

    bmin = -ratio * bmax / (1 - ratio)

    ax2.set_ylim(
        bmin,
        bmax
    )


    if psy is not None:
        ax1.plot(
            [],
            [],
            color="orange",
            alpha=0.25,
            linewidth=8,
            label="PSY"
        )

        ax1.legend(
            loc="upper left"
        )

    plt.title(title)
    plt.show()


# =========================================================
# PLOT BIMODALITY + OBSERVED SERIES + PSY (+ DIAMONDS AT POINTS OF INTEREST)
# =========================================================

def plot_bimodality_and_series_2(
    times,
    y_values,
    bimodality_series,
    y_limits,
    title=None,
    psy=None,
    marker_points=None
):
    fig, ax1 = plt.subplots(figsize=(9, 4))

    times_arr = np.asarray(times)
    y_arr = np.asarray(y_values, dtype=float)
    bimodality_arr = np.asarray(bimodality_series, dtype=float)

    # Shade PSY bubble episodes
    if psy is not None:
        for start, end in psy["bubbles"][0.95]["episodes"]:
            ax1.axvspan(
                start, end,
                color="orange",
                alpha=0.25,
                zorder=0
            )

    # Black series
    ax1.plot(
        times_arr,
        y_arr,
        color="black",
        zorder=2
    )
    ax1.set_ylabel(r"$y_t$", color="black")
    ax1.set_ylim(*y_limits)
    ax1.set_xlim(times_arr[0], times_arr[-1])

    # Red bimodality series on secondary axis
    ax2 = ax1.twinx()

    ax2.plot(
        times_arr,
        bimodality_arr,
        color="red",
        lw=1.2,
        zorder=3
    )
    ax2.set_ylabel("Bimodality", color="red")
    ax2.tick_params(axis="y", colors="red")

    # Scale the secondary axis
    y_min, y_max = y_limits
    ratio = abs(y_min) / (y_max - y_min)

    bmax = np.nanmax(bimodality_arr) * 1.2
    if not np.isfinite(bmax) or bmax <= 0:
        bmax = 1.0

    bmin = -ratio * bmax / (1 - ratio)
    ax2.set_ylim(bmin, bmax)

    # Mark selected points on the black series,
    # but draw the markers above the red line.
    if marker_points is not None:
        idx = np.asarray(marker_points, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(times_arr))]
        idx = idx[idx < len(y_arr)]
        idx = idx[np.isfinite(y_arr[idx])]

        # Convert black-series y-coordinates to ax2 coordinates
        y_markers = ax2.get_ylim()[0] + (
            (y_arr[idx] - ax1.get_ylim()[0])
            / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        ) * (
            ax2.get_ylim()[1] - ax2.get_ylim()[0]
        )

        ax2.scatter(
            times_arr[idx],
            y_markers,
            marker="D",
            s=30,
            color="black",
            edgecolors="black",
            linewidths=1.0,
            zorder=10
        )

    # Legend
    handles = []
    labels = []

    if psy is not None:
        handles.append(
            Patch(color="orange", alpha=0.25, label="PSY")
        )
        labels.append("PSY")

    if marker_points is not None:
        handles.append(
            plt.Line2D(
                [], [],
                marker="D",
                linestyle="None",
                color="black",
                markersize=7
            )
        )

    if handles:
        ax1.legend(handles, labels, loc="upper left")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================================================
# EFFECT OF X IN BIMODALITY 
# =========================================================

def shock_analysis(
    var_idx,
    X,
    y,
    params,
    predictive_density_fn,
    grid_y,
    n_std,
    n_values,
    t_last=None,
    var_names=None,
    n_plot=10,
    n_cols=5,
    prominence_factor=0.02
):

    # -------------------------------------------------
    # SETUP
    # -------------------------------------------------

    if t_last is None:
        t_last = len(y) - 1

    if var_names is None:
        var_names = [
            f"x{i}" for i in range(X.shape[1])
        ]

    var_name = var_names[var_idx]

    x_center = X[t_last, var_idx]
    x_std = np.std(X[:, var_idx])

    x_grid = np.linspace(
        x_center - n_std * x_std,
        x_center + n_std * x_std,
        n_values
    )

    densities = []
    bimods = []

    # -------------------------------------------------
    # LOOP OVER SHOCKS
    # -------------------------------------------------

    for xv in x_grid:

        X_mod = X.copy()
        X_mod[t_last, var_idx] = xv

        # predictive density (UNIVERSAL)
        dens = evaluate_predictive_density(
            predictive_density_fn=predictive_density_fn,
            grid=grid_y,
            y=y,
            t=t_last,
            params=params,
            X=X_mod
        )

        dens = np.maximum(dens, 0)

        area = integrate_area(dens, grid_y)

        if area > 0 and np.isfinite(area):
            dens = dens / area

        densities.append(dens)

        # bimodality (UNIVERSAL)
        b = bimodality_measure(
            t0=t_last,
            predictive_density_fn=predictive_density_fn,
            grid=grid_y,
            params=params,
            y=y,
            X=X_mod,
            prominence_factor=prominence_factor
        )

        bimods.append(b)

    densities = np.array(densities)
    bimods = np.array(bimods)

    # -------------------------------------------------
    # PLOT 1: SELECTED DENSITIES
    # -------------------------------------------------

    plt.figure(figsize=(10, 5))

    idxs = np.linspace(0, len(x_grid) - 1, n_plot).astype(int)

    for i in idxs:
        plt.plot(
            grid_y,
            densities[i],
            label=f"{var_name}={x_grid[i]:.2f}"
        )

    plt.title(f"Predictive density under different {var_name}")
    plt.xlabel(r"$y_{T+1}$")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------------------------------------
    # PLOT 2: GRID OF DENSITIES
    # -------------------------------------------------

    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, i in zip(axes, idxs):

        ax.plot(grid_y, densities[i])
        ax.set_title(f"{var_name}={x_grid[i]:.2f}")

        ax.set_xlabel(r"$y_{T+1}$")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)

    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Predictive densities under different {var_name}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # PLOT 3: BIMODALITY
    # -------------------------------------------------

    plt.figure(figsize=(8, 4))

    plt.plot(x_grid, bimods, marker="o")

    plt.axvline(
        x_center,
        linestyle="--",
        color="black",
        label="observed value"
    )

    plt.title(f"Bimodality vs {var_name}")
    plt.xlabel(var_name)
    plt.ylabel("Bimodality")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------------------------------------
    # RETURN
    # -------------------------------------------------

    return {
        "x_grid": x_grid,
        "densities": densities,
        "bimodality": np.array(bimods)
    }






# =========================================================
# GET PSY FOR TIME SERIES
# =========================================================

def run_psy(
    y,
    dates,
    swindow0=None,
    IC=0,
    adflag=0,
    Tb=None,
    nboot=1999
):
    y = np.asarray(y, dtype=float)
    dates = pd.to_datetime(dates)

    if swindow0 is None:
        T = len(y)
        swindow0 = int(
            np.floor(
                T * (
                    0.01
                    + 1.8 / np.sqrt(T)
                )
            )
        )

    if Tb is None:
        Tb = len(y)

    bsadf = PSY(
        y,
        swindow0=swindow0,
        IC=IC,
        adflag=adflag
    )

    cv = cvPSYwmboot(
        y,
        swindow0=swindow0,
        IC=IC,
        adflag=adflag,
        Tb=Tb,
        nboot=nboot
    )

    valid = np.isfinite(bsadf)

    bsadf_valid = bsadf[valid]
    dates_valid = dates[valid]

    critical_values = {
        0.90: cv[0],
        0.95: cv[1],
        0.99: cv[2]
    }

    bubbles = {}

    for level, critical_value in critical_values.items():
        indicator = (
            bsadf_valid > critical_value
        ).astype(int)

        episodes = locate(
            indicator,
            dates_valid
        )

        bubbles[level] = {
            "critical_value": critical_value,
            "indicator": indicator,
            "episodes": episodes
        }

    return {
        "bsadf": bsadf_valid,
        "dates": dates_valid,
        "cv": cv,
        "critical_values": critical_values,
        "bubbles": bubbles,
        "swindow0": swindow0,
        "IC": IC,
        "adflag": adflag,
        "Tb": Tb,
        "nboot": nboot
    }
