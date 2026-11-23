# ==========================================================
# MLE AND PREDICTIVE DENSITY FUNCTIONS FOR
# MAR(1,1) WITH CAUCHY AND STYUDENT-T INNOVATIONS AND FOR 
# MARX(1,1,Q)  WITH CAUCHY AND STYUDENT-T INNOVATIONS
# ==========================================================

import numpy as np
from scipy.special import gamma
from scipy.special import gammaln


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
#  MLE Function MAR(1,1) Cauchy
# =========================================================
def neg_loglik_mar11_cauchy(theta, y):

    # theta = [phi, psi, sigma]
    phi, psi, sigma = theta

    # residuals
    e = (1 + phi * psi) * y[1:-1] \
        - phi * y[:-2] \
        - psi * y[2:] \

    # Cauchy log-likelihood
    ll = np.sum(
        -np.log(np.pi * sigma)
        - np.log(1 + (e / sigma) ** 2)
    )

    return -ll


# =========================================================
#  Predictive Density MAR(1,1) Cauchy
# =========================================================
def mar_cauchy_pred_density(y_next, y, t, phi, psi, sigma):

    y = np.asarray(y, dtype=float)
    y_next = np.asarray(y_next, dtype=float)


    y_t = y[t]
    y_tm1 = y[t - 1]

    # calculate u_t
    u_t = y_t - phi * y_tm1

    # calculate y_t+1
    z = y_next - phi * y_t

 
    term1 = 1.0 / (np.pi * sigma)

    term2 = sigma**2 / ((u_t - psi * z )**2 + sigma**2)

    term3 = (sigma**2 + (1 - abs(psi))**2 * u_t**2) / (
        sigma**2 + (1 - abs(psi))**2 * z**2
    )

    density = term1 * term2 * term3

    return density


# =========================================================
#  MLE Function MARX Cauchy
# =========================================================
def neg_loglik_marx_cauchy(theta, y, X):

    # theta = [phi, psi, beta..., sigma]
    phi = theta[0]
    psi = theta[1]
    sigma = theta[-1]

    beta = np.asarray(theta[2:-1], dtype=float)

    # Optional safety check
    if X.shape[1] != len(beta):
        raise ValueError(
            f"X has {X.shape[1]} columns but theta contains {len(beta)} beta coefficients."
        )

    # residuals
    e = (
        (1 + phi * psi) * y[1:-1]
        - phi * y[:-2]
        - psi * y[2:]
        - X[1:-1] @ beta
    )

    # Cauchy log-likelihood
    ll = np.sum(
        -np.log(np.pi * sigma)
        - np.log(1 + (e / sigma) ** 2)
    )

    return -ll


# =========================================================
#  Predictive Density MARX Cauchy
# =========================================================
def marx_cauchy_pred_density(y_next, y, X, t, phi, psi, beta, sigma):

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    y_next = np.asarray(y_next, dtype=float)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Input validation
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if X.shape[1] != len(beta):
        raise ValueError("beta must match the number of columns in X.")

    if not (1 <= t < len(y)):
        raise ValueError("t must satisfy 1 <= t < len(y).")

    if abs(psi) >= 1:
        raise ValueError("The formula requires |psi| < 1.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    # Final observed time index
    T = len(y) - 1

    # Observed values
    y_t = y[t]
    y_tm1 = y[t - 1]

    # Causal component at time t
    u_t = y_t - phi * y_tm1

    # Candidate component at time t+1
    z = y_next - phi * y_t

    # Exogenous contribution beta'X
    bX = X @ beta

    # ---------------------------------------
    # EXOGENOUS INFINITE SUMS
    # ---------------------------------------

    if t < T:
        # IN-SAMPLE:
        # Use realised X values only.
        # Truncate the sums at the final
        # observed time T.

        m_t = sum(
            psi**j * bX[t + j]
            for j in range(T - t + 1)
        )

        m_tp1 = sum(
            psi**j * bX[t + 1 + j]
            for j in range(T - t)
        )

    else:
        # END-OF-SAMPLE:
        # Assume X[T+j] = X[T] for j >= 1.
        # Evaluate the geometric sums exactly.

        m_t = bX[T] / (1.0 - psi)
        m_tp1 = m_t

    # ---------------------------------------
    # CAUCHY PREDICTIVE DENSITY
    # ---------------------------------------

    # First term: normalization
    term1 = 1.0 / (np.pi * sigma)

    # Second term: innovation density
    term2 = sigma**2 / (
        (u_t - psi * z - bX[t])**2 + sigma**2
    )

    # Third term: marginal-density ratio
    a = (1.0 - abs(psi))**2

    term3 = (
        sigma**2 + a * (u_t - m_t)**2
    ) / (
        sigma**2 + a * (z - m_tp1)**2
    )

    # Predictive density
    density = term1 * term2 * term3

    return density




# =========================================================
#  MLE Function MAR(1,1) Student-t
# =========================================================
def neg_loglik_student_mar(theta, y):

    phi, psi, sigma, nu = theta

    e = (1 + phi * psi) * y[1:-1] \
        - phi * y[:-2] \
        - psi * y[2:] 

    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
    )

    ll = np.sum(c - ((nu + 1) / 2) * np.log(1 + (e / sigma) ** 2 / nu))

    return -ll

# =========================================================
#  Predictive Density MAR(1,1) Student-t
# =========================================================
def mar_studentt_pred_density(y_next, y, t, phi, psi, sigma, nu):

    y = np.asarray(y, dtype=float)
    y_next = np.asarray(y_next, dtype=float)


    y_t = y[t]
    y_tm1 = y[t-1]

    # u_t = y_t - phi y_{t-1}
    u_t = y_t - phi * y_tm1

    
    # Historical sample (1,...,t)
    u_sample = y[1:t+1] - phi * y[:t]   # u_1,...,u_t
 


    # Candidate future u_{t+1}
    u_next = y_next - phi * y_t   # shape (N,)

    
    # First term: g(u_T - ψ u_{T+1})
    term1 = student_t_pdf(
        u_t - psi * u_next,
        nu, sigma
    )

    
    # Numerator:
    # sum_t g(u_{T+1} - ψ u_t )
    num = np.sum(
        student_t_pdf(
            u_next[:, None]
            - psi * u_sample[None, :],  
            nu,
            sigma
        ),
        axis=1
    )


    # Denominator:
    # sum_t g(u_T - ψ u_t )
    den = np.sum(
        student_t_pdf(
            u_t - psi * u_sample,
            nu,
            sigma
        )
    )

    den = max(den, 1e-12)

    return term1 * (num / den)


# =========================================================
#  MLE Function MARX Student-t
# =========================================================
def neg_loglik_student_marx(theta, y, X):

    # theta = [phi, psi, beta..., sigma, nu]
    phi = theta[0]
    psi = theta[1]
    sigma = theta[-2]
    nu = theta[-1]

    beta = np.asarray(theta[2:-2], dtype=float)

    # Optional safety check
    if X.shape[1] != len(beta):
        raise ValueError(
            f"X has {X.shape[1]} columns but theta contains {len(beta)} beta coefficients."
        )

    e = (
        (1 + phi * psi) * y[1:-1]
        - phi * y[:-2]
        - psi * y[2:]
        - X[1:-1] @ beta
    )

    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
    )

    ll = np.sum(
        c - ((nu + 1) / 2) * np.log(1 + (e / sigma) ** 2 / nu)
    )

    return -ll



# =========================================================
#  Predictive Density MARX Student-t
# =========================================================
def marx_studentt_pred_density(y_next, y, X, t, phi, psi, beta, sigma, nu):

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    y_next = np.asarray(y_next, dtype=float)
    beta = np.asarray(beta, dtype=float)


    y_t = y[t]
    y_tm1 = y[t-1]
    X_t = X[t]

    # u_t = y_t - phi y_{t-1}
    u_t = y_t - phi * y_tm1

    
    # Historical sample (1,...,t)
    u_sample = y[1:t+1] - phi * y[:t]   # u_1,...,u_t


    # Future regressor X_{t+1}
    # For last point: set X_{t+1} = X_t
    if t + 1 < len(X):
        X_tp1 = X[t+1]
    else:
        X_tp1 = X_t   

    # Precompute dot products
    X_t_beta = X_t @ beta
    X_tp1_beta = X_tp1 @ beta


    # Candidate future u_{t+1}
    u_next = y_next - phi * y_t   # shape (N,)

    
    # First term: g(u_T - ψ u_{T+1} - β X_T)
    term1 = student_t_pdf(
        u_t - psi * u_next - X_t_beta,
        nu, sigma
    )

    
    # Numerator:
    # sum_t g(u_{T+1} - ψ u_t - β X_{T+1})
    num = np.sum(
        student_t_pdf(
            u_next[:, None]
            - psi * u_sample[None, :]
            - X_tp1_beta,  
            nu,
            sigma
        ),
        axis=1
    )


    # Denominator:
    # sum_t g(u_T - ψ u_t - β X_t)
    den = np.sum(
        student_t_pdf(
            u_t - psi * u_sample - X_t_beta,
            nu,
            sigma
        )
    )

    den = max(den, 1e-12)

    return term1 * (num / den)